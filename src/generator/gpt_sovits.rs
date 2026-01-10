use crate::audio::{AudioBackend, AudioStreamSegment, CancelScope, CancelToken};
use super::{Result, SynthesizedAudio, TtsEngine, TtsMetrics};
use anyhow::anyhow;
use async_trait::async_trait;
#[cfg(feature = "tts-worker")]
use bytes::Bytes;
use gpt_sovits_rs::gsv;
use gpt_sovits_rs::tch;
use gpt_sovits_rs::text::G2PConfig;
use jieba_rs::Jieba;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "tts-worker")]
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{info, warn};

use crate::internal::{env, model_locator, timing};

// 首段之后的默认流式分块 token 目标（逐步增大以提升连贯性/吞吐）：
// 25：第二段初始块，更快出声
// 50：中段块，兼顾速度与连贯
// 100：长段块，提高整体合成效率
// 首段可通过 GSV_FIRST_CHUNK_TOKENS 调小，以降低首播时延。
const DEFAULT_CHUNK_TOKEN_NUMS: [i64; 3] = [25, 50, 100];
const DEFAULT_PLAY_CHUNK_SAMPLES: usize = 2048;
const DEFAULT_BASE_DIR: &str = "v2pro";

pub struct GptSovitsTts {
    inner: Arc<StdMutex<Inner>>,
    cancel: CancelToken,
    audio: Arc<dyn AudioBackend>,
    first_call: AtomicBool,
}

/// Dynamic first-chunk token policy.
#[derive(Debug, Clone)]
pub struct GptSovitsChunkPolicy {
    pub dynamic: bool,
    pub short_chars: usize,
    pub mid_chars: usize,
    pub short_tokens: i64,
    pub mid_tokens: i64,
}

impl Default for GptSovitsChunkPolicy {
    fn default() -> Self {
        Self {
            dynamic: true,
            short_chars: 12,
            mid_chars: 24,
            short_tokens: 6,
            mid_tokens: 8,
        }
    }
}

impl GptSovitsChunkPolicy {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.dynamic = env::bool01("GSV_FIRST_CHUNK_DYNAMIC", cfg.dynamic);
        if let Some(parsed) = env::get::<usize>("GSV_FIRST_CHUNK_SHORT_CHARS") {
            cfg.short_chars = parsed.clamp(1, 200);
        }
        if let Some(parsed) = env::get::<usize>("GSV_FIRST_CHUNK_MID_CHARS") {
            cfg.mid_chars = parsed.clamp(cfg.short_chars, 400);
        }
        cfg.short_tokens =
            env::i64_clamped("GSV_FIRST_CHUNK_SHORT_TOKENS", cfg.short_tokens, 3, 25);
        cfg.mid_tokens = env::i64_clamped("GSV_FIRST_CHUNK_MID_TOKENS", cfg.mid_tokens, 3, 25);
        cfg
    }
}

/// GPT-SoVITS CUDA backend configuration.
#[derive(Debug, Clone)]
pub struct GptSovitsConfig {
    pub g2p_en_path: PathBuf,
    pub g2pw_path: PathBuf,
    pub cn_bert_path: PathBuf,
    pub ssl_model_path: PathBuf,
    pub t2s_model_path: PathBuf,
    pub vits_model_path: PathBuf,
    pub ref_wav_path: PathBuf,
    pub ref_text: String,
    pub fp16: bool,
    pub top_k: i64,
    pub top_k_first: i64,
    pub max_cut_token: i64,
    pub chunk_token_nums: Vec<i64>,
    pub chunk_policy: GptSovitsChunkPolicy,
    pub log_text_metrics: bool,
    pub jieba_bench: bool,
}

struct Inner {
    g2p: gpt_sovits_rs::text::G2p,
    speaker: gsv::SpeakerV2Pro,
    ref_params: (tch::Tensor, tch::Tensor, tch::Tensor), // (prompts, refer, sv_emb)
    ref_seq: tch::Tensor,
    ref_bert: tch::Tensor,
    top_k: i64,
    top_k_first: i64,
    max_cut_token: i64,
    chunk_token_nums: Vec<i64>,
    chunk_policy: GptSovitsChunkPolicy,
    fp16: bool,
    log_text_metrics: bool,
    jieba: Option<Jieba>,
}

fn to_float(t: tch::Tensor) -> tch::Tensor {
    if t.kind() == tch::Kind::Float {
        t
    } else {
        t.to_kind(tch::Kind::Float)
    }
}

fn to_half(t: tch::Tensor) -> tch::Tensor {
    if t.kind() == tch::Kind::Half {
        t
    } else {
        to_float(t).internal_cast_half(false)
    }
}

fn build_inner(config: &GptSovitsConfig, device: tch::Device) -> Result<Inner> {
    let fp16 = config.fp16 && device.is_cuda();
    info!("Loading GPT-SoVITS models on {:?} (fp16={})", device, fp16);

    let g2p_conf = G2PConfig::new(config.g2p_en_path.to_string_lossy().to_string()).with_chinese(
        config.g2pw_path.to_string_lossy().to_string(),
        config.cn_bert_path.to_string_lossy().to_string(),
    );
    let g2p = g2p_conf.build(device)?;

    let ssl = gsv::SSL::new(&config.ssl_model_path.to_string_lossy(), device)?;
    let t2s = gsv::T2S::new(&config.t2s_model_path.to_string_lossy(), device)?;
    let vits = gsv::Vits::new(&config.vits_model_path.to_string_lossy(), device)?;
    let speaker = gsv::SpeakerV2Pro::new("default", Arc::new(t2s), Arc::new(vits), Arc::new(ssl));

    let file = std::fs::File::open(&config.ref_wav_path)?;
    let (head, mut ref_audio_samples) = wav_io::read_from_file(file)?;
    if head.channels >= 2 {
        ref_audio_samples = wav_io::utils::stereo_to_mono(ref_audio_samples);
    }
    if head.sample_rate != 32_000 {
        ref_audio_samples =
            wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32_000);
    }
    let mut peak = 0f32;
    for &s in &ref_audio_samples {
        peak = peak.max(s.abs());
    }
    info!(
        "Reference audio: sr={}Hz, ch={} -> mono, samples={}, peak={:.4}",
        head.sample_rate,
        head.channels,
        ref_audio_samples.len(),
        peak
    );

    let ref_audio_32k = tch::Tensor::from_slice(&ref_audio_samples)
        .to_kind(tch::Kind::Float)
        .to_device(device)
        .unsqueeze(0);
    let ref_audio_32k = if fp16 { to_half(ref_audio_32k) } else { ref_audio_32k };
    if config.log_text_metrics {
        info!(
            "gpt-sovits dtype: ref_audio_32k={:?} device={:?}",
            ref_audio_32k.kind(),
            ref_audio_32k.device()
        );
    }

    let _g = tch::no_grad_guard();
    let (prompts, refer, sv_emb) = speaker.pre_handle_ref(ref_audio_32k)?;
    if config.log_text_metrics {
        info!(
            "gpt-sovits dtype: prompts={:?} refer={:?} sv_emb={:?}",
            prompts.kind(),
            refer.kind(),
            sv_emb.kind()
        );
    }

    let (ref_seq, ref_bert) = gpt_sovits_rs::text::get_phone_and_bert(&g2p, &config.ref_text)?;
    let ref_bert = if fp16 { to_half(ref_bert) } else { to_float(ref_bert) };
    if config.log_text_metrics {
        info!(
            "gpt-sovits dtype: ref_seq={:?} ref_bert={:?}",
            ref_seq.kind(),
            ref_bert.kind()
        );
    }

    Ok(Inner {
        g2p,
        speaker,
        ref_params: (prompts, refer, sv_emb),
        ref_seq,
        ref_bert,
        top_k: config.top_k,
        top_k_first: config.top_k_first,
        max_cut_token: config.max_cut_token,
        chunk_token_nums: config.chunk_token_nums.clone(),
        chunk_policy: config.chunk_policy.clone(),
        fp16,
        log_text_metrics: config.log_text_metrics,
        jieba: if config.jieba_bench { Some(Jieba::new()) } else { None },
    })
}

impl GptSovitsConfig {
    pub fn from_dir(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir: PathBuf = base_dir.into();

        let g2p_en_path = model_locator::first_existing_file_rel(
            &base_dir,
            "g2p_en model (mini-bart-g2p)",
            &["mini-bart-g2p.pt", "resource/mini-bart-g2p.pt"],
        )?;
        let g2pw_path =
            model_locator::first_existing_file_rel(&base_dir, "g2pw model", &["g2pw_model.pt", "g2pw.pt"])?;
        let cn_bert_path = model_locator::first_existing_file_rel(
            &base_dir,
            "cn_bert model",
            &["bert_model.pt", "bert.pt"],
        )?;
        let ssl_model_path = model_locator::first_existing_file_rel(
            &base_dir,
            "ssl model",
            &["ssl_model.pt", "ssl.pt", "resource/ssl_model.pt"],
        )?;

        let combined_path = base_dir.join("gpt_sovits_v2pro.cuda.pt");
        let t2s_model_path = if base_dir.join("t2s.pt").exists() {
            base_dir.join("t2s.pt")
        } else if combined_path.exists() {
            combined_path.clone()
        } else {
            model_locator::first_existing_file_rel(&base_dir, "t2s model", &["t2s.pt", "t2s.cpu.pt"])?
        };

        let vits_model_path = if base_dir.join("vits.pt").exists() {
            base_dir.join("vits.pt")
        } else if combined_path.exists() {
            combined_path
        } else {
            model_locator::first_existing_file_rel(&base_dir, "vits model", &["vits.pt", "vits.cpu.pt"])?
        };

        let ref_wav_path = model_locator::first_existing_file_rel(
            &base_dir,
            "ref.wav",
            &["ref.wav", "ref_32k.wav", "ref32k.wav"],
        )?;
        let ref_text_path = model_locator::first_existing_file_rel(&base_dir, "ref.txt", &["ref.txt"])?;
        let ref_text = std::fs::read_to_string(&ref_text_path)?
            .trim_start_matches('\u{feff}')
            .trim()
            .to_string();

        let chunk_token_nums = std::iter::once(10)
            .chain(DEFAULT_CHUNK_TOKEN_NUMS)
            .collect();

        Ok(Self {
            g2p_en_path,
            g2pw_path,
            cn_bert_path,
            ssl_model_path,
            t2s_model_path,
            vits_model_path,
            ref_wav_path,
            ref_text,
            fp16: true,
            top_k: 15,
            top_k_first: 12,
            max_cut_token: 25,
            chunk_token_nums,
            chunk_policy: GptSovitsChunkPolicy::default(),
            log_text_metrics: false,
            jieba_bench: false,
        })
    }

    pub fn from_env() -> Result<Self> {
        let base_dir = env::string("GSV_MODEL_DIR").unwrap_or_else(|| DEFAULT_BASE_DIR.to_string());
        let mut cfg = Self::from_dir(base_dir)?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(raw) = std::env::var("GSV_FP16") {
            let normalized = raw.trim().to_lowercase();
            let parsed = match normalized.as_str() {
                "1" | "true" | "yes" | "y" | "on" => Some(true),
                "0" | "false" | "no" | "n" | "off" => Some(false),
                _ => None,
            };
            match parsed {
                Some(value) => {
                    self.fp16 = value;
                    info!("GSV_FP16={} -> fp16={}", raw.trim(), value);
                    if !value {
                        info!(
                            "GSV_FP16=0: using fp32 inputs (note: some CUDA TorchScript models may be half-only; enable RUST_LOG=gpt_sovits_rs=debug for details)"
                        );
                    }
                }
                None => {
                    warn!(
                        "GSV_FP16={:?} is invalid; expected 1/0/true/false; using default fp16={}",
                        raw,
                        self.fp16
                    );
                }
            }
        }

        if let Some(parsed) = env::get::<i64>("GSV_TOP_K") {
            self.top_k = parsed.clamp(1, 50);
        }
        if let Some(parsed) = env::get::<i64>("GSV_FIRST_TOP_K") {
            self.top_k_first = parsed.clamp(1, self.top_k);
        }
        if self.top_k_first != self.top_k {
            info!("First segment top_k: {} (default={})", self.top_k_first, self.top_k);
        }

        if let Some(parsed) = env::get::<i64>("GSV_FIRST_CHUNK_TOKENS") {
            let first_chunk_tokens = parsed.clamp(3, 25);
            if self.chunk_token_nums.is_empty() {
                self.chunk_token_nums = vec![first_chunk_tokens];
            } else {
                self.chunk_token_nums[0] = first_chunk_tokens;
            }
            if first_chunk_tokens != 10 {
                info!("First chunk token target: {}", first_chunk_tokens);
            }
        }

        self.max_cut_token = env::i64_clamped("GSV_MAX_CUT_TOKEN", self.max_cut_token, 25, 1024);
        if self.max_cut_token != 25 {
            info!("Max cut token: {}", self.max_cut_token);
        }

        let voice_metrics =
            env::bool01("VOICE_TTS_METRICS", false) || env::bool01("TTS_WORKER_METRICS", false);
        self.log_text_metrics = env::bool01("GSV_TEXT_METRICS", false) || voice_metrics;
        self.jieba_bench = env::bool01("GSV_JIEBA_BENCH", false);
        if self.jieba_bench {
            info!("Jieba bench enabled (extra cut per chunk).");
        }

        self.chunk_policy = GptSovitsChunkPolicy::from_env();
    }
}

impl GptSovitsTts {
    pub fn from_default_dir(audio: Arc<dyn AudioBackend>) -> Result<Self> {
        Self::from_dir(DEFAULT_BASE_DIR, audio)
    }

    pub fn from_dir(base_dir: impl Into<PathBuf>, audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let config = GptSovitsConfig::from_dir(base_dir)?;
        Self::from_config(config, audio)
    }

    pub fn from_env(audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let config = GptSovitsConfig::from_env()?;
        Self::from_config(config, audio)
    }

    pub fn from_config(config: GptSovitsConfig, audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let device = tch::Device::cuda_if_available();
        if !matches!(device, tch::Device::Cuda(_)) {
            return Err(anyhow!(
                "CUDA device is required for `gpt-sovits` backend. Ensure CUDA libtorch is loaded."
            )
            .into());
        }
        let inner = build_inner(&config, device)?;

        Ok(Self {
            inner: Arc::new(StdMutex::new(inner)),
            cancel: CancelToken::new(),
            audio,
            first_call: AtomicBool::new(true),
        })
    }
}

trait ChunkSink {
    fn on_chunk(&mut self, samples: &[f32], cancel: &CancelScope) -> Result<bool>;
}

impl ChunkSink for AudioStreamSegment {
    fn on_chunk(&mut self, samples: &[f32], cancel: &CancelScope) -> Result<bool> {
        Ok(self.push(samples, cancel))
    }
}

struct CollectSink {
    samples: Vec<f32>,
}

impl CollectSink {
    fn new() -> Self {
        Self { samples: Vec::new() }
    }
}

impl ChunkSink for CollectSink {
    fn on_chunk(&mut self, samples: &[f32], _cancel: &CancelScope) -> Result<bool> {
        self.samples.extend_from_slice(samples);
        Ok(true)
    }
}

fn run_stream_infer(
    inner: &Inner,
    text: &str,
    is_first: bool,
    cancel_scope: &CancelScope,
    sink: &mut impl ChunkSink,
) -> Result<()> {
    let text_chars = text.chars().count();
    if let Some(jieba) = inner.jieba.as_ref() {
        let start = Instant::now();
        let _ = jieba.cut(text, true);
        let elapsed = start.elapsed();
        info!("jieba cut time: {:?} | {} chars", elapsed, text_chars);
    }

    let (text_frontend, elapsed) = timing::time_if(inner.log_text_metrics, || {
        gpt_sovits_rs::text::get_phone_and_bert(&inner.g2p, text)
    });
    let (text_seq, text_bert) = text_frontend?;
    if let Some(elapsed) = elapsed {
        info!("text frontend time: {:?} | {} chars", elapsed, text_chars);
    }
    let text_bert = if inner.fp16 { to_half(text_bert) } else { to_float(text_bert) };

    let (prompts, refer, sv_emb) = (
        inner.ref_params.0.shallow_clone(),
        inner.ref_params.1.shallow_clone(),
        inner.ref_params.2.shallow_clone(),
    );

    let ref_seq = inner.ref_seq.shallow_clone();
    let ref_bert = inner.ref_bert.shallow_clone();
    let top_k = if is_first {
        inner.top_k_first
    } else {
        inner.top_k
    };

    let (stream_res, elapsed) = timing::time_if(inner.log_text_metrics, || {
        inner.speaker.stream_infer(
            (prompts, refer, sv_emb),
            ref_seq,
            text_seq,
            ref_bert,
            text_bert,
            top_k,
        )
    });
    let mut stream = stream_res?;
    if let Some(elapsed) = elapsed {
        info!("stream_infer init time: {:?}", elapsed);
    }

    let chunk_token_nums = dynamic_chunk_token_nums(
        text_chars,
        &inner.chunk_token_nums,
        &inner.chunk_policy,
        inner.log_text_metrics,
    );
    let mut first_chunk_gen_logged = false;
    let mut first_chunk_io_logged = false;
    let mut samples_buf: Vec<f32> = Vec::new();
    while !cancel_scope.is_cancelled() {
        let chunk_start =
            (!first_chunk_gen_logged && inner.log_text_metrics).then_some(Instant::now());
        let Some(audio) = stream.next_chunk(inner.max_cut_token, &chunk_token_nums)? else {
            break;
        };
        if let Some(start) = chunk_start {
            let elapsed = start.elapsed();
            info!("next_chunk first return time: {:?}", elapsed);
            first_chunk_gen_logged = true;
        }

        let audio = audio.contiguous();
        let audio_size = audio.numel();
        if audio_size == 0 {
            continue;
        }

        let io_start =
            (!first_chunk_io_logged && inner.log_text_metrics).then_some(Instant::now());
        let audio_cpu = audio.f_to_device(tch::Device::Cpu)?.contiguous();
        samples_buf.resize(audio_size, 0.0);
        audio_cpu.f_copy_data(&mut samples_buf, audio_size)?;

        let accepted = sink.on_chunk(&samples_buf, cancel_scope)?;
        if !accepted {
            break;
        }
        if let Some(start) = io_start {
            let elapsed = start.elapsed();
            info!("first chunk cpu+append time: {:?}", elapsed);
            first_chunk_io_logged = true;
        }
    }

    Ok(())
}

#[cfg(feature = "tts-worker")]
#[derive(Debug, Default, Clone)]
pub(crate) struct Pcm16StreamStats {
    pub first_audio_ts: Option<Instant>,
    pub chunks: u64,
    pub samples: u64,
    pub bytes: u64,
}

#[cfg(feature = "tts-worker")]
pub(crate) struct Pcm16StreamOutcome {
    pub stats: Pcm16StreamStats,
    pub result: Result<()>,
}

#[cfg(feature = "tts-worker")]
struct Pcm16MpscSink {
    tx: mpsc::Sender<Bytes>,
    cancel: CancelToken,
    stats: Pcm16StreamStats,
    log_metrics: bool,
}

#[cfg(feature = "tts-worker")]
impl Pcm16MpscSink {
    fn new(tx: mpsc::Sender<Bytes>, cancel: CancelToken, log_metrics: bool) -> Self {
        Self {
            tx,
            cancel,
            stats: Pcm16StreamStats::default(),
            log_metrics,
        }
    }
}

#[cfg(feature = "tts-worker")]
impl ChunkSink for Pcm16MpscSink {
    fn on_chunk(&mut self, samples: &[f32], cancel: &CancelScope) -> Result<bool> {
        if cancel.is_cancelled() {
            return Ok(false);
        }

        if samples.is_empty() {
            return Ok(true);
        }

        let mut pcm = Vec::<u8>::with_capacity(samples.len().saturating_mul(2));
        for &s in samples {
            let scaled = (s * 32768.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            pcm.extend_from_slice(&scaled.to_le_bytes());
        }

        let bytes_len = pcm.len() as u64;
        let samples_len = samples.len() as u64;

        match self.tx.blocking_send(Bytes::from(pcm)) {
            Ok(_) => {
                if self.log_metrics {
                    self.stats.chunks += 1;
                    self.stats.samples += samples_len;
                    self.stats.bytes += bytes_len;
                    if self.stats.first_audio_ts.is_none() {
                        self.stats.first_audio_ts = Some(Instant::now());
                    }
                }
                Ok(true)
            }
            Err(_) => {
                self.cancel.cancel();
                Ok(false)
            }
        }
    }
}

#[cfg(feature = "tts-worker")]
pub(crate) struct GptSovitsWorkerModel {
    inner: StdMutex<Inner>,
    first_call: AtomicBool,
}

#[cfg(feature = "tts-worker")]
impl GptSovitsWorkerModel {
    pub(crate) fn from_env_cuda_only() -> Result<Self> {
        let config = GptSovitsConfig::from_env()?;
        let device = tch::Device::cuda_if_available();
        if !matches!(device, tch::Device::Cuda(_)) {
            return Err(anyhow!(
                "CUDA device is required for gpt-sovits worker; ensure CUDA libtorch is loaded"
            )
            .into());
        }
        let inner = build_inner(&config, device)?;
        Ok(Self {
            inner: StdMutex::new(inner),
            first_call: AtomicBool::new(true),
        })
    }

    pub(crate) fn stream_pcm16le(
        &self,
        text: &str,
        tx: mpsc::Sender<Bytes>,
        log_metrics: bool,
    ) -> Pcm16StreamOutcome {
        let cancel = CancelToken::new();
        let cancel_scope = cancel.scope();
        let is_first = self.first_call.swap(false, Ordering::AcqRel);

        let mut sink = Pcm16MpscSink::new(tx, cancel, log_metrics);
        let inner_guard = self
            .inner
            .lock()
            .map_err(|_| anyhow!("gpt-sovits worker lock poisoned"));
        let result = match inner_guard {
            Ok(guard) => {
                let inner = &*guard;
                let _g = tch::no_grad_guard();
                run_stream_infer(inner, text, is_first, &cancel_scope, &mut sink)
            }
            Err(err) => Err(err.into()),
        };

        Pcm16StreamOutcome {
            stats: sink.stats,
            result,
        }
    }
}

#[async_trait]
impl TtsEngine for GptSovitsTts {
    async fn speak(&self, text: &str) -> Result<TtsMetrics> {
        let text = text.to_owned();
        let inner = self.inner.clone();
        let cancel = self.cancel.clone();
        let audio = self.audio.clone();
        let is_first = self.first_call.swap(false, Ordering::AcqRel);

        let metrics = tokio::task::spawn_blocking(move || {
            let start_ts = Instant::now();

            let cancel_scope = cancel.scope();

            let inner_guard = inner
                .lock()
                .map_err(|_| anyhow!("gpt-sovits engine lock poisoned"))?;
            let inner = &*inner_guard;

            let _g = tch::no_grad_guard();
            // Phase 2: scope is cloned and bound to the segment writer at creation
            let mut segment = AudioStreamSegment::new(audio.as_ref(), cancel_scope.clone());
            run_stream_infer(inner, &text, is_first, &cancel_scope, &mut segment)?;
            let gen_done_ts = Instant::now();
            let (first_audio_ts, playback) = segment.finish(cancel_scope.is_cancelled());
            let play_done_ts = playback.play_done_ts;
            let play_done_rx = playback.play_done_rx;

            Ok::<TtsMetrics, anyhow::Error>(TtsMetrics {
                start_ts,
                first_audio_ts,
                gen_done_ts,
                play_done_ts,
                play_done_rx,
            })
        })
        .await??;

        Ok(metrics)
    }

    async fn stop(&self) -> Result<()> {
        self.stop_fast();
        Ok(())
    }

    fn stop_fast(&self) {
        // O(1) fast path: increment epoch + clear ring buffer
        self.cancel.cancel();
        self.audio.stop();
        self.first_call.store(true, Ordering::Release);
    }

    fn supports_synthesis_queue(&self) -> bool {
        false
    }

    async fn synthesize(&self, text: &str) -> Result<Option<SynthesizedAudio>> {
        let text = text.to_owned();
        let inner = self.inner.clone();
        let cancel = self.cancel.clone();
        let is_first = self.first_call.swap(false, Ordering::AcqRel);

        let audio = tokio::task::spawn_blocking(move || {
            let start_ts = Instant::now();
            let cancel_scope = cancel.scope();

            let inner_guard = inner
                .lock()
                .map_err(|_| anyhow!("gpt-sovits engine lock poisoned"))?;
            let inner = &*inner_guard;

            let _g = tch::no_grad_guard();

            let mut sink = CollectSink::new();
            run_stream_infer(inner, &text, is_first, &cancel_scope, &mut sink)?;
            let gen_done_ts = Instant::now();
            Ok::<SynthesizedAudio, anyhow::Error>(SynthesizedAudio {
                samples: sink.samples,
                start_ts,
                gen_done_ts,
            })
        })
        .await??;

        Ok(Some(audio))
    }

    async fn play_samples(&self, audio: SynthesizedAudio) -> Result<Option<TtsMetrics>> {
        let cancel = self.cancel.clone();
        let audio_backend = self.audio.clone();

        let metrics = tokio::task::spawn_blocking(move || {
            let cancel_scope = cancel.scope();

            // Phase 2: scope is cloned and bound to the segment writer at creation
            let mut segment = AudioStreamSegment::new(audio_backend.as_ref(), cancel_scope.clone());
            for chunk in audio.samples.chunks(DEFAULT_PLAY_CHUNK_SAMPLES) {
                if !segment.push(chunk, &cancel_scope) {
                    break;
                }
            }

            let (first_audio_ts, playback) = segment.finish(cancel_scope.is_cancelled());

            Ok::<TtsMetrics, anyhow::Error>(TtsMetrics {
                start_ts: audio.start_ts,
                first_audio_ts,
                gen_done_ts: audio.gen_done_ts,
                play_done_ts: playback.play_done_ts,
                play_done_rx: playback.play_done_rx,
            })
        })
        .await??;

        Ok(Some(metrics))
    }

    fn buffered_ms(&self) -> Option<u64> {
        self.audio.buffered_ms()
    }
}

pub fn build(audio: Arc<dyn AudioBackend>) -> Result<Arc<dyn TtsEngine>> {
    if audio.sample_rate() != 32_000 || audio.channels() != 1 {
        return Err(anyhow!(
            "GPT-SoVITS expects 32000Hz mono output; set AUDIO_SAMPLE_RATE=32000 and AUDIO_CHANNELS=1"
        )
        .into());
    }
    info!("Initializing GPT-SoVITS backend...");
    Ok(Arc::new(GptSovitsTts::from_env(audio)?))
}

fn dynamic_chunk_token_nums(
    text_chars: usize,
    base: &[i64],
    policy: &GptSovitsChunkPolicy,
    log: bool,
) -> Vec<i64> {
    if base.is_empty() {
        return Vec::new();
    }
    if !policy.dynamic {
        return base.to_vec();
    }

    let base_first = base[0].clamp(3, 25);
    let adjusted = if text_chars <= policy.short_chars {
        base_first.min(policy.short_tokens)
    } else if text_chars <= policy.mid_chars {
        base_first.min(policy.mid_tokens)
    } else {
        base_first
    };

    if adjusted == base_first {
        return base.to_vec();
    }

    let mut updated = base.to_vec();
    updated[0] = adjusted;
    if log {
        info!(
            "Dynamic first chunk tokens: {} -> {} (chars={})",
            base_first, adjusted, text_chars
        );
    }
    updated
}
