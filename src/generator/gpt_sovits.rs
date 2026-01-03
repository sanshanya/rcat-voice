use crate::audio::{AudioBackend, CancelToken};
use super::{Result, SynthesizedAudio, TtsEngine, TtsMetrics};
use anyhow::anyhow;
use async_trait::async_trait;
use gpt_sovits_rs::gsv;
use gpt_sovits_rs::tch;
use gpt_sovits_rs::text::G2PConfig;
use jieba_rs::Jieba;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::Instant;
use tracing::info;

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
        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_DYNAMIC") {
            cfg.dynamic = value != "0";
        }
        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_SHORT_CHARS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.short_chars = parsed.clamp(1, 200);
            }
        }
        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_MID_CHARS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.mid_chars = parsed.clamp(cfg.short_chars, 400);
            }
        }
        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_SHORT_TOKENS") {
            if let Ok(parsed) = value.parse::<i64>() {
                cfg.short_tokens = parsed.clamp(3, 25);
            }
        }
        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_MID_TOKENS") {
            if let Ok(parsed) = value.parse::<i64>() {
                cfg.mid_tokens = parsed.clamp(3, 25);
            }
        }
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

    let _g = tch::no_grad_guard();
    let (prompts, refer, sv_emb) = speaker.pre_handle_ref(ref_audio_32k)?;

    let (ref_seq, ref_bert) = gpt_sovits_rs::text::get_phone_and_bert(&g2p, &config.ref_text)?;
    let ref_bert = if fp16 { to_half(ref_bert) } else { to_float(ref_bert) };

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

fn first_existing(base_dir: &Path, label: &str, candidates: &[&str]) -> Result<PathBuf> {
    for rel in candidates {
        let p = base_dir.join(rel);
        if p.exists() {
            return Ok(p);
        }
    }
    let tried = candidates
        .iter()
        .map(|c| format!("`{}`", base_dir.join(c).display()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(anyhow!("{label} not found (tried: {tried})").into())
}

impl GptSovitsConfig {
    pub fn from_dir(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir: PathBuf = base_dir.into();

        let g2p_en_path = first_existing(
            &base_dir,
            "g2p_en model (mini-bart-g2p)",
            &["mini-bart-g2p.pt", "resource/mini-bart-g2p.pt"],
        )?;
        let g2pw_path = first_existing(&base_dir, "g2pw model", &["g2pw_model.pt", "g2pw.pt"])?;
        let cn_bert_path =
            first_existing(&base_dir, "cn_bert model", &["bert_model.pt", "bert.pt"])?;
        let ssl_model_path = first_existing(
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
            first_existing(&base_dir, "t2s model", &["t2s.pt", "t2s.cpu.pt"])?
        };

        let vits_model_path = if base_dir.join("vits.pt").exists() {
            base_dir.join("vits.pt")
        } else if combined_path.exists() {
            combined_path
        } else {
            first_existing(&base_dir, "vits model", &["vits.pt", "vits.cpu.pt"])?
        };

        let ref_wav_path = first_existing(
            &base_dir,
            "ref.wav",
            &["ref.wav", "ref_32k.wav", "ref32k.wav"],
        )?;
        let ref_text_path = first_existing(&base_dir, "ref.txt", &["ref.txt"])?;
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
        let base_dir = std::env::var("GSV_MODEL_DIR")
            .unwrap_or_else(|_| DEFAULT_BASE_DIR.to_string());
        let mut cfg = Self::from_dir(base_dir)?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(value) = std::env::var("GSV_TOP_K") {
            if let Ok(parsed) = value.parse::<i64>() {
                self.top_k = parsed.clamp(1, 50);
            }
        }
        if let Ok(value) = std::env::var("GSV_FIRST_TOP_K") {
            if let Ok(parsed) = value.parse::<i64>() {
                self.top_k_first = parsed.clamp(1, self.top_k);
            }
        }
        if self.top_k_first != self.top_k {
            info!("First segment top_k: {} (default={})", self.top_k_first, self.top_k);
        }

        if let Ok(value) = std::env::var("GSV_FIRST_CHUNK_TOKENS") {
            if let Ok(parsed) = value.parse::<i64>() {
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
        }

        if let Ok(value) = std::env::var("GSV_MAX_CUT_TOKEN") {
            if let Ok(parsed) = value.parse::<i64>() {
                self.max_cut_token = parsed.clamp(25, 1024);
            }
        }
        if self.max_cut_token != 25 {
            info!("Max cut token: {}", self.max_cut_token);
        }

        self.log_text_metrics = std::env::var("GSV_TEXT_METRICS")
            .map(|v| v == "1")
            .unwrap_or(false);
        self.jieba_bench = std::env::var("GSV_JIEBA_BENCH")
            .map(|v| v == "1")
            .unwrap_or(false);
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
            let mut first_audio_ts: Option<Instant> = None;

            let cancel_scope = cancel.scope();

            let inner_guard = inner
                .lock()
                .map_err(|_| anyhow!("gpt-sovits engine lock poisoned"))?;
            let inner = &*inner_guard;

            let _g = tch::no_grad_guard();
            let mut segment = audio.begin_segment();

            let text_chars = text.chars().count();
            if let Some(jieba) = inner.jieba.as_ref() {
                let start = Instant::now();
                let _ = jieba.cut(&text, true);
                let elapsed = start.elapsed();
                info!("jieba cut time: {:?} | {} chars", elapsed, text_chars);
            }

            let text_front_start = if inner.log_text_metrics {
                Some(Instant::now())
            } else {
                None
            };
            let (text_seq, text_bert) =
                gpt_sovits_rs::text::get_phone_and_bert(&inner.g2p, &text)?;
            if let Some(start) = text_front_start {
                let elapsed = start.elapsed();
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

            let stream_infer_start = if inner.log_text_metrics {
                Some(Instant::now())
            } else {
                None
            };
            let mut stream = inner.speaker.stream_infer(
                (prompts, refer, sv_emb),
                ref_seq,
                text_seq,
                ref_bert,
                text_bert,
                top_k,
            )?;
            if let Some(start) = stream_infer_start {
                let elapsed = start.elapsed();
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
            while !cancel_scope.is_cancelled() {
                let chunk_start = if !first_chunk_gen_logged && inner.log_text_metrics {
                    Some(Instant::now())
                } else {
                    None
                };
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

                let io_start = if !first_chunk_io_logged && inner.log_text_metrics {
                    Some(Instant::now())
                } else {
                    None
                };
                let audio_cpu = audio.f_to_device(tch::Device::Cpu)?.contiguous();
                let mut samples = vec![0f32; audio_size];
                audio_cpu.f_copy_data(&mut samples, audio_size)?;

                let written = segment.push(&samples, &cancel_scope);
                if written == 0 {
                    continue;
                }
                if first_audio_ts.is_none() {
                    first_audio_ts = segment.first_audio_ts();
                }
                if let Some(start) = io_start {
                    let elapsed = start.elapsed();
                    info!("first chunk cpu+append time: {:?}", elapsed);
                    first_chunk_io_logged = true;
                }
            }

            let gen_done_ts = Instant::now();
            let playback = segment.finish(cancel_scope.is_cancelled());
            if first_audio_ts.is_none() {
                first_audio_ts = playback.first_audio_ts;
            }
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
        self.cancel.cancel();

        self.audio.stop();
        self.first_call.store(true, Ordering::Release);

        Ok(())
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

            let text_chars = text.chars().count();
            if let Some(jieba) = inner.jieba.as_ref() {
                let start = Instant::now();
                let _ = jieba.cut(&text, true);
                let elapsed = start.elapsed();
                info!("jieba cut time: {:?} | {} chars", elapsed, text_chars);
            }

            let text_front_start = if inner.log_text_metrics {
                Some(Instant::now())
            } else {
                None
            };
            let (text_seq, text_bert) =
                gpt_sovits_rs::text::get_phone_and_bert(&inner.g2p, &text)?;
            if let Some(start) = text_front_start {
                let elapsed = start.elapsed();
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

            let stream_infer_start = if inner.log_text_metrics {
                Some(Instant::now())
            } else {
                None
            };
            let mut stream = inner.speaker.stream_infer(
                (prompts, refer, sv_emb),
                ref_seq,
                text_seq,
                ref_bert,
                text_bert,
                top_k,
            )?;
            if let Some(start) = stream_infer_start {
                let elapsed = start.elapsed();
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
            let mut samples_all = Vec::new();
            while !cancel_scope.is_cancelled() {
                let chunk_start = if !first_chunk_gen_logged && inner.log_text_metrics {
                    Some(Instant::now())
                } else {
                    None
                };
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

                let io_start = if !first_chunk_io_logged && inner.log_text_metrics {
                    Some(Instant::now())
                } else {
                    None
                };
                let audio_cpu = audio.f_to_device(tch::Device::Cpu)?.contiguous();
                let mut samples = vec![0f32; audio_size];
                audio_cpu.f_copy_data(&mut samples, audio_size)?;
                samples_all.extend_from_slice(&samples);

                if let Some(start) = io_start {
                    let elapsed = start.elapsed();
                    info!("first chunk cpu+append time: {:?}", elapsed);
                    first_chunk_io_logged = true;
                }
            }

            let gen_done_ts = Instant::now();
            Ok::<SynthesizedAudio, anyhow::Error>(SynthesizedAudio {
                samples: samples_all,
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
            let mut first_audio_ts: Option<Instant> = None;
            let mut segment = audio_backend.begin_segment();

            for chunk in audio.samples.chunks(DEFAULT_PLAY_CHUNK_SAMPLES) {
                if cancel_scope.is_cancelled() {
                    break;
                }
                let written = segment.push(chunk, &cancel_scope);
                if written == 0 {
                    continue;
                }
                if first_audio_ts.is_none() {
                    first_audio_ts = segment.first_audio_ts();
                }
            }

            let playback = segment.finish(cancel_scope.is_cancelled());
            if first_audio_ts.is_none() {
                first_audio_ts = playback.first_audio_ts;
            }

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
