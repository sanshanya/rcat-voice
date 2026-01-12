use crate::audio::{AudioBackend, AudioStreamSegment, CancelScope, CancelToken, SegmentPlayback};
use super::{Result, SynthesizedAudio, TtsEngine, TtsMetrics};
use anyhow::{anyhow, Context};
use async_trait::async_trait;
use gpt_sovits_onnx_rs::{LangId, SamplingParams, SamplingParamsBuilder, TTSModel};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::time::Instant;
use tracing::info;

use crate::internal::{env, model_locator};

const DEFAULT_MODEL_DIR: &str = "onnx";
const DEFAULT_EXPORT_NAME: &str = "custom";
const DEFAULT_CHUNK_SAMPLES: usize = 2048;

/// Sampling parameters for GPT-SoVITS ONNX decoding.
#[derive(Debug, Clone)]
pub struct GptSovitsOnnxSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for GptSovitsOnnxSampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 4,
            top_p: 0.9,
            repetition_penalty: 1.35,
        }
    }
}

impl GptSovitsOnnxSampling {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(parsed) = env::get::<f32>("GSV_ONNX_TEMPERATURE") {
            cfg.temperature = parsed.max(0.0);
        }
        if let Some(parsed) = env::get::<usize>("GSV_ONNX_TOP_K") {
            cfg.top_k = parsed;
        }
        if let Some(parsed) = env::get::<f32>("GSV_ONNX_TOP_P") {
            cfg.top_p = parsed;
        }
        if let Some(parsed) = env::get::<f32>("GSV_ONNX_REP_PENALTY") {
            cfg.repetition_penalty = parsed.max(0.1);
        }
        cfg
    }
}

/// GPT-SoVITS ONNX backend configuration.
#[derive(Debug, Clone)]
pub struct GptSovitsOnnxConfig {
    pub model_dir: PathBuf,
    pub export_name: String,
    pub bert_path: Option<PathBuf>,
    pub g2pw_path: Option<PathBuf>,
    pub g2p_en_path: Option<PathBuf>,
    pub sv_path: Option<PathBuf>,
    pub ref_wav_path: Option<PathBuf>,
    pub ref_text: Option<String>,
    pub lang: LangId,
    pub sampling: GptSovitsOnnxSampling,
    pub chunk_samples: usize,
}

impl GptSovitsOnnxConfig {
    pub fn from_dir(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: base_dir.into(),
            export_name: DEFAULT_EXPORT_NAME.to_string(),
            bert_path: None,
            g2pw_path: None,
            g2p_en_path: None,
            sv_path: None,
            ref_wav_path: None,
            ref_text: None,
            lang: LangId::Auto,
            sampling: GptSovitsOnnxSampling::default(),
            chunk_samples: DEFAULT_CHUNK_SAMPLES,
        }
    }

    pub fn from_env() -> Result<Self> {
        let model_dir =
            env::string("GSV_ONNX_MODEL_DIR").unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string());
        let export_name =
            env::string("GSV_ONNX_EXPORT_NAME").unwrap_or_else(|| DEFAULT_EXPORT_NAME.to_string());

        let bert_path = env::path_opt("GSV_ONNX_BERT_PATH")?;
        let g2pw_path = env::path_opt("GSV_ONNX_G2PW_PATH")?;
        let g2p_en_path = match env::path_opt("GSV_ONNX_G2P_EN_PATH")? {
            Some(path) => {
                validate_g2p_en_dir(&path)?;
                Some(path)
            }
            None => None,
        };
        let sv_path = env::path_opt("GSV_ONNX_SV_PATH")?;
        let ref_wav_path = env::path_opt("GSV_ONNX_REF_WAV")?;
        let ref_text = env::string("GSV_ONNX_REF_TEXT");
        let lang = env::string("GSV_ONNX_LANG").unwrap_or_else(|| "auto".to_string());

        let chunk_samples =
            env::usize_clamped("GSV_ONNX_CHUNK_SAMPLES", DEFAULT_CHUNK_SAMPLES, 256, 16384);

        Ok(Self {
            model_dir: PathBuf::from(model_dir),
            export_name,
            bert_path,
            g2pw_path,
            g2p_en_path,
            sv_path,
            ref_wav_path,
            ref_text,
            lang: parse_lang_id(&lang),
            sampling: GptSovitsOnnxSampling::from_env(),
            chunk_samples,
        })
    }
}

pub struct GptSovitsOnnxTts {
    model: Arc<StdMutex<TTSModel>>,
    sampling: SamplingParams,
    lang_id: LangId,
    chunk_samples: usize,
    cancel: CancelToken,
    audio: Arc<dyn AudioBackend>,
}

impl GptSovitsOnnxTts {
    pub fn from_default_dir(audio: Arc<dyn AudioBackend>) -> Result<Self> {
        Self::from_dir(DEFAULT_MODEL_DIR, audio)
    }

    pub fn from_dir(base_dir: impl Into<PathBuf>, audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let config = GptSovitsOnnxConfig::from_dir(base_dir);
        Self::from_config(config, audio)
    }

    pub fn from_env(audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let config = GptSovitsOnnxConfig::from_env()?;
        Self::from_config(config, audio)
    }

    pub fn from_config(config: GptSovitsOnnxConfig, audio: Arc<dyn AudioBackend>) -> Result<Self> {
        let base_dir = config.model_dir;
        if !base_dir.is_dir() {
            return Err(anyhow!(
                "GPT-SoVITS ONNX model dir not found: {}",
                base_dir.display()
            )
            .into());
        }

        let export_name = config.export_name;

        let sovits_candidates = vec![
            format!("{export_name}_vits.onnx"),
            "custom_vits.onnx".to_string(),
            "vits.onnx".to_string(),
        ];
        let ssl_candidates = vec![
            "ssl.onnx".to_string(),
            format!("{export_name}_ssl.onnx"),
        ];
        let t2s_encoder_candidates = vec![
            format!("{export_name}_t2s_encoder.onnx"),
            "custom_t2s_encoder.onnx".to_string(),
            "t2s_encoder.onnx".to_string(),
        ];
        let t2s_fs_decoder_candidates = vec![
            format!("{export_name}_t2s_fs_decoder.onnx"),
            "custom_t2s_fs_decoder.onnx".to_string(),
            "t2s_fs_decoder.onnx".to_string(),
        ];
        let t2s_s_decoder_candidates = vec![
            format!("{export_name}_t2s_s_decoder.onnx"),
            "custom_t2s_s_decoder.onnx".to_string(),
            "t2s_s_decoder.onnx".to_string(),
        ];

        let sovits_path =
            model_locator::first_existing_file_rel(&base_dir, "SoVITS model", &sovits_candidates)?;
        let ssl_path =
            model_locator::first_existing_file_rel(&base_dir, "SSL model", &ssl_candidates)?;
        let t2s_encoder_path = model_locator::first_existing_file_rel(
            &base_dir,
            "T2S encoder model",
            &t2s_encoder_candidates,
        )?;
        let t2s_fs_decoder_path = model_locator::first_existing_file_rel(
            &base_dir,
            "T2S FS decoder model",
            &t2s_fs_decoder_candidates,
        )?;
        let t2s_s_decoder_path = model_locator::first_existing_file_rel(
            &base_dir,
            "T2S S decoder model",
            &t2s_s_decoder_candidates,
        )?;

        let bert_candidates = ["bert.onnx"];
        let g2pw_candidates = ["g2pW.onnx", "g2pw.onnx"];
        let sv_candidates = ["sv.onnx"];

        let bert_path = config
            .bert_path
            .or_else(|| model_locator::optional_existing_file_rel(&base_dir, &bert_candidates));
        if bert_path.is_none() {
            info!("BERT model not found, using zero embeddings.");
        }

        let g2pw_path = config
            .g2pw_path
            .or_else(|| model_locator::optional_existing_file_rel(&base_dir, &g2pw_candidates));
        if g2pw_path.is_none() {
            info!("G2PW model not found, using simple pinyin fallback.");
        }

        let g2p_en_path = match config.g2p_en_path {
            Some(path) => {
                validate_g2p_en_dir(&path)?;
                Some(path)
            }
            None => detect_g2p_en_dir(&base_dir),
        };
        if g2p_en_path.is_none() {
            info!("G2P EN model not found, using ARPAbet fallback.");
        }

        let sv_path = config
            .sv_path
            .or_else(|| model_locator::optional_existing_file_rel(&base_dir, &sv_candidates));
        if sv_path.is_none() {
            info!("SV model not found, speaker embedding disabled.");
        }

        let ref_wav_path = match config.ref_wav_path {
            Some(path) => path,
            None => {
                let ref_candidates = vec![
                    "ref.wav".to_string(),
                    "ref_16k.wav".to_string(),
                    "ref16k.wav".to_string(),
                ];
                model_locator::first_existing_file_rel(&base_dir, "Reference wav", &ref_candidates)?
            }
        };

        let ref_text = match config.ref_text {
            Some(text) => text,
            None => {
                let ref_text_path = base_dir.join("ref.txt");
                std::fs::read_to_string(&ref_text_path).with_context(|| {
                    format!("Failed to read reference text: {}", ref_text_path.display())
                })?
            }
        };
        let ref_text = ref_text
            .trim_start_matches('\u{feff}')
            .trim()
            .to_string();
        if ref_text.is_empty() {
            return Err(anyhow!("Reference text is empty").into());
        }

        let lang_id = config.lang;
        let sampling = build_sampling_params(&config.sampling);
        let chunk_samples = config.chunk_samples;

        info!(
            "Loading GPT-SoVITS ONNX models from {} (export={})",
            base_dir.display(),
            export_name
        );
        let mut model = TTSModel::new(
            sovits_path,
            ssl_path,
            t2s_encoder_path,
            t2s_fs_decoder_path,
            t2s_s_decoder_path,
            bert_path,
            g2pw_path,
            g2p_en_path,
            sv_path,
        )
        .map_err(|e| anyhow!("gpt-sovits-onnx init failed: {e}"))?;

        model
            .process_reference_sync(&ref_wav_path, &ref_text, lang_id)
            .map_err(|e| anyhow!("gpt-sovits-onnx reference failed: {e}"))?;

        info!(
            "GPT-SoVITS ONNX ready: lang={:?}, top_k={:?}, top_p={:?}, temperature={}, repetition_penalty={}, chunk_samples={}",
            lang_id,
            sampling.top_k,
            sampling.top_p,
            sampling.temperature,
            sampling.repetition_penalty,
            chunk_samples
        );

        Ok(Self {
            model: Arc::new(StdMutex::new(model)),
            sampling,
            lang_id,
            chunk_samples,
            cancel: CancelToken::new(),
            audio,
        })
    }
}

fn play_chunks(
    samples: &[f32],
    chunk_samples: usize,
    audio: &dyn AudioBackend,
    cancel: &CancelScope,
) -> (Option<Instant>, Instant, SegmentPlayback) {
    // Phase 2: scope is cloned and bound to the segment writer at creation
    let mut segment = AudioStreamSegment::new(audio, cancel.clone());
    for chunk in samples.chunks(chunk_samples) {
        if !segment.push(chunk, cancel) {
            break;
        }
    }

    let gen_done_ts = Instant::now();
    let (first_audio_ts, playback) = segment.finish(cancel.is_cancelled());
    (first_audio_ts, gen_done_ts, playback)
}

#[async_trait]
impl TtsEngine for GptSovitsOnnxTts {
    async fn speak(&self, text: &str) -> Result<TtsMetrics> {
        let text = text.to_owned();
        let model = self.model.clone();
        let sampling = self.sampling;
        let lang_id = self.lang_id;
        let chunk_samples = self.chunk_samples;
        let cancel = self.cancel.clone();
        let audio = self.audio.clone();

        let metrics = tokio::task::spawn_blocking(move || {
            let start_ts = Instant::now();
            let cancel_scope = cancel.scope();

            let (spec, samples) = {
                let mut model_guard = model
                    .lock()
                    .map_err(|_| anyhow!("gpt-sovits-onnx engine lock poisoned"))?;
                model_guard
                    .synthesize_sync(&text, sampling, lang_id)
                    .map_err(|e| anyhow!("gpt-sovits-onnx synthesize failed: {e}"))?
            };
            if spec.sample_rate != audio.sample_rate() || spec.channels != audio.channels() {
                return Err(anyhow!(
                    "gpt-sovits-onnx output {}Hz/{}ch does not match audio backend {}Hz/{}ch",
                    spec.sample_rate,
                    spec.channels,
                    audio.sample_rate(),
                    audio.channels()
                )
                .into());
            }

            let (first_audio_ts, gen_done_ts, playback) =
                play_chunks(&samples, chunk_samples, audio.as_ref(), &cancel_scope);

            Ok::<TtsMetrics, anyhow::Error>(TtsMetrics {
                start_ts,
                first_audio_ts,
                gen_done_ts,
                play_done_ts: playback.play_done_ts,
                play_done_rx: playback.play_done_rx,
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
    }

    fn supports_synthesis_queue(&self) -> bool {
        false
    }

    async fn synthesize(&self, text: &str) -> Result<Option<SynthesizedAudio>> {
        let text = text.to_owned();
        let model = self.model.clone();
        let sampling = self.sampling;
        let lang_id = self.lang_id;
        let cancel = self.cancel.clone();
        let audio_backend = self.audio.clone();

        let audio = tokio::task::spawn_blocking(move || {
            let start_ts = Instant::now();
            let cancel_scope = cancel.scope();

            let (spec, samples) = {
                let mut model_guard = model
                    .lock()
                    .map_err(|_| anyhow!("gpt-sovits-onnx engine lock poisoned"))?;
                model_guard
                    .synthesize_sync(&text, sampling, lang_id)
                    .map_err(|e| anyhow!("gpt-sovits-onnx synthesize failed: {e}"))?
            };
            if spec.sample_rate != audio_backend.sample_rate()
                || spec.channels != audio_backend.channels()
            {
                return Err(anyhow!(
                    "gpt-sovits-onnx output {}Hz/{}ch does not match audio backend {}Hz/{}ch",
                    spec.sample_rate,
                    spec.channels,
                    audio_backend.sample_rate(),
                    audio_backend.channels()
                )
                .into());
            }

            if cancel_scope.is_cancelled() {
                return Ok::<Option<SynthesizedAudio>, anyhow::Error>(None);
            }

            let gen_done_ts = Instant::now();
            Ok::<Option<SynthesizedAudio>, anyhow::Error>(Some(SynthesizedAudio {
                samples,
                start_ts,
                gen_done_ts,
            }))
        })
        .await??;

        Ok(audio)
    }

    async fn play_samples(&self, audio: SynthesizedAudio) -> Result<Option<TtsMetrics>> {
        let cancel = self.cancel.clone();
        let audio_backend = self.audio.clone();
        let chunk_samples = self.chunk_samples;

        let metrics = tokio::task::spawn_blocking(move || {
            let cancel_scope = cancel.scope();
            let (first_audio_ts, _enqueue_done_ts, playback) = play_chunks(
                &audio.samples,
                chunk_samples,
                audio_backend.as_ref(),
                &cancel_scope,
            );

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

    fn cancel_token(&self) -> Option<CancelToken> {
        Some(self.cancel.clone())
    }
}

pub fn build(audio: Arc<dyn AudioBackend>) -> Result<Arc<dyn TtsEngine>> {
    info!("Initializing GPT-SoVITS ONNX backend...");
    Ok(Arc::new(GptSovitsOnnxTts::from_env(audio)?))
}

fn detect_g2p_en_dir(base_dir: &Path) -> Option<PathBuf> {
    let candidate = base_dir.join("g2p_en");
    let encoder = candidate.join("encoder_model.onnx");
    let decoder = candidate.join("decoder_model.onnx");
    if encoder.exists() && decoder.exists() {
        Some(candidate)
    } else {
        None
    }
}

fn validate_g2p_en_dir(path: &Path) -> Result<()> {
    let encoder = path.join("encoder_model.onnx");
    let decoder = path.join("decoder_model.onnx");
    if encoder.exists() && decoder.exists() {
        Ok(())
    } else {
        Err(anyhow!(
            "g2p_en dir missing encoder_model.onnx or decoder_model.onnx: {}",
            path.display()
        )
        .into())
    }
}

fn parse_lang_id(raw: &str) -> LangId {
    match raw.trim().to_lowercase().as_str() {
        "yue" | "cantonese" | "auto-yue" | "autoyue" => LangId::AutoYue,
        _ => LangId::Auto,
    }
}

fn build_sampling_params(config: &GptSovitsOnnxSampling) -> SamplingParams {
    let mut builder = SamplingParamsBuilder::new()
        .temperature(config.temperature)
        .repetition_penalty(config.repetition_penalty);
    if config.top_k > 0 {
        builder = builder.top_k(config.top_k);
    }
    if config.top_p > 0.0 && config.top_p < 1.0 {
        builder = builder.top_p(config.top_p);
    }
    builder.build()
}
