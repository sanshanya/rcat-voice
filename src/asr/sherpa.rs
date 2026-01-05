use super::AsrSegment;
use super::utils::{LinearResampler, pcm_i16_le_bytes_to_vec, pcm_i16_to_mono_f32};
use anyhow::{Result, anyhow, bail};
use sherpa_rs::paraformer::{ParaformerConfig, ParaformerRecognizer};
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::internal::{env, model_locator};

const DEFAULT_MODELS_ROOT: &str = "models";
const FALLBACK_MODELS_ROOT: &str = "asrmodel";
const DEFAULT_MODEL: &str = "paraformer-zh-small-2024-03-09";
const DEFAULT_LANG: &str = "zh";
const DEFAULT_PROVIDER: &str = "cpu";
const DEFAULT_THREADS: i32 = 2;
const TARGET_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_SEGMENT_QUEUE: usize = 8;

const SILERO_VAD_DIR: &str = "silero_vad";
const SILERO_VAD_FILE: &str = "silero_vad.onnx";
const SENSEVOICE_DIR: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
const SENSEVOICE_FUNASR_NANO_DIR: &str = "sherpa-onnx-sense-voice-funasr-nano-2025-12-17";
const SENSEVOICE_FUNASR_NANO_INT8_DIR: &str =
    "sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17";
const SENSEVOICE_MODEL: &str = "model.onnx";
const SENSEVOICE_MODEL_INT8: &str = "model.int8.onnx";
const SENSEVOICE_TOKENS: &str = "tokens.txt";

const PARAFORMER_TRILINGUAL_DIR: &str = "sherpa-onnx-paraformer-trilingual-zh-cantonese-en";
const PARAFORMER_ZH_SMALL_2024_03_09_DIR: &str = "sherpa-onnx-paraformer-zh-small-2024-03-09";
const PARAFORMER_ZH_2024_03_09_DIR: &str = "sherpa-onnx-paraformer-zh-2024-03-09";
const PARAFORMER_ZH_INT8_2025_10_07_DIR: &str = "sherpa-onnx-paraformer-zh-int8-2025-10-07";
const PARAFORMER_EN_DIR: &str = "sherpa-onnx-paraformer-en";
const PARAFORMER_EN_DIR_2024_03_09: &str = "sherpa-onnx-paraformer-en-2024-03-09";
const PARAFORMER_MODEL: &str = "model.onnx";
const PARAFORMER_MODEL_INT8: &str = "model.int8.onnx";
const PARAFORMER_TOKENS: &str = "tokens.txt";

#[derive(Debug, Clone, Copy)]
pub enum SherpaAsrModel {
    ParaformerZhSmall,
    ParaformerZh,
    ParaformerZhInt8,
    ParaformerTrilingual,
    ParaformerEn,
    SenseVoice,
    SenseVoiceInt8,
    SenseVoiceFunAsrNano,
    SenseVoiceFunAsrNanoInt8,
}

impl SherpaAsrModel {
    fn parse(raw: &str) -> Result<Self> {
        let mut key = raw.trim().to_lowercase();
        if let Some((_, tail)) = key.rsplit_once('/') {
            key = tail.to_string();
        }
        if let Some((_, tail)) = key.rsplit_once('\\') {
            key = tail.to_string();
        }
        if let Some(tail) = key.strip_prefix("sherpa-onnx-") {
            key = tail.to_string();
        }
        key = key.replace('_', "-");

        if key.starts_with("sense-voice-funasr-nano-int8")
            || key.starts_with("sensevoice-funasr-nano-int8")
            || key.starts_with("funasr-nano-int8")
        {
            return Ok(Self::SenseVoiceFunAsrNanoInt8);
        }
        if key.starts_with("sense-voice-funasr-nano")
            || key.starts_with("sensevoice-funasr-nano")
            || key.starts_with("funasr-nano")
        {
            return Ok(Self::SenseVoiceFunAsrNano);
        }

        match key.as_str() {
            "paraformer-zh-small"
            | "paraformer-zh-small-2024-03-09" => Ok(Self::ParaformerZhSmall),
            "paraformer-zh"
            | "paraformer-zh-2024-03-09" => Ok(Self::ParaformerZh),
            "paraformer-zh-int8"
            | "paraformer-zh-int8-2025-10-07" => Ok(Self::ParaformerZhInt8),
            "paraformer-trilingual"
            | "paraformer-trilingual-zh-cantonese-en" => Ok(Self::ParaformerTrilingual),
            "paraformer-en"
            | "paraformer-en-2024-03-09" => Ok(Self::ParaformerEn),
            "sensevoice" | "sense-voice" => Ok(Self::SenseVoice),
            "sensevoice-int8" | "sense-voice-int8" => Ok(Self::SenseVoiceInt8),
            s if s.starts_with("sense-voice-") || s.starts_with("sensevoice-") => Ok(Self::SenseVoice),
            other => bail!("Unknown ASR_MODEL: {other}"),
        }
    }
}

/// Silero VAD parameters (used for offline ASR streaming).
#[derive(Debug, Clone)]
pub struct SherpaVadConfig {
    pub model: PathBuf,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub max_speech_duration: f32,
    pub threshold: f32,
    pub window_size: i32,
    pub buffer_size_in_seconds: f32,
}

impl SherpaVadConfig {
    fn from_env(models_root: &Path) -> Result<Self> {
        let model = match env::string("ASR_VAD_PATH") {
            Some(value) => PathBuf::from(value),
            None => {
                let in_dir = models_root.join(SILERO_VAD_DIR).join(SILERO_VAD_FILE);
                if in_dir.exists() {
                    in_dir
                } else {
                    models_root.join(SILERO_VAD_FILE)
                }
            }
        };
        if !model.exists() {
            bail!(
                "VAD model not found: {} (expected {} under ASR_MODELS_ROOT or set ASR_VAD_PATH)",
                model.display(),
                SILERO_VAD_FILE
            );
        }

        let min_silence_duration = env::get::<f32>("ASR_VAD_MIN_SILENCE")
            .unwrap_or(0.25)
            .max(0.05);
        let min_speech_duration = env::get::<f32>("ASR_VAD_MIN_SPEECH").unwrap_or(0.1).max(0.0);
        let max_speech_duration = env::get::<f32>("ASR_VAD_MAX_SPEECH")
            .unwrap_or(30.0)
            .max(0.5);
        let threshold = env::f32_clamped("ASR_VAD_THRESHOLD", 0.5, 0.0, 1.0);
        let window_size = env::i32_clamped("ASR_VAD_WINDOW", 512, 128, 8192);
        let buffer_size_in_seconds =
            env::f32_clamped("ASR_VAD_BUFFER_SECONDS", 100.0, 1.0, 600.0);

        Ok(Self {
            model,
            min_silence_duration,
            min_speech_duration,
            max_speech_duration,
            threshold,
            window_size,
            buffer_size_in_seconds,
        })
    }
}

/// sherpa-rs ASR backend config (matches voiceapi's defaults as much as possible).
#[derive(Debug, Clone)]
pub struct SherpaAsrConfig {
    pub models_root: PathBuf,
    pub model: SherpaAsrModel,
    pub model_dtype: AsrModelDtype,
    pub lang: String,
    pub provider: String,
    pub threads: i32,
    pub infer_log: bool,
    pub segment_queue: usize,
    pub vad_chunk_ms: u64,
    pub vad: SherpaVadConfig,
}

impl SherpaAsrConfig {
    pub fn from_env() -> Result<Self> {
        let models_root = env::string("ASR_MODELS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let fallback = PathBuf::from(FALLBACK_MODELS_ROOT);
                if fallback.exists() {
                    fallback
                } else {
                    PathBuf::from(DEFAULT_MODELS_ROOT)
                }
            });

        let model = env::string("ASR_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let model = SherpaAsrModel::parse(&model)?;
        let model_dtype = AsrModelDtype::from_env()?;

        let lang = env::string("ASR_LANG").unwrap_or_else(|| DEFAULT_LANG.to_string());
        let provider = env::string("ASR_PROVIDER").unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
        let threads = env::i32_clamped("ASR_THREADS", DEFAULT_THREADS, 1, 32);
        let infer_log = env::bool01("ASR_INFER_LOG", false);
        let segment_queue =
            env::usize_clamped("ASR_SEGMENT_QUEUE", DEFAULT_SEGMENT_QUEUE, 1, 128);
        let vad_chunk_ms = env::u64_clamped("ASR_VAD_CHUNK_MS", 20, 5, 2000);

        let vad = SherpaVadConfig::from_env(&models_root)?;

        Ok(Self {
            models_root,
            model,
            model_dtype,
            lang,
            provider,
            threads,
            infer_log,
            segment_queue,
            vad_chunk_ms,
            vad,
        })
    }

    fn resolve_model_paths(&self) -> Result<(PathBuf, PathBuf)> {
        match self.model {
            SherpaAsrModel::SenseVoice => resolve_sense_voice(
                &self.models_root,
                &[SENSEVOICE_DIR],
                false,
            ),
            SherpaAsrModel::SenseVoiceInt8 => resolve_sense_voice(
                &self.models_root,
                &[SENSEVOICE_DIR],
                true,
            ),
            SherpaAsrModel::SenseVoiceFunAsrNano => resolve_sense_voice(
                &self.models_root,
                &[SENSEVOICE_FUNASR_NANO_DIR, SENSEVOICE_FUNASR_NANO_INT8_DIR],
                false,
            ),
            SherpaAsrModel::SenseVoiceFunAsrNanoInt8 => resolve_sense_voice(
                &self.models_root,
                &[SENSEVOICE_FUNASR_NANO_INT8_DIR, SENSEVOICE_FUNASR_NANO_DIR],
                true,
            ),
            SherpaAsrModel::ParaformerZhSmall => resolve_paraformer_dir(
                &self.models_root.join(PARAFORMER_ZH_SMALL_2024_03_09_DIR),
                self.model_dtype,
            ),
            SherpaAsrModel::ParaformerZh => {
                resolve_paraformer_dir(
                    &self.models_root.join(PARAFORMER_ZH_2024_03_09_DIR),
                    self.model_dtype,
                )
            }
            SherpaAsrModel::ParaformerZhInt8 => resolve_paraformer_dir(
                &self.models_root.join(PARAFORMER_ZH_INT8_2025_10_07_DIR),
                self.model_dtype,
            ),
            SherpaAsrModel::ParaformerTrilingual => {
                resolve_paraformer_dir(&self.models_root.join(PARAFORMER_TRILINGUAL_DIR), self.model_dtype)
            }
            SherpaAsrModel::ParaformerEn => {
                // voiceapi uses `sherpa-onnx-paraformer-en`, but the official archive uses
                // `sherpa-onnx-paraformer-en-2024-03-09`. Try both for convenience.
                let base = model_locator::first_existing_dir(&self.models_root, "ASR model dir", &[
                    PARAFORMER_EN_DIR,
                    PARAFORMER_EN_DIR_2024_03_09,
                ])?;
                resolve_paraformer_dir(&base, self.model_dtype)
            }
        }
    }
}

fn resolve_sense_voice(
    models_root: &Path,
    dir_candidates: &[&str],
    prefer_int8: bool,
) -> Result<(PathBuf, PathBuf)> {
    let base = model_locator::first_existing_dir(models_root, "ASR model dir", dir_candidates)?;
    let tokens = model_locator::first_existing_file(&base, "ASR tokens", &[SENSEVOICE_TOKENS])?;

    let model_candidates: &[&str] = if prefer_int8 {
        &[SENSEVOICE_MODEL_INT8, SENSEVOICE_MODEL]
    } else {
        &[SENSEVOICE_MODEL, SENSEVOICE_MODEL_INT8]
    };
    let model = model_locator::first_existing_file(&base, "ASR model file", model_candidates)?;
    Ok((model, tokens))
}

fn resolve_paraformer_dir(base: &Path, dtype: AsrModelDtype) -> Result<(PathBuf, PathBuf)> {
    if !base.exists() {
        bail!("ASR model dir not found: {}", base.display());
    }

    let tokens = base.join(PARAFORMER_TOKENS);
    if !tokens.exists() {
        bail!("ASR tokens not found: {}", tokens.display());
    }

    let int8 = base.join(PARAFORMER_MODEL_INT8);
    let fp32 = base.join(PARAFORMER_MODEL);
    let model = match dtype {
        AsrModelDtype::Auto => {
            if int8.exists() {
                int8
            } else {
                fp32
            }
        }
        AsrModelDtype::Int8 => {
            if int8.exists() {
                int8
            } else {
                bail!(
                    "ASR model file not found: {} (set ASR_MODEL_DTYPE=auto or download {})",
                    int8.display(),
                    PARAFORMER_MODEL_INT8
                );
            }
        }
        AsrModelDtype::Fp32 => {
            if fp32.exists() {
                fp32
            } else {
                bail!(
                    "ASR model file not found: {} (set ASR_MODEL_DTYPE=auto or download {})",
                    fp32.display(),
                    PARAFORMER_MODEL
                );
            }
        }
    };
    if !model.exists() {
        bail!(
            "ASR model file not found: {} (expected {} or {})",
            model.display(),
            PARAFORMER_MODEL_INT8,
            PARAFORMER_MODEL
        );
    }

    Ok((model, tokens))
}

#[derive(Debug, Clone, Copy)]
pub enum AsrModelDtype {
    Auto,
    Int8,
    Fp32,
}

impl Default for AsrModelDtype {
    fn default() -> Self {
        Self::Auto
    }
}

impl AsrModelDtype {
    fn parse(value: &str) -> Result<Self> {
        let value = value.trim().to_lowercase();
        match value.as_str() {
            "auto" => Ok(Self::Auto),
            "int8" | "i8" => Ok(Self::Int8),
            "fp32" | "f32" | "float" => Ok(Self::Fp32),
            other => bail!("Unknown ASR_MODEL_DTYPE: {other} (expected auto|int8|fp32)"),
        }
    }

    fn from_env() -> Result<Self> {
        let Some(value) = env::string("ASR_MODEL_DTYPE") else {
            return Ok(Self::Auto);
        };
        Self::parse(&value)
    }
}

enum InputMsg {
    Samples(Vec<f32>),
    End,
}

#[derive(Debug)]
struct InputFormat {
    sample_rate: u32,
    channels: u16,
}

enum SherpaRecognizer {
    SenseVoice(SenseVoiceRecognizer),
    Paraformer(ParaformerRecognizer),
}

impl SherpaRecognizer {
    fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> String {
        match self {
            Self::SenseVoice(r) => r.transcribe(sample_rate, samples).text,
            Self::Paraformer(r) => r.transcribe(sample_rate, samples).text,
        }
    }
}

fn is_sense_voice_model(model: SherpaAsrModel) -> bool {
    matches!(
        model,
        SherpaAsrModel::SenseVoice
            | SherpaAsrModel::SenseVoiceInt8
            | SherpaAsrModel::SenseVoiceFunAsrNano
            | SherpaAsrModel::SenseVoiceFunAsrNanoInt8
    )
}

fn init_recognizer(config: &SherpaAsrConfig, model_path: &Path, tokens_path: &Path) -> Result<SherpaRecognizer> {
    if is_sense_voice_model(config.model) {
        init_sense_voice(config, model_path, tokens_path)
    } else {
        init_paraformer(config, model_path, tokens_path)
    }
}

fn init_sense_voice(config: &SherpaAsrConfig, model_path: &Path, tokens_path: &Path) -> Result<SherpaRecognizer> {
    let recognizer = SenseVoiceRecognizer::new(SenseVoiceConfig {
        model: model_path.to_string_lossy().to_string(),
        tokens: tokens_path.to_string_lossy().to_string(),
        language: config.lang.clone(),
        use_itn: true,
        provider: Some(config.provider.clone()),
        num_threads: Some(config.threads),
        debug: false,
    })
    .map_err(|e| anyhow!("failed to init SenseVoiceRecognizer: {e}"))?;
    Ok(SherpaRecognizer::SenseVoice(recognizer))
}

fn init_paraformer(config: &SherpaAsrConfig, model_path: &Path, tokens_path: &Path) -> Result<SherpaRecognizer> {
    let recognizer = ParaformerRecognizer::new(ParaformerConfig {
        model: model_path.to_string_lossy().to_string(),
        tokens: tokens_path.to_string_lossy().to_string(),
        provider: Some(config.provider.clone()),
        num_threads: Some(config.threads),
        debug: false,
    })
    .map_err(|e| anyhow!("failed to init ParaformerRecognizer: {e}"))?;
    Ok(SherpaRecognizer::Paraformer(recognizer))
}

/// Streaming offline ASR with VAD segmenting (Paraformer / SenseVoice + Silero VAD).
///
/// - `write_*` accepts PCM audio and pushes to an internal queue.
/// - `read()` returns recognized segments as they become available.
pub struct SherpaAsrStream {
    tx: mpsc::Sender<InputMsg>,
    rx: mpsc::Receiver<AsrSegment>,
    input_format: Option<InputFormat>,
    resampler: Option<LinearResampler>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl SherpaAsrStream {
    pub fn new(config: SherpaAsrConfig) -> Result<Self> {
        let (model_path, tokens_path) = config.resolve_model_paths()?;

        let recognizer = init_recognizer(&config, &model_path, &tokens_path)?;

        let vad_cfg = SileroVadConfig {
            model: config.vad.model.to_string_lossy().to_string(),
            min_silence_duration: config.vad.min_silence_duration,
            min_speech_duration: config.vad.min_speech_duration,
            max_speech_duration: config.vad.max_speech_duration,
            threshold: config.vad.threshold,
            sample_rate: TARGET_SAMPLE_RATE,
            window_size: config.vad.window_size,
            provider: Some(config.provider.clone()),
            num_threads: Some(config.threads),
            debug: false,
        };
        let mut vad = SileroVad::new(vad_cfg, config.vad.buffer_size_in_seconds)
            .map_err(|e| anyhow!("failed to init SileroVad: {e}"))?;

        let (tx, mut in_rx) = mpsc::channel::<InputMsg>(64);
        let (out_tx, rx) = mpsc::channel::<AsrSegment>(64);

        let log_infer = config.infer_log;
        let segment_queue = config.segment_queue;
        let vad_chunk_ms = config.vad_chunk_ms;
        let vad_chunk_samples = ((TARGET_SAMPLE_RATE as u64 * vad_chunk_ms) / 1000).max(1) as usize;

        let join = tokio::spawn(async move {

            let (segment_tx, segment_rx) = mpsc::channel::<VadSegment>(segment_queue);
            let infer_handle = tokio::task::spawn_blocking(move || {
                run_inference_loop(recognizer, segment_rx, out_tx, log_infer);
            });

            let mut dropped_segments: u64 = 0;
            let mut last_drop_log = Instant::now();
            let mut stop = false;
            while let Some(msg) = in_rx.recv().await {
                match msg {
                    InputMsg::Samples(samples) => {
                        if samples.is_empty() {
                            continue;
                        }
                        for chunk in samples.chunks(vad_chunk_samples) {
                            vad.accept_waveform(chunk);
                            if enqueue_vad_segments(
                                &mut vad,
                                &segment_tx,
                                &mut dropped_segments,
                                &mut last_drop_log,
                            ) {
                                stop = true;
                                break;
                            }
                        }
                    }
                    InputMsg::End => {
                        vad.flush();
                        let _ = enqueue_vad_segments(
                            &mut vad,
                            &segment_tx,
                            &mut dropped_segments,
                            &mut last_drop_log,
                        );
                        break;
                    }
                }
                if stop {
                    break;
                }
            }

            drop(segment_tx);
            let _ = infer_handle.await;
        });

        info!("asr: sherpa stream started (target_sr={TARGET_SAMPLE_RATE})");
        Ok(Self {
            tx,
            rx,
            input_format: None,
            resampler: None,
            join: Some(join),
        })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(SherpaAsrConfig::from_env()?)
    }

    pub async fn write_pcm_i16_le_bytes(
        &mut self,
        pcm: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> Result<()> {
        let pcm = pcm_i16_le_bytes_to_vec(pcm)?;
        self.write_pcm_i16(&pcm, sample_rate, channels).await
    }

    pub async fn write_pcm_i16(
        &mut self,
        pcm: &[i16],
        sample_rate: u32,
        channels: u16,
    ) -> Result<()> {
        if sample_rate == 0 {
            bail!("sample_rate must be > 0");
        }
        let input_format = self.input_format.get_or_insert_with(|| InputFormat {
            sample_rate,
            channels,
        });
        if input_format.sample_rate != sample_rate || input_format.channels != channels {
            bail!(
                "input audio format changed: {}Hz/{}ch -> {}Hz/{}ch",
                input_format.sample_rate,
                input_format.channels,
                sample_rate,
                channels
            );
        }

        let mono = pcm_i16_to_mono_f32(pcm, channels)?;
        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            mono
        } else {
            if self.resampler.is_none() {
                self.resampler = Some(LinearResampler::new(sample_rate, TARGET_SAMPLE_RATE)?);
            }
            self.resampler.as_mut().expect("resampler").push(&mono)
        };

        if samples.is_empty() {
            return Ok(());
        }
        self.tx
            .send(InputMsg::Samples(samples))
            .await
            .map_err(|_| anyhow!("asr input channel closed"))?;
        Ok(())
    }

    pub async fn read(&mut self) -> Option<AsrSegment> {
        self.rx.recv().await
    }

    pub fn try_read(&mut self) -> Option<AsrSegment> {
        match self.rx.try_recv() {
            Ok(seg) => Some(seg),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => None,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => None,
        }
    }

    pub async fn finish(&mut self) -> Result<()> {
        let _ = self.tx.send(InputMsg::End).await;
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.await.map_err(|e| anyhow!("asr task join failed: {e}"))?;
        Ok(())
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.finish().await
    }
}

impl Drop for SherpaAsrStream {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

#[derive(Debug)]
struct VadSegment {
    samples: Vec<f32>,
    start: f32,
    end: f32,
}

fn enqueue_vad_segments(
    vad: &mut SileroVad,
    segment_tx: &mpsc::Sender<VadSegment>,
    dropped_segments: &mut u64,
    last_drop_log: &mut Instant,
) -> bool {
    while !vad.is_empty() {
        let segment = vad.front();
        vad.pop();

        if segment.samples.is_empty() {
            continue;
        }

        let start_samples = segment.start.max(0) as usize;
        let start = start_samples as f32 / TARGET_SAMPLE_RATE as f32;
        let end = (start_samples + segment.samples.len()) as f32 / TARGET_SAMPLE_RATE as f32;
        let segment = VadSegment {
            samples: segment.samples,
            start,
            end,
        };

        match segment_tx.try_send(segment) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                return true;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                *dropped_segments = dropped_segments.saturating_add(1);
                let now = Instant::now();
                if now.duration_since(*last_drop_log) >= Duration::from_secs(1) {
                    warn!(
                        "asr: dropped {} segments (segment queue full)",
                        *dropped_segments
                    );
                    *dropped_segments = 0;
                    *last_drop_log = now;
                }
            }
        }
    }
    false
}

fn run_inference_loop(
    mut recognizer: SherpaRecognizer,
    mut segment_rx: mpsc::Receiver<VadSegment>,
    out_tx: mpsc::Sender<AsrSegment>,
    log_infer: bool,
) {
    let mut idx: usize = 0;
    while let Some(segment) = segment_rx.blocking_recv() {
        let infer_start = Instant::now();
        let text = recognizer.transcribe(TARGET_SAMPLE_RATE, &segment.samples);
        let infer_ms = infer_start.elapsed().as_millis() as u64;

        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }

        if log_infer {
            info!(
                "asr: segment idx={} start={:.2}s end={:.2}s infer_ms={}",
                idx, segment.start, segment.end, infer_ms
            );
        }

        let seg = AsrSegment {
            text,
            finished: true,
            idx,
            start: segment.start,
            end: segment.end,
            channel: None,
        };
        idx = idx.saturating_add(1);

        if out_tx.blocking_send(seg).is_err() {
            break;
        }
    }
}
