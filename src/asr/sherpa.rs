use super::AsrSegment;
use super::utils::{LinearResampler, pcm_i16_le_bytes_to_vec, pcm_i16_to_mono_f32};
use anyhow::{Result, anyhow, bail};
use sherpa_rs::paraformer::{ParaformerConfig, ParaformerRecognizer};
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::mpsc;
use tracing::info;

const DEFAULT_MODELS_ROOT: &str = "models";
const DEFAULT_MODEL: &str = "paraformer-zh-small-2024-03-09";
const DEFAULT_LANG: &str = "zh";
const DEFAULT_PROVIDER: &str = "cpu";
const DEFAULT_THREADS: i32 = 2;
const TARGET_SAMPLE_RATE: u32 = 16_000;

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
        let model = match std::env::var("ASR_VAD_PATH") {
            Ok(value) => PathBuf::from(value),
            Err(_) => {
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

        let min_silence_duration = std::env::var("ASR_VAD_MIN_SILENCE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.25)
            .max(0.05);
        let min_speech_duration = std::env::var("ASR_VAD_MIN_SPEECH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.1)
            .max(0.0);
        let max_speech_duration = std::env::var("ASR_VAD_MAX_SPEECH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(30.0)
            .max(0.5);
        let threshold = std::env::var("ASR_VAD_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let window_size = std::env::var("ASR_VAD_WINDOW")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(512)
            .clamp(128, 8192);
        let buffer_size_in_seconds = std::env::var("ASR_VAD_BUFFER_SECONDS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(100.0)
            .clamp(1.0, 600.0);

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
    pub lang: String,
    pub provider: String,
    pub threads: i32,
    pub vad: SherpaVadConfig,
}

impl SherpaAsrConfig {
    pub fn from_env() -> Result<Self> {
        let models_root = std::env::var("ASR_MODELS_ROOT")
            .unwrap_or_else(|_| DEFAULT_MODELS_ROOT.to_string());
        let models_root = PathBuf::from(models_root);

        let model = std::env::var("ASR_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let model = SherpaAsrModel::parse(&model)?;

        let lang = std::env::var("ASR_LANG").unwrap_or_else(|_| DEFAULT_LANG.to_string());
        let provider = std::env::var("ASR_PROVIDER").unwrap_or_else(|_| DEFAULT_PROVIDER.to_string());
        let threads = std::env::var("ASR_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(DEFAULT_THREADS)
            .clamp(1, 32);

        let vad = SherpaVadConfig::from_env(&models_root)?;

        Ok(Self {
            models_root,
            model,
            lang,
            provider,
            threads,
            vad,
        })
    }

    fn resolve_model_paths(&self) -> Result<(PathBuf, PathBuf)> {
        match self.model {
            SherpaAsrModel::SenseVoice
            | SherpaAsrModel::SenseVoiceInt8
            | SherpaAsrModel::SenseVoiceFunAsrNano
            | SherpaAsrModel::SenseVoiceFunAsrNanoInt8 => {
                let (base, fallback) = match self.model {
                    SherpaAsrModel::SenseVoice | SherpaAsrModel::SenseVoiceInt8 => {
                        (self.models_root.join(SENSEVOICE_DIR), None)
                    }
                    SherpaAsrModel::SenseVoiceFunAsrNano => {
                        let fp32 = self.models_root.join(SENSEVOICE_FUNASR_NANO_DIR);
                        let int8 = self.models_root.join(SENSEVOICE_FUNASR_NANO_INT8_DIR);
                        let base = if fp32.exists() { fp32.clone() } else { int8.clone() };
                        let fallback = Some(if fp32.exists() { int8 } else { fp32 });
                        (base, fallback)
                    }
                    SherpaAsrModel::SenseVoiceFunAsrNanoInt8 => {
                        let int8 = self.models_root.join(SENSEVOICE_FUNASR_NANO_INT8_DIR);
                        let fp32 = self.models_root.join(SENSEVOICE_FUNASR_NANO_DIR);
                        let base = if int8.exists() { int8.clone() } else { fp32.clone() };
                        let fallback = Some(if int8.exists() { fp32 } else { int8 });
                        (base, fallback)
                    }
                    _ => unreachable!("model already matched"),
                };
                if !base.exists() {
                    if let Some(fallback) = fallback {
                        bail!(
                            "ASR model dir not found: {} (tried {})",
                            base.display(),
                            fallback.display()
                        );
                    }
                    bail!("ASR model dir not found: {}", base.display());
                }

                let tokens = base.join(SENSEVOICE_TOKENS);
                if !tokens.exists() {
                    bail!("ASR tokens not found: {}", tokens.display());
                }

                let prefer_int8 = matches!(
                    self.model,
                    SherpaAsrModel::SenseVoiceInt8 | SherpaAsrModel::SenseVoiceFunAsrNanoInt8
                );
                let model = if prefer_int8 {
                    let int8 = base.join(SENSEVOICE_MODEL_INT8);
                    if int8.exists() {
                        int8
                    } else {
                        base.join(SENSEVOICE_MODEL)
                    }
                } else {
                    let fp32 = base.join(SENSEVOICE_MODEL);
                    if fp32.exists() {
                        fp32
                    } else {
                        base.join(SENSEVOICE_MODEL_INT8)
                    }
                };
                if !model.exists() {
                    bail!(
                        "ASR model file not found: {} (expected {} or {})",
                        model.display(),
                        SENSEVOICE_MODEL_INT8,
                        SENSEVOICE_MODEL
                    );
                }

                Ok((model, tokens))
            }
            SherpaAsrModel::ParaformerZhSmall => resolve_paraformer_dir(
                &self.models_root.join(PARAFORMER_ZH_SMALL_2024_03_09_DIR),
            ),
            SherpaAsrModel::ParaformerZh => {
                resolve_paraformer_dir(&self.models_root.join(PARAFORMER_ZH_2024_03_09_DIR))
            }
            SherpaAsrModel::ParaformerZhInt8 => resolve_paraformer_dir(
                &self.models_root.join(PARAFORMER_ZH_INT8_2025_10_07_DIR),
            ),
            SherpaAsrModel::ParaformerTrilingual => {
                resolve_paraformer_dir(&self.models_root.join(PARAFORMER_TRILINGUAL_DIR))
            }
            SherpaAsrModel::ParaformerEn => {
                // voiceapi uses `sherpa-onnx-paraformer-en`, but the official archive uses
                // `sherpa-onnx-paraformer-en-2024-03-09`. Try both for convenience.
                let base = if self.models_root.join(PARAFORMER_EN_DIR).exists() {
                    self.models_root.join(PARAFORMER_EN_DIR)
                } else {
                    self.models_root.join(PARAFORMER_EN_DIR_2024_03_09)
                };
                if !base.exists() {
                    bail!(
                        "ASR model dir not found: {} (tried {} and {})",
                        base.display(),
                        self.models_root.join(PARAFORMER_EN_DIR).display(),
                        self.models_root.join(PARAFORMER_EN_DIR_2024_03_09).display()
                    );
                }

                resolve_paraformer_dir(&base)
            }
        }
    }
}

fn resolve_paraformer_dir(base: &Path) -> Result<(PathBuf, PathBuf)> {
    if !base.exists() {
        bail!("ASR model dir not found: {}", base.display());
    }

    let tokens = base.join(PARAFORMER_TOKENS);
    if !tokens.exists() {
        bail!("ASR tokens not found: {}", tokens.display());
    }

    let int8 = base.join(PARAFORMER_MODEL_INT8);
    let fp32 = base.join(PARAFORMER_MODEL);
    let dtype = asr_model_dtype_from_env()?;
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
enum AsrModelDtype {
    Auto,
    Int8,
    Fp32,
}

fn asr_model_dtype_from_env() -> Result<AsrModelDtype> {
    let Ok(value) = std::env::var("ASR_MODEL_DTYPE") else {
        return Ok(AsrModelDtype::Auto);
    };
    let value = value.trim().to_lowercase();
    match value.as_str() {
        "" | "auto" => Ok(AsrModelDtype::Auto),
        "int8" | "i8" => Ok(AsrModelDtype::Int8),
        "fp32" | "f32" | "float" => Ok(AsrModelDtype::Fp32),
        other => bail!("Unknown ASR_MODEL_DTYPE: {other} (expected auto|int8|fp32)"),
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

        let recognizer = match config.model {
            SherpaAsrModel::SenseVoice
            | SherpaAsrModel::SenseVoiceInt8
            | SherpaAsrModel::SenseVoiceFunAsrNano
            | SherpaAsrModel::SenseVoiceFunAsrNanoInt8 => {
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
                SherpaRecognizer::SenseVoice(recognizer)
            }
            SherpaAsrModel::ParaformerZhSmall
            | SherpaAsrModel::ParaformerZh
            | SherpaAsrModel::ParaformerZhInt8
            | SherpaAsrModel::ParaformerTrilingual
            | SherpaAsrModel::ParaformerEn => {
                let recognizer = ParaformerRecognizer::new(ParaformerConfig {
                    model: model_path.to_string_lossy().to_string(),
                    tokens: tokens_path.to_string_lossy().to_string(),
                    provider: Some(config.provider.clone()),
                    num_threads: Some(config.threads),
                    debug: false,
                })
                .map_err(|e| anyhow!("failed to init ParaformerRecognizer: {e}"))?;
                SherpaRecognizer::Paraformer(recognizer)
            }
        };
        let recognizer = Arc::new(StdMutex::new(recognizer));

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

        let join = tokio::spawn(async move {
            let log_infer = std::env::var("ASR_INFER_LOG")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            let vad_chunk_ms = std::env::var("ASR_VAD_CHUNK_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(20)
                .clamp(5, 2000);
            let vad_chunk_samples =
                ((TARGET_SAMPLE_RATE as u64 * vad_chunk_ms) / 1000).max(1) as usize;
            let mut idx = 0usize;
            while let Some(msg) = in_rx.recv().await {
                match msg {
                    InputMsg::Samples(samples) => {
                        if samples.is_empty() {
                            continue;
                        }
                        if samples.len() <= vad_chunk_samples {
                            vad.accept_waveform(samples);
                            if process_vad_segments(
                                &mut vad,
                                &recognizer,
                                &out_tx,
                                &mut idx,
                                log_infer,
                            )
                            .await
                            {
                                break;
                            }
                        } else {
                            for chunk in samples.chunks(vad_chunk_samples) {
                                vad.accept_waveform(chunk.to_vec());
                                if process_vad_segments(
                                    &mut vad,
                                    &recognizer,
                                    &out_tx,
                                    &mut idx,
                                    log_infer,
                                )
                                .await
                                {
                                    return;
                                }
                            }
                        }
                    }
                    InputMsg::End => {
                        vad.flush();
                        let _ =
                            process_vad_segments(&mut vad, &recognizer, &out_tx, &mut idx, log_infer)
                                .await;
                        break;
                    }
                }
            }
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

async fn process_vad_segments(
    vad: &mut SileroVad,
    recognizer: &Arc<StdMutex<SherpaRecognizer>>,
    out_tx: &mpsc::Sender<AsrSegment>,
    idx: &mut usize,
    log_infer: bool,
) -> bool {
    while !vad.is_empty() {
        let segment = vad.front();
        vad.pop();

        if segment.samples.is_empty() {
            continue;
        }

        let start = segment.start as f32 / TARGET_SAMPLE_RATE as f32;
        let end = (segment.start as usize + segment.samples.len()) as f32 / TARGET_SAMPLE_RATE as f32;
        let samples = segment.samples;
        let recognizer = recognizer.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut guard = recognizer
                .lock()
                .map_err(|_| anyhow!("sherpa recognizer lock poisoned"))?;
            let infer_start = std::time::Instant::now();
            let text = guard.transcribe(TARGET_SAMPLE_RATE, &samples);
            let infer_ms = infer_start.elapsed().as_millis() as u64;
            Ok::<_, anyhow::Error>((text, infer_ms))
        })
        .await;

        let (result, infer_ms) = match result {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                tracing::warn!("asr: transcribe failed: {e}");
                continue;
            }
            Err(e) => {
                tracing::warn!("asr: transcribe task failed: {e}");
                continue;
            }
        };

        let text = result.trim().to_string();
        if text.is_empty() {
            continue;
        }

        if log_infer {
            info!(
                "asr: segment idx={} start={:.2}s end={:.2}s infer_ms={}",
                *idx, start, end, infer_ms
            );
        }

        let seg_idx = *idx;
        let seg = AsrSegment {
            text,
            finished: true,
            idx: seg_idx,
            start,
            end,
            channel: None,
        };
        *idx = idx.saturating_add(1);
        if out_tx.send(seg).await.is_err() {
            return true;
        }
    }
    false
}
