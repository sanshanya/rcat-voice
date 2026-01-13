use anyhow::{Context, Result, anyhow, bail};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::sync::mpsc;

use crate::asr::utils::{LinearResampler, pcm_i16_to_mono_f32};
use crate::internal::{env, models, ort_log};

const WINDOW_SECONDS: usize = 8;

#[derive(Debug, Clone)]
pub struct SmartTurnConfig {
    pub model: PathBuf,
    pub threshold: f32,
}

impl SmartTurnConfig {
    pub fn from_env() -> Result<Self> {
        let model = env::string("SMART_TURN_MODEL")
            .map(PathBuf::from)
            .or_else(models::turn_dir)
            .context(
                "SMART_TURN_MODEL is required (path to smart-turn-v3*.onnx, a directory containing it, or set RCAT_MODELS_DIR)",
            )?;
        let model = resolve_smart_turn_model_path(model)?;
        let threshold = env::f32_clamped("SMART_TURN_THRESHOLD", 0.5, 0.0, 1.0);
        Ok(Self { model, threshold })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SmartTurnDecision {
    pub probability: f32,
    pub endpoint: bool,
}

#[derive(Clone)]
pub struct SmartTurnModel {
    predictor: Arc<StdMutex<smart_turn_rs::SmartTurnPredictor>>,
}

impl SmartTurnModel {
    pub fn new(model: impl AsRef<Path>) -> Result<Self> {
        ort_log::apply_from_env();

        let model = model.as_ref();
        if !model.exists() {
            bail!("SMART_TURN_MODEL not found: {}", model.display());
        }
        if model.is_dir() {
            bail!(
                "SMART_TURN_MODEL must point to a .onnx file, got a directory: {}",
                model.display()
            );
        }
        if !is_model_path(model) {
            bail!(
                "SMART_TURN_MODEL must point to a .onnx file, got: {}",
                model.display()
            );
        }

        #[cfg(all(target_os = "windows", feature = "gpt-sovits"))]
        {
            if is_smart_turn_cpu_model_path(model)
                && is_gpt_sovits_backend_selected()
                && !env::bool01("SMART_TURN_ALLOW_CPU_MODEL", false)
            {
                bail!(
                    "SMART_TURN_MODEL is set to a CPU smart-turn model ({}) while TTS_BACKEND=gpt-sovits.\n\
This combination has been observed to crash on Windows (STATUS_HEAP_CORRUPTION 0xc0000374).\n\
Fix: switch to `smart-turn-*-gpu.onnx`, or set `SMART_TURN_ALLOW_CPU_MODEL=1` to force CPU anyway.",
                    model.display()
                );
            }
        }

        let predictor = smart_turn_rs::SmartTurnPredictor::new(model)
            .with_context(|| format!("failed to load smart-turn model: {}", model.display()))?;

        Ok(Self {
            predictor: Arc::new(StdMutex::new(predictor)),
        })
    }

    pub fn from_env() -> Result<Self> {
        let model = model_path_from_env()?;
        Self::new(model)
    }

    pub fn predict_probability(&self, audio_16k_mono_right_padded: &[f32]) -> Result<f32> {
        let features = smart_turn_rs::features::log_mel_spectrogram(audio_16k_mono_right_padded)?;
        let mut predictor = self
            .predictor
            .lock()
            .map_err(|_| anyhow!("smart_turn predictor lock poisoned"))?;
        let result = predictor.predict(features)?;
        Ok(result.probability)
    }
}

#[derive(Debug, Clone, Copy)]
struct InputFormat {
    sample_rate: u32,
    channels: u16,
}

#[derive(Debug)]
struct TurnAudioBuffer {
    max_samples: usize,
    data: VecDeque<f32>,
}

impl TurnAudioBuffer {
    fn new(max_samples: usize) -> Self {
        Self {
            max_samples,
            data: VecDeque::with_capacity(max_samples),
        }
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn push(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let keep = self.max_samples;
        if samples.len() >= keep {
            self.data.clear();
            self.data
                .extend(samples.iter().copied().skip(samples.len() - keep));
            return;
        }

        self.data.extend(samples.iter().copied());
        while self.data.len() > keep {
            let _ = self.data.pop_front();
        }
    }

    fn snapshot_right_padded(&self) -> Vec<f32> {
        let keep = self.max_samples;
        let len = self.data.len();
        if len >= keep {
            return self.data.iter().copied().collect();
        }

        let mut out = vec![0.0f32; keep];
        let offset = keep - len;
        for (i, v) in self.data.iter().copied().enumerate() {
            out[offset + i] = v;
        }
        out
    }
}

pub struct SmartTurnDetector {
    model: SmartTurnModel,
    threshold: f32,
    input_format: Option<InputFormat>,
    resampler: Option<LinearResampler>,
    audio: TurnAudioBuffer,
}

impl SmartTurnDetector {
    pub fn new(config: SmartTurnConfig) -> Result<Self> {
        let model = SmartTurnModel::new(&config.model)?;
        let max_samples = smart_turn_rs::features::SAMPLE_RATE * WINDOW_SECONDS;
        Ok(Self {
            model,
            threshold: config.threshold,
            input_format: None,
            resampler: None,
            audio: TurnAudioBuffer::new(max_samples),
        })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(SmartTurnConfig::from_env()?)
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn model(&self) -> SmartTurnModel {
        self.model.clone()
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Reset accumulated turn audio (start a new user turn).
    ///
    /// Also clears input_format and resampler to allow format changes
    /// (e.g., warmup at 16kHz → microphone at 48kHz).
    pub fn reset(&mut self) {
        self.audio.clear();
        self.input_format = None;
        self.resampler = None;
    }

    /// Returns the current 8s (right-aligned) audio window as 16kHz mono samples, padded with zeros.
    pub fn snapshot_audio(&self) -> Vec<f32> {
        self.audio.snapshot_right_padded()
    }

    /// Append raw PCM to the current turn buffer.
    ///
    /// The detector will convert input to 16kHz mono internally (as expected by Smart Turn).
    pub fn push_pcm_i16(&mut self, pcm: &[i16], sample_rate: u32, channels: u16) -> Result<()> {
        if sample_rate == 0 {
            bail!("sample_rate must be > 0");
        }
        if channels == 0 {
            bail!("channels must be >= 1");
        }

        let input_format = self.input_format.get_or_insert(InputFormat {
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
        let samples = if sample_rate as usize == smart_turn_rs::features::SAMPLE_RATE {
            mono
        } else {
            if self.resampler.is_none() {
                self.resampler = Some(LinearResampler::new(
                    sample_rate,
                    smart_turn_rs::features::SAMPLE_RATE as u32,
                )?);
            }
            self.resampler.as_mut().expect("resampler").push(&mono)
        };

        self.audio.push(&samples);
        Ok(())
    }

    /// Run Smart Turn on the current (right-aligned) 8s audio window.
    ///
    /// Intended usage: call this when VAD detects a pause/silence.
    pub fn predict_endpoint(&mut self) -> Result<SmartTurnDecision> {
        let audio = self.audio.snapshot_right_padded();
        let p = self.model.predict_probability(&audio)?;
        Ok(SmartTurnDecision {
            probability: p,
            endpoint: p >= self.threshold,
        })
    }
}

/// Convenience: load a Smart Turn model from the default env var and return the resolved path.
pub fn model_path_from_env() -> Result<PathBuf> {
    let model = env::string("SMART_TURN_MODEL")
        .map(PathBuf::from)
        .or_else(models::turn_dir)
        .context(
            "SMART_TURN_MODEL is required (path to smart-turn-v3*.onnx, a directory containing it, or set RCAT_MODELS_DIR)",
        )?;
    resolve_smart_turn_model_path(model)
}

pub fn is_model_path(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
}

fn resolve_smart_turn_model_path(model: PathBuf) -> Result<PathBuf> {
    if model.is_file() {
        if !is_model_path(&model) {
            bail!(
                "SMART_TURN_MODEL must point to a .onnx file, got: {}",
                model.display()
            );
        }
        return Ok(model);
    }

    if model.is_dir() {
        return resolve_smart_turn_model_in_dir(&model);
    }

    if !model.exists() {
        bail!("SMART_TURN_MODEL not found: {}", model.display());
    }
    bail!(
        "SMART_TURN_MODEL must point to a .onnx file (or a directory containing smart-turn*.onnx), got: {}",
        model.display()
    );
}

fn resolve_smart_turn_model_in_dir(dir: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::<PathBuf>::new();
    for entry in std::fs::read_dir(dir).with_context(|| {
        format!(
            "failed to read SMART_TURN_MODEL directory: {}",
            dir.display()
        )
    })? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || !is_model_path(&path) {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if file_name.contains("smart-turn") || file_name.contains("smart_turn") {
            candidates.push(path);
        }
    }

    if candidates.is_empty() {
        bail!(
            "SMART_TURN_MODEL points to a directory but no smart-turn*.onnx file was found: {}",
            dir.display()
        );
    }
    if candidates.len() == 1 {
        return Ok(candidates.swap_remove(0));
    }

    let prefer = env::string("SMART_TURN_VARIANT")
        .unwrap_or_else(|| "gpu".to_string())
        .to_lowercase();
    let prefer = prefer.trim();
    let pick = match prefer {
        "gpu" => pick_smart_turn_variant(&candidates, "gpu"),
        "cpu" => pick_smart_turn_variant(&candidates, "cpu"),
        other => bail!("SMART_TURN_VARIANT must be `cpu` or `gpu`, got: {other}"),
    };
    if let Some(path) = pick {
        return Ok(path);
    }

    candidates.sort();
    let list = candidates
        .iter()
        .map(|p| format!("- {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    bail!(
        "SMART_TURN_MODEL points to a directory with multiple candidates and none matched SMART_TURN_VARIANT={prefer}. Please set SMART_TURN_MODEL to an explicit file path.\n{list}"
    );
}

fn pick_smart_turn_variant(candidates: &[PathBuf], needle: &str) -> Option<PathBuf> {
    let mut matches = candidates
        .iter()
        .filter_map(|path| {
            let file_name = path.file_name()?.to_str()?.to_lowercase();
            if file_name.contains(needle) {
                Some(path.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.pop()
}

#[cfg(all(target_os = "windows", feature = "gpt-sovits"))]
fn is_gpt_sovits_backend_selected() -> bool {
    let Some(value) = env::string("TTS_BACKEND") else {
        return false;
    };
    matches!(
        value.trim().to_lowercase().as_str(),
        "gpt-sovits" | "gpt_sovits" | "gsv"
    )
}

#[cfg(all(target_os = "windows", feature = "gpt-sovits"))]
fn is_smart_turn_cpu_model_path(model: &Path) -> bool {
    let file_name = model
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or_default()
        .to_lowercase();
    (file_name.contains("smart-turn") || file_name.contains("smart_turn"))
        && file_name.contains("cpu")
        && !file_name.contains("gpu")
}

// ============================================================================
// TurnBoundaryDetector Implementation for SmartTurnDetector
// ============================================================================

use super::types::{AudioFrameRef, TurnBoundaryDetector, TurnDetectorConfig, TurnEvent};
use crate::asr::VadEvent;
use smallvec::SmallVec;
use tokio::time::Instant;

/// Smart Turn 端点检测器（实现 TurnBoundaryDetector trait）
///
/// 结合 VAD 事件做静音门控，在静音期间调用 Smart Turn 模型推理端点概率。
pub struct SmartTurnBoundaryDetector {
    detector: SmartTurnDetector,
    config: TurnDetectorConfig,
    /// 是否在说话
    speaking: bool,
    /// 静音开始时刻
    silence_start_ts: Option<Instant>,
    /// 上次评估时刻
    last_eval_ts: Option<Instant>,
    /// 端点是否已 armed（模型判定为端点）
    endpoint_armed: bool,
    /// 是否已提交
    committed: bool,

    eval_generation: u64,
    eval_inflight: bool,
    eval_tx: mpsc::Sender<SmartTurnEvalResult>,
    eval_rx: mpsc::Receiver<SmartTurnEvalResult>,
}

impl SmartTurnBoundaryDetector {
    pub fn new(detector: SmartTurnDetector, config: TurnDetectorConfig) -> Self {
        let (eval_tx, eval_rx) = mpsc::channel();
        Self {
            detector,
            config,
            speaking: false,
            silence_start_ts: None,
            last_eval_ts: None,
            endpoint_armed: false,
            committed: false,
            eval_generation: 0,
            eval_inflight: false,
            eval_tx,
            eval_rx,
        }
    }

    pub fn from_env() -> Result<Self> {
        let detector = SmartTurnDetector::from_env()?;
        let min_silence_ms = env::u64_clamped("SMART_TURN_MIN_SILENCE_MS", 400, 50, 2000);
        let commit_ms = env::u64_clamped("SMART_TURN_COMMIT_MS", 300, 0, 1000);
        let min_force_end_ms = min_silence_ms.saturating_add(commit_ms);
        let config = TurnDetectorConfig {
            min_silence_ms,
            commit_ms,
            force_end_ms: env::u64_clamped(
                "SMART_TURN_FORCE_END_MS",
                2000,
                min_force_end_ms,
                60_000,
            ),
            eval_interval_ms: env::u64_clamped("SMART_TURN_EVAL_INTERVAL_MS", 200, 10, 500),
            silence_threshold: env::u16_clamped("TURN_SILENCE_THRESHOLD", 200, 0, 20_000),
        };
        Ok(Self::new(detector, config))
    }

    /// 获取内部的 SmartTurnDetector
    pub fn inner(&self) -> &SmartTurnDetector {
        &self.detector
    }

    /// 获取内部的 SmartTurnDetector (mut)
    pub fn inner_mut(&mut self) -> &mut SmartTurnDetector {
        &mut self.detector
    }

    fn trailing_silence_ms(&self, now: Instant) -> u64 {
        self.silence_start_ts
            .map(|start| now.saturating_duration_since(start).as_millis() as u64)
            .unwrap_or(0)
    }

    fn should_eval(&self, now: Instant) -> bool {
        self.last_eval_ts
            .map(|t| {
                now.saturating_duration_since(t).as_millis() as u64 >= self.config.eval_interval_ms
            })
            .unwrap_or(true)
    }

    fn drain_eval_results(&mut self) {
        while let Ok(result) = self.eval_rx.try_recv() {
            if result.generation != self.eval_generation {
                continue;
            }
            self.eval_inflight = false;

            if self.committed || self.speaking || self.silence_start_ts.is_none() {
                continue;
            }

            match result.decision {
                Ok(decision) => {
                    self.endpoint_armed = decision.endpoint;
                }
                Err(err) => {
                    tracing::debug!("SmartTurn inference failed: {err}");
                }
            }
        }
    }
}

impl TurnBoundaryDetector for SmartTurnBoundaryDetector {
    fn push_audio(&mut self, frame: AudioFrameRef<'_>, _out: &mut SmallVec<[TurnEvent; 4]>) {
        if self.committed {
            return;
        }
        // 总是推送音频到 Smart Turn 模型缓冲区
        let _ = self
            .detector
            .push_pcm_i16(frame.samples, frame.sample_rate, frame.channels);
    }

    fn push_vad(&mut self, event: VadEvent, out: &mut SmallVec<[TurnEvent; 4]>) {
        if self.committed {
            return;
        }
        match event {
            VadEvent::SpeechStart { ts } => {
                self.speaking = true;
                self.silence_start_ts = None;
                self.last_eval_ts = None;
                self.endpoint_armed = false;

                out.push(TurnEvent::speech_start(ts));
            }
            VadEvent::SpeechEnd { ts, .. } => {
                self.speaking = false;
                self.silence_start_ts = Some(ts);
                self.last_eval_ts = None;
                self.endpoint_armed = false;

                out.push(TurnEvent::speech_end(ts));
            }
        }
    }

    fn tick(&mut self, now: Instant, out: &mut SmallVec<[TurnEvent; 4]>) {
        self.drain_eval_results();
        if self.committed {
            return;
        }
        // 只在有语音且当前静音时评估
        if self.speaking || self.silence_start_ts.is_none() {
            return;
        }

        let silence_ms = self.trailing_silence_ms(now);

        // 强制结束检查
        if silence_ms >= self.config.force_end_ms {
            out.push(TurnEvent::turn_committed(now));
            self.committed = true;
            return;
        }

        // 只在超过 min_silence 后开始评估
        if silence_ms < self.config.min_silence_ms {
            return;
        }

        // 评估间隔检查 + Smart Turn 推理
        if self.should_eval(now) && !self.endpoint_armed {
            self.last_eval_ts = Some(now);

            if !self.eval_inflight {
                let audio = self.detector.snapshot_audio();
                let model = self.detector.model();
                let threshold = self.detector.threshold();
                let tx = self.eval_tx.clone();
                let generation = self.eval_generation;

                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        self.eval_inflight = true;
                        handle.spawn_blocking(move || {
                            let decision = model.predict_probability(&audio).map(|p| SmartTurnDecision {
                                probability: p,
                                endpoint: p >= threshold,
                            });
                            let _ = tx.send(SmartTurnEvalResult { generation, decision });
                        });
                    }
                    Err(_) => {
                        // No Tokio runtime in context: fall back to synchronous inference.
                        if let Ok(decision) = self.detector.predict_endpoint() {
                            self.endpoint_armed = decision.endpoint;
                        }
                    }
                }
            }
        }

        // 端点已 armed 且达到 commit 阈值
        if self.endpoint_armed {
            let commit_threshold = self
                .config
                .min_silence_ms
                .saturating_add(self.config.commit_ms);
            if silence_ms >= commit_threshold {
                out.push(TurnEvent::turn_committed(now));
                self.committed = true;
            }
        }
    }

    fn reset(&mut self) {
        self.eval_generation = self.eval_generation.wrapping_add(1);
        self.eval_inflight = false;
        while self.eval_rx.try_recv().is_ok() {}
        self.detector.reset();
        self.speaking = false;
        self.silence_start_ts = None;
        self.last_eval_ts = None;
        self.endpoint_armed = false;
        self.committed = false;
    }
}

impl SmartTurnBoundaryDetector {
    /// 获取配置
    pub fn config(&self) -> &TurnDetectorConfig {
        &self.config
    }
}

struct SmartTurnEvalResult {
    generation: u64,
    decision: Result<SmartTurnDecision>,
}
