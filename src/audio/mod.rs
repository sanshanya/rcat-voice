use anyhow::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use tokio::time::Instant;

/// 单个音频片段的播放时间信息。
pub struct SegmentPlayback {
    /// 首个音频样本时间戳（流式时可用）。
    pub first_audio_ts: Option<Instant>,
    /// 播放完成时间戳。
    pub play_done_ts: Instant,
    /// 实际播放完成时间戳的可选回传通道。
    pub play_done_rx: Option<oneshot::Receiver<Instant>>,
}

/// Cancellation token for audio synthesis/playback.
#[derive(Debug, Clone)]
pub struct CancelToken {
    epoch: std::sync::Arc<AtomicU64>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            epoch: std::sync::Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn cancel(&self) {
        let _ = self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub fn scope(&self) -> CancelScope {
        CancelScope {
            epoch: self.epoch.load(Ordering::Acquire),
            token: self.epoch.clone(),
        }
    }
}

/// Snapshot of a cancellation token for a single operation.
#[derive(Debug, Clone)]
pub struct CancelScope {
    epoch: u64,
    token: std::sync::Arc<AtomicU64>,
}

impl CancelScope {
    pub fn is_cancelled(&self) -> bool {
        self.token.load(Ordering::Acquire) != self.epoch
    }
}

/// 将 PCM 流式写入音频后端的写入器。
pub trait SegmentWriter: Send {
    /// 写入采样数据，返回实际写入数量。
    fn push(&mut self, samples: &[f32], cancel: &CancelScope) -> usize;
    /// 结束片段并返回播放时间信息。
    fn finish(self: Box<Self>, cancelled: bool) -> SegmentPlayback;
    /// 首个音频样本时间戳（若可用）。
    fn first_audio_ts(&self) -> Option<Instant>;
}

/// 流式播放的音频后端抽象。
pub trait AudioBackend: Send + Sync {
    /// 开始一个新的片段写入。
    fn begin_segment(&self) -> Box<dyn SegmentWriter>;
    /// 停止播放并清空已排队音频。
    fn stop(&self);
    /// 输出采样率（Hz）。
    fn sample_rate(&self) -> u32;
    /// 输出声道数。
    fn channels(&self) -> u16;
    /// 已缓存待播音频时长（毫秒，若支持）。
    fn buffered_ms(&self) -> Option<u64> {
        None
    }
}

/// Audio backend selection and parameters.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub backend: AudioBackendKind,
}

/// Rodio playback configuration.
#[derive(Debug, Clone)]
pub struct RodioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub ring_seconds: u64,
    pub prefill_ms: u64,
}

impl Default for RodioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32_000,
            channels: 1,
            ring_seconds: 60,
            prefill_ms: 50,
        }
    }
}

impl RodioConfig {
    pub fn from_env() -> Self {
        let sample_rate = std::env::var("AUDIO_SAMPLE_RATE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(32_000)
            .clamp(8_000, 96_000);
        let channels = std::env::var("AUDIO_CHANNELS")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(1)
            .clamp(1, 2);
        let ring_seconds = std::env::var("AUDIO_RING_SECONDS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(60)
            .clamp(1, 60);
        let prefill_ms = std::env::var("AUDIO_PREFILL_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(50)
            .min(2000);
        Self {
            sample_rate,
            channels,
            ring_seconds,
            prefill_ms,
        }
    }

    pub fn normalize(mut self) -> Self {
        self.sample_rate = self.sample_rate.clamp(8_000, 96_000);
        self.channels = self.channels.clamp(1, 2);
        self.ring_seconds = self.ring_seconds.clamp(1, 60);
        self.prefill_ms = self.prefill_ms.min(2000);
        self
    }
}

/// Audio backend variants.
#[derive(Debug, Clone)]
pub enum AudioBackendKind {
    Rodio(RodioConfig),
    Wasapi,
    System,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            backend: AudioBackendKind::Rodio(RodioConfig::default()),
        }
    }
}

impl AudioConfig {
    pub fn from_env() -> Result<Self> {
        let backend = std::env::var("AUDIO_BACKEND").unwrap_or_else(|_| "rodio".to_string());
        match backend.as_str() {
            "rodio" => Ok(Self {
                backend: AudioBackendKind::Rodio(RodioConfig::from_env()),
            }),
            "wasapi" => Ok(Self {
                backend: AudioBackendKind::Wasapi,
            }),
            "system" => Ok(Self {
                backend: AudioBackendKind::System,
            }),
            _ => bail!("Unknown AUDIO_BACKEND: {backend}"),
        }
    }
}

pub fn build(config: &AudioConfig) -> Result<Arc<dyn AudioBackend>> {
    match &config.backend {
        AudioBackendKind::Rodio(cfg) => build_rodio_backend(cfg.clone()),
        AudioBackendKind::Wasapi => {
            Err(anyhow::anyhow!("AUDIO_BACKEND=wasapi is not implemented yet"))
        }
        AudioBackendKind::System => {
            Err(anyhow::anyhow!("AUDIO_BACKEND=system is not implemented yet"))
        }
    }
}

pub fn build_from_env() -> Result<Arc<dyn AudioBackend>> {
    let config = AudioConfig::from_env()?;
    build(&config)
}

fn build_rodio_backend(config: RodioConfig) -> Result<Arc<dyn AudioBackend>> {
    let _ = &config;
    #[cfg(feature = "audio-rodio")]
    {
        Ok(Arc::new(rodio::RodioBackend::from_config(config)?))
    }
    #[cfg(not(feature = "audio-rodio"))]
    {
        Err(anyhow::anyhow!(
            "AUDIO_BACKEND=rodio requires the `audio-rodio` feature (use `--features audio-rodio` or `--all-features`)"
        ))
    }
}

#[cfg(feature = "audio-rodio")]
pub mod rodio;
#[cfg(feature = "audio-rodio")]
pub use rodio::RodioBackend;

pub mod wasapi;
pub mod system;
