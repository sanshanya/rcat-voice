use anyhow::{Context, Result, bail};
use crossbeam_queue::ArrayQueue;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::internal::env;

#[derive(Debug, Clone)]
pub struct MicConfig {
    /// Preferred ASR feed size (ms). Used by consumers to chunk samples.
    pub feed_ms: u64,
    /// Ring buffer capacity in seconds.
    pub ring_seconds: u64,
    /// Optional input device name substring match.
    pub device_hint: Option<String>,
    /// Optional fixed buffer size for the platform audio backend (frames).
    pub buffer_frames: Option<u32>,
}

impl Default for MicConfig {
    fn default() -> Self {
        Self {
            feed_ms: 20,
            ring_seconds: 30,
            device_hint: None,
            buffer_frames: None,
        }
    }
}

impl MicConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.feed_ms = env::u64_clamped("ASR_FEED_MS", cfg.feed_ms, 5, 200);
        cfg.ring_seconds = env::u64_clamped("ASR_MIC_RING_SECONDS", cfg.ring_seconds, 1, 120);
        cfg.device_hint = env::string("ASR_MIC_DEVICE");
        cfg.buffer_frames = env::get::<u32>("ASR_MIC_BUFFER_FRAMES").filter(|v| *v > 0);
        cfg
    }
}

pub struct MicStream {
    device_name: String,
    sample_rate: u32,
    channels: u16,
    feed_ms: u64,
    queue: Arc<ArrayQueue<i16>>,
    dropped: Arc<AtomicU64>,
    _stream: cpal::Stream,
}

impl MicStream {
    pub fn from_env() -> Result<Self> {
        Self::new(MicConfig::from_env())
    }

    pub fn new(config: MicConfig) -> Result<Self> {
        use cpal::traits::{DeviceTrait, StreamTrait};

        let host = cpal::default_host();
        let device = select_input_device(&host, config.device_hint)?;
        let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

        let supported_config = device
            .default_input_config()
            .context("failed to get default input config")?;
        let sample_format = supported_config.sample_format();
        let mut stream_config: cpal::StreamConfig = supported_config.into();

        if let Some(frames) = config.buffer_frames.filter(|v| *v > 0) {
            stream_config.buffer_size = cpal::BufferSize::Fixed(frames);
        }

        let sample_rate = stream_config.sample_rate;
        let channels = stream_config.channels;
        if sample_rate == 0 || channels == 0 {
            bail!(
                "Invalid input audio format: {}Hz/{}ch",
                sample_rate,
                channels
            );
        }

        let ring_capacity = (sample_rate as usize)
            .saturating_mul(channels as usize)
            .saturating_mul(config.ring_seconds as usize)
            .max(1024);

        let queue: Arc<ArrayQueue<i16>> = Arc::new(ArrayQueue::new(ring_capacity));
        let dropped = Arc::new(AtomicU64::new(0));

        let stream = build_cpal_stream(
            &device,
            &stream_config,
            sample_format,
            queue.clone(),
            dropped.clone(),
        )
        .context("failed to build input stream")?;
        stream.play().context("failed to start input stream")?;

        tracing::info!(
            "mic: device={} format={:?} input={}Hz/{}ch feed_ms={} ring={}s cap_samples={}",
            device_name,
            sample_format,
            sample_rate,
            channels,
            config.feed_ms,
            config.ring_seconds,
            ring_capacity
        );

        Ok(Self {
            device_name,
            sample_rate,
            channels,
            feed_ms: config.feed_ms,
            queue,
            dropped,
            _stream: stream,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn feed_ms(&self) -> u64 {
        self.feed_ms
    }

    pub fn take_dropped_samples(&self) -> u64 {
        self.dropped.swap(0, Ordering::AcqRel)
    }

    pub fn try_pop_sample(&self) -> Option<i16> {
        self.queue.pop()
    }

    /// Extract a thread-safe handle for sample access.
    /// The handle is Send + Sync and can be used from async tasks.
    pub fn handle(&self) -> MicHandle {
        MicHandle {
            queue: self.queue.clone(),
            dropped: self.dropped.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            feed_ms: self.feed_ms,
        }
    }
}

/// Thread-safe handle for microphone sample access.
/// This is `Send + Sync` and can be used from async tasks.
#[derive(Clone)]
pub struct MicHandle {
    queue: Arc<ArrayQueue<i16>>,
    dropped: Arc<AtomicU64>,
    sample_rate: u32,
    channels: u16,
    feed_ms: u64,
}

impl MicHandle {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn feed_ms(&self) -> u64 {
        self.feed_ms
    }

    pub fn take_dropped_samples(&self) -> u64 {
        self.dropped.swap(0, Ordering::AcqRel)
    }

    pub fn try_pop_sample(&self) -> Option<i16> {
        self.queue.pop()
    }
}

pub fn select_input_device(host: &cpal::Host, hint: Option<String>) -> Result<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let mut devices: Vec<(String, cpal::Device)> = Vec::new();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        devices.push((name, device));
    }

    if let Some(hint) = hint.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) {
        let needle = hint.to_lowercase();
        if let Some(index) = devices
            .iter()
            .position(|(name, _)| name.to_lowercase().contains(&needle))
        {
            return Ok(devices.swap_remove(index).1);
        }
        let available = devices
            .iter()
            .map(|(name, _)| format!("- {name}"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!("ASR_MIC_DEVICE={hint} did not match any input device. Available:\n{available}");
    }

    host.default_input_device()
        .context("no default input device (set ASR_MIC_DEVICE to select one)")
}

fn build_cpal_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    queue: Arc<ArrayQueue<i16>>,
    dropped: Arc<AtomicU64>,
) -> Result<cpal::Stream> {
    use cpal::traits::DeviceTrait;

    let err_fn = |err| tracing::error!("cpal input stream error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => {
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        let scaled = (sample * 32767.0).round();
                        let clamped = scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                        if queue.push(clamped).is_err() {
                            dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        if queue.push(sample).is_err() {
                            dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        let sample = sample as i32 - 32768;
                        let sample = sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        if queue.push(sample).is_err() {
                            dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => bail!("Unsupported input sample format: {other:?}"),
    }
}
