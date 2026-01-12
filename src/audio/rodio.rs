use super::{AudioBackend, CancelScope, RodioConfig, SegmentPlayback, SegmentWriter};
use anyhow::Result;
use crossbeam_queue::ArrayQueue;
use rodio::source::SeekError;
use rodio::{OutputStream, OutputStreamBuilder, Source};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{debug, info, warn};

const MARKER_CHECK_INTERVAL: u64 = 256;

pub struct RodioBackend {
    inner: Arc<RodioInner>,
}

struct RodioInner {
    ring: Arc<RingBuffer>,
    playback: Arc<PlaybackState>,
    sample_rate: u32,
    channels: u16,
    _stream: OutputStream,
    active_writer: AtomicBool,
}

struct RodioSegmentWriter {
    inner: Arc<RodioInner>,
    active: bool,
    /// Phase 2: scope bound at creation, gate uses internal scope not external param
    scope: CancelScope,
    start_written: Option<u64>,
    written_total: u64,
    first_audio_ts: Option<Instant>,
}

impl RodioSegmentWriter {
    fn ts_at_sample(&self, stream_start: Instant, sample_idx: u64) -> Instant {
        let samples_per_second = self.inner.sample_rate as f64 * self.inner.channels as f64;
        if samples_per_second <= 0.0 {
            return stream_start;
        }

        let played = self.inner.playback.played();
        let consumed = self.inner.playback.consumed();
        let gap = played.saturating_sub(consumed);
        let mut wall_sample_idx = sample_idx.saturating_add(gap);
        if wall_sample_idx < played {
            wall_sample_idx = played;
        }

        stream_start + Duration::from_secs_f64(wall_sample_idx as f64 / samples_per_second)
    }

    fn estimate_first_audio_ts(&self, stream_start: Instant) -> Option<Instant> {
        Some(self.ts_at_sample(stream_start, self.start_written?))
    }

    fn update_first_audio_ts(&mut self, stream_start: Instant) {
        if self.first_audio_ts.is_none() {
            self.first_audio_ts = self.estimate_first_audio_ts(stream_start);
        }
    }
}

impl RodioBackend {
    pub fn new() -> Result<Self> {
        Self::from_config(RodioConfig::default())
    }

    pub fn from_config(config: RodioConfig) -> Result<Self> {
        let config = config.normalize();
        let stream = OutputStreamBuilder::open_default_stream()
            .map_err(|e| anyhow::anyhow!("open default output stream failed: {e}"))?;
        let ring_capacity =
            (config.sample_rate as u64 * config.ring_seconds * config.channels as u64) as usize;
        let prebuffer_samples =
            config.sample_rate as u64 * config.prefill_ms / 1000 * config.channels as u64;

        let playback = Arc::new(PlaybackState::new(prebuffer_samples));
        let ring = Arc::new(RingBuffer::new(ring_capacity));
        stream.mixer().add(RingBufferSource::new(
            ring.clone(),
            playback.clone(),
            config.sample_rate,
            config.channels,
        ));
        info!(
            "Audio ring buffer: {}s ({} samples)",
            config.ring_seconds, ring_capacity
        );
        info!(
            "Audio format: {}Hz, {}ch",
            config.sample_rate, config.channels
        );
        info!(
            "Audio prebuffer: {}ms ({} samples)",
            config.prefill_ms, prebuffer_samples
        );

        Ok(Self {
            inner: Arc::new(RodioInner {
                ring,
                playback,
                sample_rate: config.sample_rate,
                channels: config.channels,
                _stream: stream,
                active_writer: AtomicBool::new(false),
            }),
        })
    }
}

impl AudioBackend for RodioBackend {
    fn begin_segment(&self, scope: CancelScope) -> Box<dyn SegmentWriter> {
        let active = self
            .inner
            .active_writer
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        if !active {
            warn!(
                "RodioBackend only supports a single active writer; segment audio will be dropped."
            );
        }
        Box::new(RodioSegmentWriter {
            inner: Arc::clone(&self.inner),
            active,
            scope, // Phase 2: bind scope at creation
            start_written: None,
            written_total: 0,
            first_audio_ts: None,
        })
    }

    fn stop(&self) {
        self.inner.playback.reset();
        self.inner.ring.clear();
        self.inner.active_writer.store(false, Ordering::Release);
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    fn channels(&self) -> u16 {
        self.inner.channels
    }

    fn buffered_ms(&self) -> Option<u64> {
        let queued = self.inner.ring.len() as u64;
        let denom = self.inner.sample_rate as u64 * self.inner.channels as u64;
        if denom == 0 {
            return None;
        }
        Some(queued.saturating_mul(1000) / denom)
    }
}

impl SegmentWriter for RodioSegmentWriter {
    fn first_audio_ts(&self) -> Option<Instant> {
        self.first_audio_ts
    }

    fn push(&mut self, samples: &[f32], _cancel: &CancelScope) -> usize {
        if !self.active {
            return 0;
        }
        // Phase 2: Generation Gate - use internal scope, ignore external param
        // This ensures old-generation writers cannot write to new-generation output
        if self.scope.is_cancelled() {
            return 0;
        }
        if samples.is_empty() {
            return 0;
        }

        if self.start_written.is_none() {
            self.start_written = Some(self.inner.playback.begin_write());
        }

        // Use internal scope for ring buffer push
        let written = self.inner.ring.push_blocking(samples, &self.scope);
        if written == 0 {
            return 0;
        }
        self.inner.playback.on_write(written as u64);
        self.written_total += written as u64;

        if self.first_audio_ts.is_none() {
            let stream_start = self
                .inner
                .playback
                .try_start(self.inner.ring.len() as u64)
                .or_else(|| self.inner.playback.stream_start());
            if let Some(stream_start) = stream_start {
                self.update_first_audio_ts(stream_start);
            }
        }

        written
    }

    fn finish(mut self: Box<Self>, cancelled: bool) -> SegmentPlayback {
        let now = Instant::now();
        if self.active {
            self.inner.active_writer.store(false, Ordering::Release);
        }
        if cancelled || self.written_total == 0 {
            return SegmentPlayback {
                first_audio_ts: self.first_audio_ts,
                play_done_ts: now,
                play_done_rx: None,
            };
        }

        let stream_start = self
            .inner
            .playback
            .stream_start()
            .unwrap_or_else(|| self.inner.playback.force_start());
        self.update_first_audio_ts(stream_start);

        let start_at = self
            .start_written
            .unwrap_or_else(|| self.inner.playback.written());
        let end_written = start_at + self.written_total;
        let play_done_ts = self.ts_at_sample(stream_start, end_written);

        let (tx, rx) = oneshot::channel();
        self.inner.playback.push_marker(end_written, tx);

        SegmentPlayback {
            first_audio_ts: self.first_audio_ts,
            play_done_ts,
            play_done_rx: Some(rx),
        }
    }
}

impl Drop for RodioSegmentWriter {
    fn drop(&mut self) {
        if self.active {
            self.inner.active_writer.store(false, Ordering::Release);
        }
    }
}

struct PlayMarker {
    end_sample: u64,
    done_tx: oneshot::Sender<Instant>,
}

struct PlaybackState {
    started: AtomicBool,
    stream_start: StdMutex<Option<Instant>>,
    played_samples: AtomicU64,
    written_samples: AtomicU64,
    consumed_samples: AtomicU64,
    prebuffer_samples: u64,
    markers: StdMutex<VecDeque<PlayMarker>>,
}

impl PlaybackState {
    fn new(prebuffer_samples: u64) -> Self {
        Self {
            started: AtomicBool::new(false),
            stream_start: StdMutex::new(None),
            played_samples: AtomicU64::new(0),
            written_samples: AtomicU64::new(0),
            consumed_samples: AtomicU64::new(0),
            prebuffer_samples,
            markers: StdMutex::new(VecDeque::new()),
        }
    }

    fn is_started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    fn begin_write(&self) -> u64 {
        let consumed = self.consumed_samples.load(Ordering::Acquire);
        let mut written = self.written_samples.load(Ordering::Acquire);
        while written < consumed {
            match self.written_samples.compare_exchange(
                written,
                consumed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return consumed,
                Err(current) => written = current,
            }
        }
        written
    }

    fn on_write(&self, samples: u64) {
        let _ = self.written_samples.fetch_add(samples, Ordering::AcqRel);
    }

    fn consumed(&self) -> u64 {
        self.consumed_samples.load(Ordering::Acquire)
    }

    fn played(&self) -> u64 {
        self.played_samples.load(Ordering::Acquire)
    }

    fn written(&self) -> u64 {
        self.written_samples.load(Ordering::Acquire)
    }

    fn on_output_sample(&self, consumed: bool) {
        let played = self.played_samples.fetch_add(1, Ordering::Relaxed) + 1;
        if consumed {
            let _ = self.consumed_samples.fetch_add(1, Ordering::Relaxed);
        }
        if played % MARKER_CHECK_INTERVAL == 0 {
            let consumed = self.consumed_samples.load(Ordering::Acquire);
            self.check_markers(consumed);
        }
    }

    fn reset(&self) {
        self.started.store(false, Ordering::Release);
        *self
            .stream_start
            .lock()
            .expect("stream start lock poisoned") = None;
        self.played_samples.store(0, Ordering::Release);
        self.written_samples.store(0, Ordering::Release);
        self.consumed_samples.store(0, Ordering::Release);
        self.clear_markers();
    }

    fn try_start(&self, buffered_samples: u64) -> Option<Instant> {
        if self.started.load(Ordering::Acquire) {
            return self.stream_start();
        }
        if buffered_samples < self.prebuffer_samples {
            return None;
        }
        let now = Instant::now();
        let mut guard = self
            .stream_start
            .lock()
            .expect("stream start lock poisoned");
        if guard.is_none() {
            *guard = Some(now);
        }
        self.started.store(true, Ordering::Release);
        *guard
    }

    fn force_start(&self) -> Instant {
        if let Some(ts) = self.stream_start() {
            self.started.store(true, Ordering::Release);
            return ts;
        }
        let now = Instant::now();
        let mut guard = self
            .stream_start
            .lock()
            .expect("stream start lock poisoned");
        *guard = Some(now);
        self.started.store(true, Ordering::Release);
        now
    }

    fn stream_start(&self) -> Option<Instant> {
        *self
            .stream_start
            .lock()
            .expect("stream start lock poisoned")
    }

    fn push_marker(&self, end_sample: u64, done_tx: oneshot::Sender<Instant>) {
        let consumed = self.consumed_samples.load(Ordering::Acquire);
        if consumed >= end_sample {
            let _ = done_tx.send(Instant::now());
            return;
        }
        let mut markers = self.markers.lock().expect("playback marker lock poisoned");
        markers.push_back(PlayMarker {
            end_sample,
            done_tx,
        });
    }

    fn clear_markers(&self) {
        let mut markers = self.markers.lock().expect("playback marker lock poisoned");
        while let Some(marker) = markers.pop_front() {
            let _ = marker.done_tx.send(Instant::now());
        }
    }

    fn check_markers(&self, consumed_samples: u64) {
        let mut markers = self.markers.lock().expect("playback marker lock poisoned");
        while let Some(marker) = markers.front() {
            if marker.end_sample > consumed_samples {
                break;
            }
            if let Some(marker) = markers.pop_front() {
                let _ = marker.done_tx.send(Instant::now());
            }
        }
    }
}

struct RingBuffer {
    queue: ArrayQueue<f32>,
    full_count: AtomicU64,
    blocked_us: AtomicU64,
}

impl RingBuffer {
    fn new(cap: usize) -> Self {
        Self {
            queue: ArrayQueue::new(cap),
            full_count: AtomicU64::new(0),
            blocked_us: AtomicU64::new(0),
        }
    }

    fn push_blocking(&self, data: &[f32], cancel: &CancelScope) -> usize {
        let start = std::time::Instant::now();
        let mut written = 0usize;
        let mut backoff = 0u32;
        let mut full_events = 0u64;
        while written < data.len() {
            if cancel.is_cancelled() {
                break;
            }
            match self.queue.push(data[written]) {
                Ok(()) => {
                    written += 1;
                    backoff = 0;
                }
                Err(_) => {
                    full_events += 1;
                    let delay_us = 100u64 << backoff.min(3);
                    std::thread::sleep(Duration::from_micros(delay_us));
                    backoff = backoff.saturating_add(1);
                }
            }
        }
        // Track metrics
        if full_events > 0 {
            self.full_count.fetch_add(full_events, Ordering::Relaxed);
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us >= 1000 {
                self.blocked_us.fetch_add(elapsed_us, Ordering::Relaxed);
                if std::env::var("AUDIO_RING_METRICS").is_ok() {
                    debug!(
                        "ring_buffer: blocked {}us ({} full events)",
                        elapsed_us, full_events
                    );
                }
            }
        }
        written
    }

    fn pop(&self) -> Option<f32> {
        self.queue.pop()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn clear(&self) {
        while self.queue.pop().is_some() {}
    }
}

struct RingBufferSource {
    ring: Arc<RingBuffer>,
    playback: Arc<PlaybackState>,
    sample_rate: u32,
    channels: u16,
}

impl RingBufferSource {
    fn new(
        ring: Arc<RingBuffer>,
        playback: Arc<PlaybackState>,
        sample_rate: u32,
        channels: u16,
    ) -> Self {
        Self {
            ring,
            playback,
            sample_rate,
            channels,
        }
    }
}

impl Iterator for RingBufferSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.playback.is_started() {
            return Some(0.0);
        }
        let mut consumed_sample = false;
        let sample = match self.ring.pop() {
            Some(value) => {
                consumed_sample = true;
                value
            }
            None => 0.0,
        };
        self.playback.on_output_sample(consumed_sample);
        Some(sample)
    }
}

impl Source for RingBufferSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }

    fn try_seek(&mut self, _: Duration) -> Result<(), SeekError> {
        Err(SeekError::NotSupported {
            underlying_source: std::any::type_name::<Self>(),
        })
    }
}
