use crate::generator::{SynthesizedAudio, TtsEngine, TtsMetrics};
use crate::internal::env;
use crate::metrics::{MetricEvent, MetricEventKind, MetricsSink, default_sink};
use crate::tokenizer::Segment;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::{Id, JoinSet};
use tokio::time::{Instant, sleep_until};
use tracing::warn;

/// Pipeline scheduling knobs.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub parallel_synth: bool,
    pub synth_inflight: usize,
    pub backlog_limit: usize,
    pub synth_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel_synth: true,
            synth_inflight: 1,
            backlog_limit: 32,
            synth_timeout_ms: 30_000,
        }
    }
}

impl PipelineConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        cfg.parallel_synth = env::bool01("TTS_PARALLEL_SYNTH", cfg.parallel_synth);
        cfg.synth_inflight = env::usize_clamped("TTS_SYNTH_INFLIGHT", cfg.synth_inflight, 1, 8);
        cfg.backlog_limit = env::usize_clamped("TTS_BACKLOG_LIMIT", cfg.backlog_limit, 1, 4096);
        cfg.synth_timeout_ms =
            env::u64_clamped("TTS_SYNTH_TIMEOUT_MS", cfg.synth_timeout_ms, 0, 10 * 60_000);
        cfg
    }
}

pub struct Pipeline {
    chunk_rx: mpsc::Receiver<Segment>,
    engine: Arc<dyn TtsEngine>,
    config: PipelineConfig,
    metrics: Arc<dyn MetricsSink>,
}

struct SynthOutcome {
    seq: u64,
    segment: Segment,
    result: std::result::Result<Option<SynthesizedAudio>, anyhow::Error>,
}

struct SynthJob {
    seq: u64,
    segment: Segment,
}

impl Pipeline {
    pub fn new(
        chunk_rx: mpsc::Receiver<Segment>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            chunk_rx,
            engine,
            config,
            metrics: default_sink(),
        }
    }

    pub fn new_with_metrics(
        chunk_rx: mpsc::Receiver<Segment>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
        metrics: Arc<dyn MetricsSink>,
    ) -> Self {
        Self {
            chunk_rx,
            engine,
            config,
            metrics,
        }
    }

    pub async fn run(mut self) {
        if self.config.parallel_synth && self.engine.supports_synthesis_queue() {
            self.run_parallel().await;
            return;
        }

        let last_play_done_ts: Arc<StdMutex<Option<Instant>>> = Arc::new(StdMutex::new(None));
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();
        let mut current_turn_id: Option<u64> = None;
        let mut first_audio_emitted = false;
        while let Some(segment) = self.chunk_rx.recv().await {
            if segment.first_token_ts.is_some() {
                self.metrics.on_event(MetricEvent {
                    turn_id: segment.turn_id,
                    kind: MetricEventKind::TtsFirstSegmentSent,
                    ts: segment.segment_sent_ts,
                });
            }

            if current_turn_id != Some(segment.turn_id) {
                current_turn_id = Some(segment.turn_id);
                first_audio_emitted = false;
            }

            match self.engine.speak(&segment.text).await {
                Ok(metrics) => {
                    if !first_audio_emitted && segment.first_token_ts.is_some() {
                        if let Some(ts) = metrics.first_audio_ts {
                            self.metrics.on_event(MetricEvent {
                                turn_id: segment.turn_id,
                                kind: MetricEventKind::TtsFirstAudio,
                                ts,
                            });
                            first_audio_emitted = true;
                        }
                    }
                    track_playback_drain(metrics, &last_play_done_ts, &mut play_done_tasks);
                }
                Err(e) => {
                    warn!(turn_id = segment.turn_id, error = %e, "TTS engine failed");
                }
            }
        }

        await_playback_drain(&mut play_done_tasks, &last_play_done_ts).await;
    }

    async fn run_parallel(&mut self) {
        let last_play_done_ts: Arc<StdMutex<Option<Instant>>> = Arc::new(StdMutex::new(None));
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();
        let mut current_turn_id: Option<u64> = None;
        let mut first_audio_emitted = false;

        let max_inflight = self.config.synth_inflight.max(1);
        let backlog_limit = self.config.backlog_limit.max(1);
        let synth_timeout = if self.config.synth_timeout_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(self.config.synth_timeout_ms))
        };

        let semaphore = Arc::new(Semaphore::new(max_inflight));
        let mut synth_tasks: JoinSet<SynthOutcome> = JoinSet::new();
        let mut synth_jobs: HashMap<Id, SynthJob> = HashMap::new();

        let mut seq: u64 = 0;
        let mut next_seq: u64 = 0;
        let mut input_closed = false;
        let mut pending: BTreeMap<u64, SynthOutcome> = BTreeMap::new();
        let mut backlog: VecDeque<SynthJob> = VecDeque::new();

        loop {
            while let Some(job) = backlog.pop_front() {
                if let Err(job) = try_spawn_synth(
                    job,
                    self.engine.clone(),
                    semaphore.clone(),
                    synth_timeout,
                    &mut synth_tasks,
                    &mut synth_jobs,
                ) {
                    backlog.push_front(job);
                    break;
                }
            }

            tokio::select! {
                maybe_segment = self.chunk_rx.recv(), if !input_closed && backlog.len() < backlog_limit => {
                    match maybe_segment {
                        Some(segment) => {
                            if segment.first_token_ts.is_some() {
                                self.metrics.on_event(MetricEvent {
                                    turn_id: segment.turn_id,
                                    kind: MetricEventKind::TtsFirstSegmentSent,
                                    ts: segment.segment_sent_ts,
                                });
                            }
                            if current_turn_id != Some(segment.turn_id) {
                                current_turn_id = Some(segment.turn_id);
                                first_audio_emitted = false;
                            }
                            let seq_id = seq;
                            seq = seq.wrapping_add(1);
                            let job = SynthJob {
                                seq: seq_id,
                                segment,
                            };
                            if let Err(job) = try_spawn_synth(
                                job,
                                self.engine.clone(),
                                semaphore.clone(),
                                synth_timeout,
                                &mut synth_tasks,
                                &mut synth_jobs,
                            ) {
                                backlog.push_back(job);
                            }
                        }
                        None => {
                            input_closed = true;
                        }
                    }
                }
                maybe_join = synth_tasks.join_next_with_id(), if !synth_tasks.is_empty() => {
                    if let Some(joined) = maybe_join {
                        match joined {
                            Ok((id, outcome)) => {
                                let _ = synth_jobs.remove(&id);
                                pending.insert(outcome.seq, outcome);
                            }
                            Err(err) => {
                                let id = err.id();
                                let Some(job) = synth_jobs.remove(&id) else {
                                    continue;
                                };
                                pending.insert(
                                    job.seq,
                                    SynthOutcome {
                                        seq: job.seq,
                                        segment: job.segment,
                                        result: Err(anyhow::anyhow!(
                                            "synth task failed: {err}"
                                        )),
                                    },
                                );
                            }
                        }
                    }
                }
            }

            while let Some(outcome) = pending.remove(&next_seq) {
                next_seq = next_seq.wrapping_add(1);
                match outcome.result {
                    Ok(Some(audio)) => match self.engine.play_samples(audio).await {
                        Ok(Some(metrics)) => {
                            if !first_audio_emitted && outcome.segment.first_token_ts.is_some() {
                                if let Some(ts) = metrics.first_audio_ts {
                                    self.metrics.on_event(MetricEvent {
                                        turn_id: outcome.segment.turn_id,
                                        kind: MetricEventKind::TtsFirstAudio,
                                        ts,
                                    });
                                    first_audio_emitted = true;
                                }
                            }
                            track_playback_drain(metrics, &last_play_done_ts, &mut play_done_tasks);
                        }
                        Ok(None) => {
                            warn!(
                                turn_id = outcome.segment.turn_id,
                                "TTS engine play_samples returned None"
                            );
                        }
                        Err(e) => {
                            warn!(turn_id = outcome.segment.turn_id, error = %e, "TTS engine failed");
                        }
                    },
                    Ok(None) => {
                        warn!(
                            turn_id = outcome.segment.turn_id,
                            "TTS engine synthesize returned None"
                        );
                    }
                    Err(e) => {
                        warn!(turn_id = outcome.segment.turn_id, error = %e, "TTS engine failed");
                    }
                }
            }

            if input_closed && backlog.is_empty() && synth_tasks.is_empty() && pending.is_empty() {
                break;
            }
        }

        synth_tasks.abort_all();
        synth_tasks.detach_all();
        await_playback_drain(&mut play_done_tasks, &last_play_done_ts).await;
    }
}

fn track_playback_drain(
    metrics: TtsMetrics,
    last_play_done_ts: &Arc<StdMutex<Option<Instant>>>,
    play_done_tasks: &mut JoinSet<()>,
) {
    if let Some(play_done_rx) = metrics.play_done_rx {
        let last_done = last_play_done_ts.clone();
        play_done_tasks.spawn(async move {
            if let Ok(ts) = play_done_rx.await {
                let mut guard = last_done.lock().expect("playback done lock poisoned");
                if guard.map_or(true, |prev| ts > prev) {
                    *guard = Some(ts);
                }
            }
        });
        return;
    }

    let ts = metrics.play_done_ts;
    if let Ok(mut guard) = last_play_done_ts.lock() {
        if guard.map_or(true, |prev| ts > prev) {
            *guard = Some(ts);
        }
    }
}

async fn await_playback_drain(
    play_done_tasks: &mut JoinSet<()>,
    last_play_done_ts: &Arc<StdMutex<Option<Instant>>>,
) {
    while play_done_tasks.join_next().await.is_some() {}
    let done_ts = last_play_done_ts.lock().map(|guard| *guard).unwrap_or(None);
    let Some(done_ts) = done_ts else {
        return;
    };
    let now = Instant::now();
    if done_ts > now {
        sleep_until(done_ts).await;
    }
}

fn try_spawn_synth(
    job: SynthJob,
    engine: Arc<dyn TtsEngine>,
    semaphore: Arc<Semaphore>,
    synth_timeout: Option<Duration>,
    tasks: &mut JoinSet<SynthOutcome>,
    task_index: &mut HashMap<Id, SynthJob>,
) -> std::result::Result<(), SynthJob> {
    let Ok(permit) = semaphore.clone().try_acquire_owned() else {
        return Err(job);
    };
    spawn_synth_with_permit(job, permit, engine, synth_timeout, tasks, task_index);
    Ok(())
}

fn spawn_synth_with_permit(
    job: SynthJob,
    permit: OwnedSemaphorePermit,
    engine: Arc<dyn TtsEngine>,
    synth_timeout: Option<Duration>,
    tasks: &mut JoinSet<SynthOutcome>,
    task_index: &mut HashMap<Id, SynthJob>,
) {
    let job_for_map = SynthJob {
        seq: job.seq,
        segment: job.segment.clone(),
    };
    let abort = tasks.spawn(async move {
        let _permit = permit;
        let result = match synth_timeout {
            Some(timeout) => {
                match tokio::time::timeout(timeout, engine.synthesize(&job.segment.text)).await {
                    Ok(res) => res,
                    Err(_) => Err(anyhow::anyhow!(
                        "synthesize timed out after {}ms",
                        timeout.as_millis()
                    )),
                }
            }
            None => engine.synthesize(&job.segment.text).await,
        };
        SynthOutcome {
            seq: job.seq,
            segment: job.segment,
            result,
        }
    });
    task_index.insert(abort.id(), job_for_map);
}
