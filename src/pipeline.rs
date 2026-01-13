use anyhow::anyhow;
use crate::generator::{SynthesizedAudio, TtsEngine, TtsMetrics};
use crate::internal::env;
use crate::metrics::{MetricEvent, MetricEventKind, MetricsSink, default_sink};
use crate::tokenizer::TextSegment;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::{Id, JoinSet};
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    Auto,
    Serial,
    Decoupled,
}

impl PipelineMode {
    pub fn from_env() -> Self {
        if let Some(raw) = env::string("TTS_PIPELINE_MODE") {
            if let Some(mode) = Self::parse(&raw) {
                return mode;
            }
        }
        let parallel = env::bool01("TTS_PARALLEL_SYNTH", true);
        if parallel {
            PipelineMode::Auto
        } else {
            PipelineMode::Serial
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "auto" => Some(PipelineMode::Auto),
            "serial" | "sync" => Some(PipelineMode::Serial),
            "decoupled" | "parallel" | "synth" => Some(PipelineMode::Decoupled),
            _ => None,
        }
    }
}

/// Pipeline scheduling knobs.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub mode: PipelineMode,
    pub synth_inflight: usize,
    pub backlog_limit: usize,
    pub synth_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            mode: PipelineMode::Auto,
            synth_inflight: 1,
            // Keep a large default so streaming Tokenizer/Orchestrator can absorb fast LLM output
            // without immediately backpressuring the upstream sender.
            backlog_limit: 4096,
            synth_timeout_ms: 30_000,
        }
    }
}

impl PipelineConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        cfg.mode = PipelineMode::from_env();
        cfg.synth_inflight = env::usize_clamped("TTS_SYNTH_INFLIGHT", cfg.synth_inflight, 1, 8);
        cfg.backlog_limit = env::usize_clamped("TTS_BACKLOG_LIMIT", cfg.backlog_limit, 1, 4096);
        cfg.synth_timeout_ms =
            env::u64_clamped("TTS_SYNTH_TIMEOUT_MS", cfg.synth_timeout_ms, 0, 10 * 60_000);
        cfg
    }
}

pub struct Pipeline {
    text_seg_rx: mpsc::Receiver<TextSegment>,
    engine: Arc<dyn TtsEngine>,
    config: PipelineConfig,
    metrics: Arc<dyn MetricsSink>,
    cancel: CancellationToken,
}

pub(crate) struct PipelineState {
    current_turn_id: Option<u64>,
    first_audio_emitted: bool,
}

impl PipelineState {
    pub(crate) fn new() -> Self {
        Self {
            current_turn_id: None,
            first_audio_emitted: false,
        }
    }

    pub(crate) fn on_segment(&mut self, text_segment: &TextSegment, metrics: &dyn MetricsSink) {
        if text_segment.first_token_ts.is_some() {
            metrics.on_event(MetricEvent {
                turn_id: text_segment.turn_id,
                kind: MetricEventKind::TtsFirstSegmentSent,
                ts: text_segment.segment_sent_ts,
            });
        }
        if self.current_turn_id != Some(text_segment.turn_id) {
            self.current_turn_id = Some(text_segment.turn_id);
            self.first_audio_emitted = false;
        }
    }

    pub(crate) fn on_metrics(
        &mut self,
        text_segment: &TextSegment,
        m: &TtsMetrics,
        metrics: &dyn MetricsSink,
    ) {
        if !self.first_audio_emitted && text_segment.first_token_ts.is_some() {
            if let Some(ts) = m.first_audio_ts {
                metrics.on_event(MetricEvent {
                    turn_id: text_segment.turn_id,
                    kind: MetricEventKind::TtsFirstAudio,
                    ts,
                });
                self.first_audio_emitted = true;
            }
        }
    }
}

enum OutcomeKind {
    NeedPlay(SynthesizedAudio),
    Played(TtsMetrics),
}

struct JobOutcome {
    seq: u64,
    text_segment: TextSegment,
    result: std::result::Result<Option<OutcomeKind>, anyhow::Error>,
}

struct Job {
    seq: u64,
    text_segment: TextSegment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunnerKind {
    Serial,
    Decoupled,
}

impl Pipeline {
    pub fn new(
        text_seg_rx: mpsc::Receiver<TextSegment>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            text_seg_rx,
            engine,
            config,
            metrics: default_sink(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn new_with_metrics(
        text_seg_rx: mpsc::Receiver<TextSegment>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
        metrics: Arc<dyn MetricsSink>,
    ) -> Self {
        Self {
            text_seg_rx,
            engine,
            config,
            metrics,
            cancel: CancellationToken::new(),
        }
    }

    pub fn with_cancel_token(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub async fn run(mut self) {
        let runner = match self.config.mode {
            PipelineMode::Serial => RunnerKind::Serial,
            PipelineMode::Decoupled => {
                if self.engine.supports_synthesis_queue() {
                    RunnerKind::Decoupled
                } else {
                    warn!("PipelineMode::Decoupled requested but engine does not support synthesis queue; falling back to Serial");
                    RunnerKind::Serial
                }
            }
            PipelineMode::Auto => {
                if self.engine.supports_synthesis_queue() {
                    RunnerKind::Decoupled
                } else {
                    RunnerKind::Serial
                }
            }
        };

        let max_inflight = match runner {
            RunnerKind::Serial => 1,
            RunnerKind::Decoupled => self.config.synth_inflight.max(1),
        };
        let backlog_limit = self.config.backlog_limit.max(1);
        let synth_timeout = match runner {
            RunnerKind::Decoupled => {
                if self.config.synth_timeout_ms == 0 {
                    None
                } else {
                    Some(Duration::from_millis(self.config.synth_timeout_ms))
                }
            }
            RunnerKind::Serial => None,
        };

        let last_play_done_ts: Arc<StdMutex<Option<Instant>>> = Arc::new(StdMutex::new(None));
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();
        let mut state = PipelineState::new();

        let semaphore = Arc::new(Semaphore::new(max_inflight));
        let mut tasks: JoinSet<JobOutcome> = JoinSet::new();
        let mut task_index: HashMap<Id, Job> = HashMap::new();

        let mut seq: u64 = 0;
        let mut next_seq: u64 = 0;
        let mut input_closed = false;
        let mut pending: BTreeMap<u64, JobOutcome> = BTreeMap::new();
        let mut backlog: VecDeque<Job> = VecDeque::new();
        let cancel = self.cancel.clone();
        let mut cancelled = false;

        loop {
            while let Some(job) = backlog.pop_front() {
                if let Err(job) = try_spawn_job(
                    job,
                    runner,
                    self.engine.clone(),
                    semaphore.clone(),
                    synth_timeout,
                    &mut tasks,
                    &mut task_index,
                ) {
                    backlog.push_front(job);
                    break;
                }
            }

            tokio::select! {
                _ = cancel.cancelled(), if !cancelled => {
                    cancelled = true;
                    break;
                }
                maybe_segment = self.text_seg_rx.recv(), if !input_closed && backlog.len() < backlog_limit => {
                    match maybe_segment {
                        Some(text_segment) => {
                            state.on_segment(&text_segment, self.metrics.as_ref());
                            let seq_id = seq;
                            seq = seq.wrapping_add(1);
                            let job = Job {
                                seq: seq_id,
                                text_segment,
                            };
                            if let Err(job) = try_spawn_job(
                                job,
                                runner,
                                self.engine.clone(),
                                semaphore.clone(),
                                synth_timeout,
                                &mut tasks,
                                &mut task_index,
                            ) {
                                backlog.push_back(job);
                            }
                        }
                        None => {
                            input_closed = true;
                        }
                    }
                }
                maybe_join = tasks.join_next_with_id(), if !tasks.is_empty() => {
                    if let Some(joined) = maybe_join {
                        match joined {
                            Ok((id, outcome)) => {
                                let _ = task_index.remove(&id);
                                pending.insert(outcome.seq, outcome);
                            }
                            Err(err) => {
                                let id = err.id();
                                let Some(job) = task_index.remove(&id) else {
                                    continue;
                                };
                                pending.insert(
                                    job.seq,
                                    JobOutcome {
                                        seq: job.seq,
                                        text_segment: job.text_segment,
                                        result: Err(anyhow!("job task failed: {err}")),
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
                    Ok(Some(OutcomeKind::Played(metrics))) => {
                        state.on_metrics(&outcome.text_segment, &metrics, self.metrics.as_ref());
                        track_playback_drain(metrics, &last_play_done_ts, &mut play_done_tasks);
                    }
                    Ok(Some(OutcomeKind::NeedPlay(audio))) => match self.engine.play_samples(audio).await {
                        Ok(Some(metrics)) => {
                            state.on_metrics(&outcome.text_segment, &metrics, self.metrics.as_ref());
                            track_playback_drain(metrics, &last_play_done_ts, &mut play_done_tasks);
                        }
                        Ok(None) => {
                            warn!(
                                turn_id = outcome.text_segment.turn_id,
                                "TTS engine play_samples returned None"
                            );
                        }
                        Err(e) => {
                            warn!(turn_id = outcome.text_segment.turn_id, error = %e, "TTS engine failed");
                        }
                    },
                    Ok(None) => {
                        warn!(
                            turn_id = outcome.text_segment.turn_id,
                            "TTS engine synthesize returned None"
                        );
                    }
                    Err(e) => {
                        warn!(turn_id = outcome.text_segment.turn_id, error = %e, "TTS engine failed");
                    }
                }
            }

            if input_closed && backlog.is_empty() && tasks.is_empty() && pending.is_empty() {
                break;
            }
        }

        tasks.abort_all();
        tasks.detach_all();
        if !cancelled {
            await_playback_drain(&mut play_done_tasks, &last_play_done_ts).await;
        }
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

fn try_spawn_job(
    job: Job,
    runner: RunnerKind,
    engine: Arc<dyn TtsEngine>,
    semaphore: Arc<Semaphore>,
    synth_timeout: Option<Duration>,
    tasks: &mut JoinSet<JobOutcome>,
    task_index: &mut HashMap<Id, Job>,
) -> std::result::Result<(), Job> {
    let Ok(permit) = semaphore.clone().try_acquire_owned() else {
        return Err(job);
    };
    spawn_job_with_permit(
        job,
        permit,
        runner,
        engine,
        synth_timeout,
        tasks,
        task_index,
    );
    Ok(())
}

fn spawn_job_with_permit(
    job: Job,
    permit: OwnedSemaphorePermit,
    runner: RunnerKind,
    engine: Arc<dyn TtsEngine>,
    synth_timeout: Option<Duration>,
    tasks: &mut JoinSet<JobOutcome>,
    task_index: &mut HashMap<Id, Job>,
) {
    let job_for_map = Job {
        seq: job.seq,
        text_segment: job.text_segment.clone(),
    };
    let abort = tasks.spawn(async move {
        let _permit = permit;
        let result = match runner {
            RunnerKind::Serial => engine
                .speak(&job.text_segment.text)
                .await
                .map(|m| Some(OutcomeKind::Played(m))),
            RunnerKind::Decoupled => {
                let synth = match synth_timeout {
                    Some(timeout) => {
                        match tokio::time::timeout(timeout, engine.synthesize(&job.text_segment.text)).await {
                            Ok(res) => res,
                            Err(_) => Err(anyhow!(
                                "synthesize timed out after {}ms",
                                timeout.as_millis()
                            )),
                        }
                    }
                    None => engine.synthesize(&job.text_segment.text).await,
                };
                match synth {
                    Ok(Some(audio)) => Ok(Some(OutcomeKind::NeedPlay(audio))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            }
        };
        JobOutcome {
            seq: job.seq,
            text_segment: job.text_segment,
            result,
        }
    });
    task_index.insert(abort.id(), job_for_map);
}
