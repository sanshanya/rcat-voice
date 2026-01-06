use crate::generator::{SynthesizedAudio, TtsEngine};
use crate::internal::env;
use crate::internal::metrics;
use crate::tokenizer::Segment;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio::task::{Id, JoinSet};
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
        cfg.synth_inflight =
            env::usize_clamped("TTS_SYNTH_INFLIGHT", cfg.synth_inflight, 1, 8);
        cfg.backlog_limit = env::usize_clamped("TTS_BACKLOG_LIMIT", cfg.backlog_limit, 1, 4096);
        cfg.synth_timeout_ms = env::u64_clamped(
            "TTS_SYNTH_TIMEOUT_MS",
            cfg.synth_timeout_ms,
            0,
            10 * 60_000,
        );
        cfg
    }
}

pub struct Pipeline {
    chunk_rx: mpsc::Receiver<Segment>,
    cancel_rx: watch::Receiver<bool>,
    interrupt_rx: watch::Receiver<u64>,
    engine: Arc<dyn TtsEngine>,
    config: PipelineConfig,
}

struct SynthOutcome {
    seq: u64,
    epoch: u64,
    segment: Segment,
    result: std::result::Result<Option<SynthesizedAudio>, anyhow::Error>,
}

struct SynthJob {
    seq: u64,
    epoch: u64,
    segment: Segment,
}

impl Pipeline {
    pub fn new(
        chunk_rx: mpsc::Receiver<Segment>,
        cancel_rx: watch::Receiver<bool>,
        interrupt_rx: watch::Receiver<u64>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            chunk_rx,
            cancel_rx,
            interrupt_rx,
            engine,
            config,
        }
    }

    pub async fn run(mut self) {
        if self.config.parallel_synth && self.engine.supports_synthesis_queue() {
            self.run_parallel().await;
            return;
        }

        let mut intro_logged = false;
        let last_play_done_ts: Arc<StdMutex<Option<tokio::time::Instant>>> =
            Arc::new(StdMutex::new(None));
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();
        let mut cancel_closed = false;
        let mut interrupt_closed = false;

        loop {
            if *self.cancel_rx.borrow() {
                break;
            }
            let maybe_segment = tokio::select! {
                res = self.interrupt_rx.changed(), if !interrupt_closed => {
                    if res.is_err() {
                        interrupt_closed = true;
                        None
                    } else {
                        let _ = self.engine.stop().await;
                        while self.chunk_rx.try_recv().is_ok() {}
                        play_done_tasks.abort_all();
                        while play_done_tasks.try_join_next().is_some() {}
                        intro_logged = false;
                        if let Ok(mut guard) = last_play_done_ts.lock() {
                            *guard = None;
                        }
                    None
                    }
                }
                res = self.cancel_rx.changed(), if !cancel_closed => {
                    if res.is_err() {
                        cancel_closed = true;
                    }
                    None
                },
                maybe = self.chunk_rx.recv() => maybe,
            };
            let Some(segment) = maybe_segment else {
                if self.chunk_rx.is_closed() {
                    break;
                }
                continue;
            };

            metrics::log_segment_intro(&segment, &mut intro_logged);

            match self.engine.speak(&segment.text).await {
                Ok(metrics) => {
                    metrics::log_playback_metrics(
                        &segment,
                        metrics,
                        &last_play_done_ts,
                        &mut play_done_tasks,
                    );
                }
                Err(e) => {
                    warn!("TTS engine failed: {}", e);
                }
            }
        }

        metrics::await_playback_drain(&mut play_done_tasks, &last_play_done_ts).await;
    }

    async fn run_parallel(&mut self) {
        let mut intro_logged = false;
        let last_play_done_ts: Arc<StdMutex<Option<tokio::time::Instant>>> =
            Arc::new(StdMutex::new(None));
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();
        let mut cancel_closed = false;
        let mut interrupt_closed = false;

        let max_inflight = self.config.synth_inflight.max(1);
        let backlog_limit = self.config.backlog_limit.max(1);
        let synth_timeout = if self.config.synth_timeout_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(self.config.synth_timeout_ms))
        };

        let mut semaphore = Arc::new(Semaphore::new(max_inflight));
        let mut synth_tasks: JoinSet<SynthOutcome> = JoinSet::new();
        let mut synth_jobs: HashMap<Id, SynthJob> = HashMap::new();

        let mut seq: u64 = 0;
        let mut next_seq: u64 = 0;
        let mut input_closed = false;
        let mut epoch: u64 = 0;
        let mut pending: BTreeMap<u64, SynthOutcome> = BTreeMap::new();
        let mut backlog: VecDeque<SynthJob> = VecDeque::new();

        loop {
            if *self.cancel_rx.borrow() {
                break;
            }

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
                res = self.interrupt_rx.changed(), if !interrupt_closed => {
                    if res.is_err() {
                        interrupt_closed = true;
                        continue;
                    }
                    let _ = self.engine.stop().await;
                    while self.chunk_rx.try_recv().is_ok() {}
                    play_done_tasks.abort_all();
                    while play_done_tasks.try_join_next().is_some() {}
                    pending.clear();
                    backlog.clear();
                    synth_tasks.abort_all();
                    synth_tasks.detach_all();
                    synth_jobs.clear();
                    semaphore = Arc::new(Semaphore::new(max_inflight));
                    epoch = epoch.wrapping_add(1);
                    seq = 0;
                    next_seq = 0;
                    intro_logged = false;
                    if let Ok(mut guard) = last_play_done_ts.lock() {
                        *guard = None;
                    }
                }
                res = self.cancel_rx.changed(), if !cancel_closed => {
                    if res.is_err() {
                        cancel_closed = true;
                    }
                }
                maybe_segment = self.chunk_rx.recv(), if !input_closed && backlog.len() < backlog_limit => {
                    match maybe_segment {
                        Some(segment) => {
                            metrics::log_segment_intro(&segment, &mut intro_logged);
                            let seq_id = seq;
                            seq = seq.wrapping_add(1);
                            let job = SynthJob {
                                seq: seq_id,
                                epoch,
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
                                if outcome.epoch == epoch {
                                    pending.insert(outcome.seq, outcome);
                                }
                            }
                            Err(err) => {
                                let id = err.id();
                                let Some(job) = synth_jobs.remove(&id) else {
                                    continue;
                                };
                                if job.epoch == epoch {
                                    pending.insert(job.seq, SynthOutcome {
                                        seq: job.seq,
                                        epoch: job.epoch,
                                        segment: job.segment,
                                        result: Err(anyhow::anyhow!("synth task failed: {err}")),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            while let Some(outcome) = pending.remove(&next_seq) {
                if outcome.epoch != epoch {
                    continue;
                }
                next_seq = next_seq.wrapping_add(1);
                match outcome.result {
                    Ok(Some(audio)) => match self.engine.play_samples(audio).await {
                        Ok(Some(metrics)) => {
                            metrics::log_playback_metrics(
                                &outcome.segment,
                                metrics,
                                &last_play_done_ts,
                                &mut play_done_tasks,
                            );
                        }
                        Ok(None) => {
                            warn!("TTS engine play_samples returned None");
                        }
                        Err(e) => {
                            warn!("TTS engine failed: {}", e);
                        }
                    },
                    Ok(None) => {
                        warn!("TTS engine synthesize returned None");
                    }
                    Err(e) => {
                        warn!("TTS engine failed: {}", e);
                    }
                }
            }

            if input_closed && backlog.is_empty() && synth_tasks.is_empty() && pending.is_empty() {
                break;
            }
        }

        synth_tasks.abort_all();
        synth_tasks.detach_all();
        metrics::await_playback_drain(&mut play_done_tasks, &last_play_done_ts).await;
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
        epoch: job.epoch,
        segment: job.segment.clone(),
    };
    let abort = tasks.spawn(async move {
        let _permit = permit;
        let result = match synth_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, engine.synthesize(&job.segment.text)).await {
                Ok(res) => res,
                Err(_) => Err(anyhow::anyhow!("synthesize timed out after {}ms", timeout.as_millis())),
            },
            None => engine.synthesize(&job.segment.text).await,
        };
        SynthOutcome {
            seq: job.seq,
            epoch: job.epoch,
            segment: job.segment,
            result,
        }
    });
    task_index.insert(abort.id(), job_for_map);
}
