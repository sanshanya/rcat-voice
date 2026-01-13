use crate::generator::{SynthesizedAudio, TtsEngine, TtsMetrics};
use crate::internal::env;
use crate::metrics::{MetricEvent, MetricEventKind, MetricsSink, default_sink};
use crate::pipeline::{PipelineConfig, PipelineMode, PipelineState};
use crate::tokenizer::{
    EAGER_DEFAULT, NORMAL_DEFAULT, RELAX_DEFAULT, FlushThresholds, TextSegment, TokenizerConfig,
    find_flush_index, log_relax_transition,
};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::{AbortHandle, Id, JoinHandle, JoinSet};
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tracing::warn;

#[derive(Debug, Clone)]
pub enum StreamMsg {
    Delta { text: String, epoch: u64 },
    Eof { epoch: u64 },
}

/// Internal handles for O(1) cancellation.
///
/// Phase 3 fix: abort receiver tasks to make upstream send() fail immediately.
#[derive(Clone)]
struct SessionCancel {
    tts_engine: Arc<dyn TtsEngine>,
    cancel: CancellationToken,
    /// Abort handle for orchestrator task (holds stream_rx)
    orchestrator_abort: AbortHandle,
}

impl SessionCancel {
    /// Interrupt current turn: O(1) cancel path.
    ///
    /// Order:
    /// 1. stop_fast() -> epoch++ (invalidates all CancelScopes immediately)
    /// 2. abort tasks (drops receivers, makes send() fail)
    /// 3. optional signal (for compatibility)
    async fn interrupt(&self) -> Result<()> {
        // Step 1: Stop TTS and increment epoch (MUST be first!)
        // This is the authority - makes all CancelScope.is_cancelled() == true
        // stop_fast() is O(1), only sets flag and clears ring buffer
        self.tts_engine.stop_fast();

        // Step 2: Broadcast soft cancel to cooperative tasks
        self.cancel.cancel();

        // Step 3: Abort orchestrator (drops receivers)
        // O(1) - just sets abort flag
        self.orchestrator_abort.abort();

        Ok(())
    }

    /// Cancel session: O(1) cancel path.
    async fn cancel(&self) -> Result<()> {
        // Step 1: Stop TTS and increment epoch (authority for CancelScope)
        self.tts_engine.stop_fast();

        // Step 2: Broadcast soft cancel to cooperative tasks
        self.cancel.cancel();

        // Step 3: Abort orchestrator
        self.orchestrator_abort.abort();

        Ok(())
    }

    /// Best-effort cancel without Result propagation.
    fn cancel_best_effort(&self) {
        // O(1) fast path
        self.tts_engine.stop_fast();
        self.cancel.cancel();
        self.orchestrator_abort.abort();
    }
}

/// Stream session configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub delta_channel: usize,
    pub segment_channel: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            // Target: fit a typical max LLM output window (e.g. 8192 tokens) without
            // backpressuring the upstream streaming sender.
            delta_channel: 8192,
            // Text segments are usually far fewer than deltas, but keep a large default so
            // Tokenizer/Orchestrator does not stall on slow TTS by default.
            segment_channel: 4096,
        }
    }
}

impl StreamConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.delta_channel = env::usize_clamped(
            "STREAM_DELTA_CAPACITY",
            cfg.delta_channel,
            1,
            65_536,
        );
        cfg.segment_channel = env::usize_clamped(
            "STREAM_SEGMENT_CAPACITY",
            cfg.segment_channel,
            1,
            65_536,
        );
        cfg
    }
}

/// Builder for `StreamSession` with explicit configs.
pub struct StreamSessionBuilder {
    tts_engine: Arc<dyn TtsEngine>,
    stream: StreamConfig,
    tokenizer: TokenizerConfig,
    pipeline: PipelineConfig,
    turn_id: u64,
    metrics: Arc<dyn MetricsSink>,
}

impl StreamSessionBuilder {
    pub fn new(tts_engine: Arc<dyn TtsEngine>) -> Self {
        Self {
            tts_engine,
            stream: StreamConfig::default(),
            tokenizer: TokenizerConfig::default(),
            pipeline: PipelineConfig::default(),
            turn_id: 0,
            metrics: default_sink(),
        }
    }

    pub fn from_env(tts_engine: Arc<dyn TtsEngine>) -> Self {
        Self {
            tts_engine,
            stream: StreamConfig::from_env(),
            tokenizer: TokenizerConfig::from_env(),
            pipeline: PipelineConfig::from_env(),
            turn_id: 0,
            metrics: default_sink(),
        }
    }

    pub fn stream_config(mut self, config: StreamConfig) -> Self {
        self.stream = config;
        self
    }

    pub fn tokenizer_config(mut self, config: TokenizerConfig) -> Self {
        self.tokenizer = config;
        self
    }

    pub fn pipeline_config(mut self, config: PipelineConfig) -> Self {
        self.pipeline = config;
        self
    }

    /// Set a metrics sink for this session.
    pub fn metrics_sink(mut self, metrics: Arc<dyn MetricsSink>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Bind a `turn_id` to all segments produced by this session.
    ///
    /// Convention: `0` means "unknown/unbound".
    pub fn turn_id(mut self, turn_id: u64) -> Self {
        self.turn_id = turn_id;
        self
    }

    pub fn build(self) -> StreamSession {
        StreamSession::new_with_configs(
            self.tts_engine,
            self.stream,
            self.tokenizer,
            self.pipeline,
            self.turn_id,
            self.metrics,
        )
    }
}

/// 流式输入与生命周期控制句柄。
#[derive(Clone)]
pub struct StreamHandle {
    stream_tx: mpsc::Sender<StreamMsg>,
    input_finished: Arc<AtomicBool>,
    cancel: SessionCancel,
    llm_start: Arc<OnceLock<Instant>>,
    turn_id: u64,
    metrics: Arc<dyn MetricsSink>,
    epoch: Arc<AtomicU64>,
}

impl StreamHandle {
    /// 返回可克隆的输入 sender，用于写入 StreamMsg。
    pub fn sender(&self) -> mpsc::Sender<StreamMsg> {
        self.stream_tx.clone()
    }

    /// 发送一段 LLM 增量文本。
    pub async fn push_delta(&self, delta: String) -> Result<()> {
        if self.input_finished.load(Ordering::Acquire) {
            return Err(anyhow!("input already finished"));
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        self.stream_tx
            .send(StreamMsg::Delta { text: delta, epoch })
            .await
            .map_err(|_| anyhow!("stream channel closed"))?;
        Ok(())
    }

    /// 标记输入结束（发送 EOF）。
    pub async fn finish_input(&self) -> Result<()> {
        if self.input_finished.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if self.stream_tx.is_closed() {
            return Ok(());
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        self.stream_tx
            .send(StreamMsg::Eof { epoch })
            .await
            .map_err(|_| anyhow!("stream channel closed"))?;
        Ok(())
    }

    /// 标记 LLM 请求开始时间，用于指标统计。
    pub fn mark_llm_start(&self) {
        let ts = Instant::now();
        if self.llm_start.set(ts).is_ok() {
            self.metrics.on_event(MetricEvent {
                turn_id: self.turn_id,
                kind: MetricEventKind::LlmStart,
                ts,
            });
        }
    }

    /// 打断当前轮次：停止播放并清空已排队音频（不可恢复）。
    ///
    /// 注意：该操作不会自动停止上游 LLM 流；如需“打断后立刻重新生成”的体验，
    /// 调用方仍应同时取消旧的 LLM 流，并开始新的 StreamSession / 新一轮发送。
    pub async fn interrupt(&self) -> Result<()> {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.interrupt().await
    }

    /// 自动选择 finish 或 interrupt 的停止接口。
    /// - 若已 finish_input，则不再中断播放（由 session drain）。
    /// - 否则立即打断。
    pub async fn stop(&self) -> Result<()> {
        if self.input_finished.load(Ordering::Acquire) {
            return Ok(());
        }
        self.interrupt().await
    }

    /// 取消当前流并停止播放。
    pub async fn cancel(&self) -> Result<()> {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.cancel.cancel().await
    }
}

/// 将 LLM 流式增量接入分段器与播放管线的会话。
pub struct StreamSession {
    handle: StreamHandle,
    orchestrator_handle: JoinHandle<()>,
}

impl StreamSession {
    /// 使用默认配置创建会话；如需读取环境变量请使用 `from_env`。
    pub fn new(tts_engine: Arc<dyn TtsEngine>) -> Self {
        StreamSessionBuilder::new(tts_engine).build()
    }

    /// 使用环境变量配置创建会话。
    pub fn from_env(tts_engine: Arc<dyn TtsEngine>) -> Self {
        StreamSessionBuilder::from_env(tts_engine).build()
    }

    /// 返回可配置的构建器。
    pub fn builder(tts_engine: Arc<dyn TtsEngine>) -> StreamSessionBuilder {
        StreamSessionBuilder::new(tts_engine)
    }

    fn new_with_configs(
        tts_engine: Arc<dyn TtsEngine>,
        stream_config: StreamConfig,
        tokenizer_config: TokenizerConfig,
        pipeline_config: PipelineConfig,
        turn_id: u64,
        metrics: Arc<dyn MetricsSink>,
    ) -> Self {
        let session_start_ts = Instant::now();
        let llm_start = Arc::new(OnceLock::new());
        let (stream_tx, stream_rx) = mpsc::channel::<StreamMsg>(stream_config.delta_channel);
        let cancel = CancellationToken::new();
        let epoch = Arc::new(AtomicU64::new(0));
        let mut pipeline_config = pipeline_config;
        let seg_limit = stream_config.segment_channel.max(1);
        pipeline_config.backlog_limit = pipeline_config.backlog_limit.min(seg_limit);

        let orchestrator = Orchestrator::new(
            stream_rx,
            tts_engine.clone(),
            tokenizer_config,
            pipeline_config,
            metrics.clone(),
            session_start_ts,
            llm_start.clone(),
            turn_id,
            cancel.clone(),
            epoch.clone(),
        );
        let orchestrator_handle = tokio::spawn(orchestrator.run());
        let orchestrator_abort = orchestrator_handle.abort_handle();

        let cancel = SessionCancel {
            tts_engine,
            cancel,
            orchestrator_abort,
        };

        let handle = StreamHandle {
            stream_tx,
            input_finished: Arc::new(AtomicBool::new(false)),
            cancel,
            llm_start,
            turn_id,
            metrics,
            epoch,
        };

        Self {
            handle,
            orchestrator_handle,
        }
    }

    pub fn control(&self) -> StreamHandle {
        self.handle.clone()
    }

    /// 终止会话并等待后台任务结束。
    pub async fn shutdown(self) -> Result<()> {
        self.handle.cancel.cancel().await?;
        drop(self.handle.stream_tx);
        let _ = self.orchestrator_handle.await;
        Ok(())
    }

    /// Gracefully finish the current stream: close the input channel and wait for
    /// orchestrator tasks to drain and for audio playback to complete.
    pub async fn finish(self) -> Result<()> {
        let _ = self.handle.finish_input().await;
        let _ = self.orchestrator_handle.await;
        Ok(())
    }

    /// Finish the stream like [`finish`](Self::finish), but abort quickly if `cancel` is triggered.
    ///
    /// Intended for "full duplex + barge-in": once the LLM stream ends, we still want to wait for
    /// TTS playback to drain, but we must be able to stop immediately when the user starts speaking.
    pub async fn finish_or_cancel(mut self) -> Result<()> {
        let _ = self.handle.finish_input().await;
        let cancel = self.handle.cancel.cancel.clone();
        let mut cancelled = cancel.is_cancelled();
        let mut orchestrator_done = false;

        while !orchestrator_done {
            tokio::select! {
                _ = cancel.cancelled(), if !cancelled => {
                    cancelled = true;
                    self.handle.cancel.cancel_best_effort();
                }
                _ = &mut self.orchestrator_handle => {
                    orchestrator_done = true;
                }
            }
        }

        if cancelled {
            self.handle.cancel.cancel_best_effort();
        }

        Ok(())
    }
}

struct Orchestrator {
    stream_rx: mpsc::Receiver<StreamMsg>,
    engine: Arc<dyn TtsEngine>,
    tokenizer_config: TokenizerConfig,
    pipeline_config: PipelineConfig,
    metrics: Arc<dyn MetricsSink>,
    session_start_ts: Instant,
    llm_start: Arc<OnceLock<Instant>>,
    turn_id: u64,
    cancel: CancellationToken,
    epoch: Arc<AtomicU64>,
}

impl Orchestrator {
    fn new(
        stream_rx: mpsc::Receiver<StreamMsg>,
        engine: Arc<dyn TtsEngine>,
        tokenizer_config: TokenizerConfig,
        pipeline_config: PipelineConfig,
        metrics: Arc<dyn MetricsSink>,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        turn_id: u64,
        cancel: CancellationToken,
        epoch: Arc<AtomicU64>,
    ) -> Self {
        Self {
            stream_rx,
            engine,
            tokenizer_config,
            pipeline_config,
            metrics,
            session_start_ts,
            llm_start,
            turn_id,
            cancel,
            epoch,
        }
    }

    async fn run(self) {
        let mut stream_rx = self.stream_rx;
        let engine = self.engine;
        let metrics = self.metrics;
        let session_start_ts = self.session_start_ts;
        let llm_start = self.llm_start;
        let turn_id = self.turn_id;
        let cancel = self.cancel;
        let epoch = self.epoch;
        let tokenizer_config = self.tokenizer_config;
        let pipeline_config = self.pipeline_config;

        let runner = match pipeline_config.mode {
            PipelineMode::Serial => RunnerKind::Serial,
            PipelineMode::Decoupled => {
                if engine.supports_synthesis_queue() {
                    RunnerKind::Decoupled
                } else {
                    warn!(
                        "PipelineMode::Decoupled requested but engine does not support synthesis queue; falling back to Serial"
                    );
                    RunnerKind::Serial
                }
            }
            PipelineMode::Auto => {
                if engine.supports_synthesis_queue() {
                    RunnerKind::Decoupled
                } else {
                    RunnerKind::Serial
                }
            }
        };

        let max_inflight = match runner {
            RunnerKind::Serial => 1,
            RunnerKind::Decoupled => pipeline_config.synth_inflight.max(1),
        };
        let backlog_limit = pipeline_config.backlog_limit.max(1);
        let synth_timeout = match runner {
            RunnerKind::Decoupled => {
                if pipeline_config.synth_timeout_ms == 0 {
                    None
                } else {
                    Some(Duration::from_millis(pipeline_config.synth_timeout_ms))
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
        let mut cancelled = false;

        let eager_thresholds = FlushThresholds::from_env("TOKENIZER_EAGER", EAGER_DEFAULT);
        let normal_thresholds = FlushThresholds::from_env("TOKENIZER_NORMAL", NORMAL_DEFAULT);
        let relax_thresholds = FlushThresholds::from_env("TOKENIZER_RELAX", RELAX_DEFAULT);
        let relax_buffer_ms = tokenizer_config.relax_buffer_ms;
        let relax_log = tokenizer_config.relax_log;
        let mut buf = String::new();
        let mut first = true;
        let mut first_delta_ts: Option<Instant> = None;
        let mut last_delta_ts: Option<Instant> = None;
        let mut llm_first_token_emitted = false;
        let mut eager_chunks_remaining = tokenizer_config.eager_chunks;
        let mut relax_active = false;

        loop {
            while let Some(job) = backlog.pop_front() {
                if let Err(job) = try_spawn_job(
                    job,
                    runner,
                    engine.clone(),
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
                maybe_msg = stream_rx.recv(), if !input_closed && backlog.len() < backlog_limit => {
                    let mut is_eof = false;
                    match maybe_msg {
                        Some(StreamMsg::Delta { text, epoch: msg_epoch }) => {
                            let current_epoch = epoch.load(Ordering::Acquire);
                            if msg_epoch != current_epoch || input_closed {
                                continue;
                            }

                            let now = Instant::now();
                            if first_delta_ts.is_none() && !text.is_empty() {
                                first_delta_ts = Some(now);
                                if !llm_first_token_emitted {
                                    metrics.on_event(MetricEvent {
                                        turn_id,
                                        kind: MetricEventKind::LlmFirstToken,
                                        ts: now,
                                    });
                                    llm_first_token_emitted = true;
                                }
                            }
                            if !text.is_empty() {
                                last_delta_ts = Some(now);
                            }
                            buf.push_str(&text);
                        }
                        Some(StreamMsg::Eof { epoch: msg_epoch }) => {
                            let current_epoch = epoch.load(Ordering::Acquire);
                            if msg_epoch != current_epoch {
                                continue;
                            }
                            input_closed = true;
                            is_eof = true;
                        }
                        None => {
                            input_closed = true;
                            is_eof = true;
                        }
                    }

                    if is_eof && buf.is_empty() {
                        continue;
                    }

                    loop {
                        let mut buffered_ms_for_log: Option<u64> = None;
                        let (thresholds, relax_now) = if eager_chunks_remaining > 0 {
                            (eager_thresholds, false)
                        } else {
                            let buffered_ms = engine.buffered_ms().unwrap_or(0);
                            buffered_ms_for_log = Some(buffered_ms);
                            let relax_now = relax_buffer_ms > 0 && buffered_ms >= relax_buffer_ms;
                            let thresholds = if relax_now {
                                relax_thresholds
                            } else {
                                normal_thresholds
                            };
                            (thresholds, relax_now)
                        };

                        let (min_c, soft_max, hard_max) = (
                            thresholds.min_chars,
                            thresholds.soft_max,
                            thresholds.hard_max,
                        );
                        if relax_log && relax_now != relax_active {
                            let buffered_ms = buffered_ms_for_log
                                .unwrap_or_else(|| engine.buffered_ms().unwrap_or(0));
                            log_relax_transition(
                                relax_now,
                                &mut relax_active,
                                buffered_ms,
                                (min_c, soft_max, hard_max),
                            );
                        }

                        let Some(cut_idx) = find_flush_index(&buf, min_c, soft_max, hard_max) else {
                            if is_eof {
                                let pending = std::mem::take(&mut buf);
                                if let Some(segment) = build_text_segment(
                                    pending,
                                    &mut first,
                                    first_delta_ts,
                                    last_delta_ts,
                                    &llm_start,
                                    session_start_ts,
                                    turn_id,
                                ) {
                                    schedule_text_segment(
                                        segment,
                                        &mut state,
                                        &mut seq,
                                        runner,
                                        engine.clone(),
                                        semaphore.clone(),
                                        synth_timeout,
                                        &mut tasks,
                                        &mut task_index,
                                        &mut backlog,
                                        &metrics,
                                    );
                                }
                            }
                            break;
                        };

                        let remaining = buf.split_off(cut_idx);
                        let pending = std::mem::replace(&mut buf, remaining);
                        if let Some(segment) = build_text_segment(
                            pending,
                            &mut first,
                            first_delta_ts,
                            last_delta_ts,
                            &llm_start,
                            session_start_ts,
                            turn_id,
                        ) {
                            schedule_text_segment(
                                segment,
                                &mut state,
                                &mut seq,
                                runner,
                                engine.clone(),
                                semaphore.clone(),
                                synth_timeout,
                                &mut tasks,
                                &mut task_index,
                                &mut backlog,
                                &metrics,
                            );
                        }
                        if eager_chunks_remaining > 0 {
                            eager_chunks_remaining -= 1;
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
                    Ok(Some(OutcomeKind::Played(tts_metrics))) => {
                        state.on_metrics(&outcome.text_segment, &tts_metrics, metrics.as_ref());
                        track_playback_drain(
                            tts_metrics,
                            &last_play_done_ts,
                            &mut play_done_tasks,
                        );
                    }
                    Ok(Some(OutcomeKind::NeedPlay(audio))) => match engine.play_samples(audio).await {
                        Ok(Some(tts_metrics)) => {
                            state.on_metrics(&outcome.text_segment, &tts_metrics, metrics.as_ref());
                            track_playback_drain(
                                tts_metrics,
                                &last_play_done_ts,
                                &mut play_done_tasks,
                            );
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

fn build_text_segment(
    text: String,
    first: &mut bool,
    first_delta_ts: Option<Instant>,
    last_delta_ts: Option<Instant>,
    llm_start: &Arc<OnceLock<Instant>>,
    session_start_ts: Instant,
    turn_id: u64,
) -> Option<TextSegment> {
    let mut text = text;
    if text.trim().is_empty() {
        return None;
    }
    if text.starts_with('\u{feff}') {
        text = text.trim_start_matches('\u{feff}').to_string();
    }

    let llm_start_ts = llm_start.get().copied().unwrap_or(session_start_ts);
    let segment = TextSegment {
        turn_id,
        text,
        llm_start_ts,
        first_token_ts: if *first { first_delta_ts } else { None },
        last_token_ts: last_delta_ts,
        segment_sent_ts: Instant::now(),
    };
    *first = false;
    Some(segment)
}

fn schedule_text_segment(
    segment: TextSegment,
    state: &mut PipelineState,
    seq: &mut u64,
    runner: RunnerKind,
    engine: Arc<dyn TtsEngine>,
    semaphore: Arc<Semaphore>,
    synth_timeout: Option<Duration>,
    tasks: &mut JoinSet<JobOutcome>,
    task_index: &mut HashMap<Id, Job>,
    backlog: &mut VecDeque<Job>,
    metrics: &Arc<dyn MetricsSink>,
) {
    state.on_segment(&segment, metrics.as_ref());
    let seq_id = *seq;
    *seq = seq.wrapping_add(1);
    let job = Job {
        seq: seq_id,
        text_segment: segment,
    };
    if let Err(job) = try_spawn_job(
        job,
        runner,
        engine,
        semaphore,
        synth_timeout,
        tasks,
        task_index,
    ) {
        backlog.push_back(job);
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
