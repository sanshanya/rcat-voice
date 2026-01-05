use crate::generator::TtsEngine;
use crate::internal::env;
use anyhow::Result;
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::tokenizer::{Segment, Tokenizer, TokenizerConfig};
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tracing::debug;

#[derive(Clone)]
struct SessionCancel {
    cancel_tx: watch::Sender<bool>,
    interrupt_tx: watch::Sender<u64>,
    tts_engine: Arc<dyn TtsEngine>,
}

impl SessionCancel {
    async fn interrupt(&self) -> Result<()> {
        let next = self.interrupt_tx.borrow().wrapping_add(1);
        if let Err(e) = self.interrupt_tx.send(next) {
            debug!("stream: interrupt signal send failed: {e:?}");
        }
        self.tts_engine.stop().await?;
        Ok(())
    }

    #[allow(dead_code)]
    #[deprecated(note = "Use interrupt() (this is a barge-in style interrupt, not pause/resume).")]
    async fn pause(&self) -> Result<()> {
        self.interrupt().await
    }

    async fn cancel(&self) -> Result<()> {
        if let Err(e) = self.cancel_tx.send(true) {
            debug!("stream: cancel signal send failed: {e:?}");
        }
        self.tts_engine.stop().await?;
        Ok(())
    }

    async fn cancel_best_effort(&self) {
        if let Err(e) = self.cancel_tx.send(true) {
            debug!("stream: cancel signal send failed: {e:?}");
        }
        if let Err(e) = self.tts_engine.stop().await {
            debug!("stream: stop failed during cancel: {e}");
        }
    }
}

/// Stream session configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub buffer_poll_ms: u64,
    pub delta_channel: usize,
    pub segment_channel: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_poll_ms: 20,
            delta_channel: 8192,
            segment_channel: 4096,
        }
    }
}

impl StreamConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.buffer_poll_ms =
            env::u64_clamped("AUDIO_BUFFER_POLL_MS", cfg.buffer_poll_ms, 5, 500);
        cfg
    }
}

/// Builder for `StreamSession` with explicit configs.
pub struct StreamSessionBuilder {
    tts_engine: Arc<dyn TtsEngine>,
    stream: StreamConfig,
    tokenizer: TokenizerConfig,
    pipeline: PipelineConfig,
}

impl StreamSessionBuilder {
    pub fn new(tts_engine: Arc<dyn TtsEngine>) -> Self {
        Self {
            tts_engine,
            stream: StreamConfig::default(),
            tokenizer: TokenizerConfig::default(),
            pipeline: PipelineConfig::default(),
        }
    }

    pub fn from_env(tts_engine: Arc<dyn TtsEngine>) -> Self {
        Self {
            tts_engine,
            stream: StreamConfig::from_env(),
            tokenizer: TokenizerConfig::from_env(),
            pipeline: PipelineConfig::from_env(),
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

    pub fn build(self) -> StreamSession {
        StreamSession::new_with_configs(
            self.tts_engine,
            self.stream,
            self.tokenizer,
            self.pipeline,
        )
    }
}

/// 流式输入与生命周期控制句柄。
#[derive(Clone)]
pub struct StreamControl {
    delta_tx: mpsc::Sender<String>,
    cancel: SessionCancel,
    llm_start: Arc<OnceLock<Instant>>,
}

impl StreamControl {
    /// 返回可克隆的 delta sender，用于写入 LLM 流式增量。
    ///
    /// ⚠️ 注意：持有此 sender 的任何 clone 都会阻止 channel 关闭，导致
    /// [`StreamSession::finish`] / [`StreamSession::finish_or_cancel`] 无法正常结束。
    ///
    /// 如需可控结束：
    /// - 在调用 `finish()` 前 drop 所有 sender clone；或
    /// - 改用 [`StreamSession::shutdown`]（立即取消并停止播放）；并通过
    ///   [`StreamSession::cancel_handle`] 获取不持有 sender 的取消句柄。
    pub fn sender(&self) -> mpsc::Sender<String> {
        self.delta_tx.clone()
    }

    /// 标记 LLM 请求开始时间，用于指标统计。
    pub fn mark_llm_start(&self) {
        let _ = self.llm_start.get_or_init(Instant::now);
    }

    /// 打断当前轮次：停止播放并清空已排队音频（不可恢复）。
    ///
    /// 注意：该操作不会自动停止上游 LLM 流；如需“打断后立刻重新生成”的体验，
    /// 调用方仍应同时取消旧的 LLM 流，并开始新的 StreamSession / 新一轮发送。
    pub async fn interrupt(&self) -> Result<()> {
        self.cancel.interrupt().await
    }

    #[deprecated(note = "Use interrupt() (pause never resumes; this is a barge-in style interrupt).")]
    pub async fn pause(&self) -> Result<()> {
        self.interrupt().await
    }

    /// 取消当前流并停止播放。
    pub async fn cancel(&self) -> Result<()> {
        self.cancel.cancel().await
    }
}

/// Stream cancellation handle that does not keep the delta input channel open.
#[derive(Clone)]
pub struct StreamCancelHandle {
    cancel: SessionCancel,
    llm_start: Arc<OnceLock<Instant>>,
}

impl StreamCancelHandle {
    /// 标记 LLM 请求开始时间，用于指标统计。
    pub fn mark_llm_start(&self) {
        let _ = self.llm_start.get_or_init(Instant::now);
    }

    /// 打断当前轮次：停止播放并清空已排队音频（不可恢复）。
    ///
    /// 注意：该操作不会自动停止上游 LLM 流；如需“打断后立刻重新生成”的体验，
    /// 调用方仍应同时取消旧的 LLM 流，并开始新的 StreamSession / 新一轮发送。
    pub async fn interrupt(&self) -> Result<()> {
        self.cancel.interrupt().await
    }

    #[deprecated(note = "Use interrupt() (pause never resumes; this is a barge-in style interrupt).")]
    pub async fn pause(&self) -> Result<()> {
        self.interrupt().await
    }

    /// 取消当前流并停止播放。
    pub async fn cancel(&self) -> Result<()> {
        self.cancel.cancel().await
    }
}

/// 将 LLM 流式增量接入分段器与播放管线的会话。
pub struct StreamSession {
    control: StreamControl,
    pipeline_handle: JoinHandle<()>,
    tokenizer_handle: JoinHandle<()>,
    buffer_handle: JoinHandle<()>,
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
    ) -> Self {
        let session_start_ts = Instant::now();
        let llm_start = Arc::new(OnceLock::new());
        let (delta_tx, delta_rx) = mpsc::channel::<String>(stream_config.delta_channel);
        let (chunk_tx, chunk_rx) = mpsc::channel::<Segment>(stream_config.segment_channel);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let (interrupt_tx, interrupt_rx) = watch::channel(0u64);
        let (buffer_tx, buffer_rx) = watch::channel(0u64);

        let buffer_engine = tts_engine.clone();
        let buffer_handle = tokio::spawn(async move {
            let mut last_ms = None;
            let mut interval =
                tokio::time::interval(Duration::from_millis(stream_config.buffer_poll_ms));
            loop {
                interval.tick().await;
                if buffer_tx.is_closed() {
                    break;
                }
                let ms = buffer_engine.buffered_ms().unwrap_or(0);
                if last_ms != Some(ms) {
                    let _ = buffer_tx.send(ms);
                    last_ms = Some(ms);
                }
            }
        });

        let pipeline = Pipeline::new(
            chunk_rx,
            cancel_rx.clone(),
            interrupt_rx.clone(),
            tts_engine.clone(),
            pipeline_config,
        );
        let pipeline_handle = tokio::spawn(pipeline.run());

        let tokenizer = Tokenizer::new(
            delta_rx,
            chunk_tx,
            cancel_rx.clone(),
            interrupt_rx.clone(),
            buffer_rx,
            session_start_ts,
            llm_start.clone(),
            tokenizer_config,
        );
        let tokenizer_handle = tokio::spawn(tokenizer.run());

        let cancel = SessionCancel {
            cancel_tx,
            interrupt_tx,
            tts_engine,
        };

        let control = StreamControl {
            delta_tx,
            cancel,
            llm_start,
        };

        Self {
            control,
            pipeline_handle,
            tokenizer_handle,
            buffer_handle,
        }
    }

    pub fn control(&self) -> StreamControl {
        self.control.clone()
    }

    pub fn cancel_handle(&self) -> StreamCancelHandle {
        StreamCancelHandle {
            cancel: self.control.cancel.clone(),
            llm_start: self.control.llm_start.clone(),
        }
    }

    /// 终止会话并等待后台任务结束。
    pub async fn shutdown(self) -> Result<()> {
        self.control.cancel.cancel().await?;
        drop(self.control.delta_tx);
        let _ = self.tokenizer_handle.await;
        let _ = self.pipeline_handle.await;
        let _ = self.buffer_handle.await;
        Ok(())
    }

    /// Gracefully finish the current stream: close the delta input channel and wait for
    /// tokenizer/pipeline tasks to drain and for audio playback to complete.
    pub async fn finish(self) -> Result<()> {
        drop(self.control.delta_tx);
        let _ = self.tokenizer_handle.await;
        let _ = self.pipeline_handle.await;
        let _ = self.buffer_handle.await;
        Ok(())
    }

    /// Finish the stream like [`finish`](Self::finish), but abort quickly if `cancel` is triggered.
    ///
    /// Intended for "full duplex + barge-in": once the LLM stream ends, we still want to wait for
    /// TTS playback to drain, but we must be able to stop immediately when the user starts speaking.
    pub async fn finish_or_cancel(mut self, mut cancel: watch::Receiver<bool>) -> Result<()> {
        drop(self.control.delta_tx);

        let mut tokenizer_done = false;
        let mut pipeline_done = false;
        let mut buffer_done = false;
        let mut cancelled = *cancel.borrow();

        while !(tokenizer_done && pipeline_done && buffer_done) {
            tokio::select! {
                res = cancel.changed(), if !cancelled => {
                    if res.is_ok() && *cancel.borrow() {
                        cancelled = true;
                        self.control.cancel.cancel_best_effort().await;
                    }
                }
                _ = &mut self.tokenizer_handle, if !tokenizer_done => {
                    tokenizer_done = true;
                }
                _ = &mut self.pipeline_handle, if !pipeline_done => {
                    pipeline_done = true;
                }
                _ = &mut self.buffer_handle, if !buffer_done => {
                    buffer_done = true;
                }
            }
        }

        if cancelled {
            self.control.cancel.cancel_best_effort().await;
        }

        Ok(())
    }
}
