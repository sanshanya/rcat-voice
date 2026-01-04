use crate::generator::TtsEngine;
use anyhow::Result;
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::tokenizer::{Segment, Tokenizer, TokenizerConfig};
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};

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
        if let Ok(value) = std::env::var("AUDIO_BUFFER_POLL_MS") {
            if let Ok(parsed) = value.parse::<u64>() {
                cfg.buffer_poll_ms = parsed.clamp(5, 500);
            }
        }
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
    cancel_tx: watch::Sender<bool>,
    pause_tx: watch::Sender<bool>,
    tts_engine: Arc<dyn TtsEngine>,
    llm_start: Arc<OnceLock<Instant>>,
}

impl StreamControl {
    /// 可克隆的 sender，用于写入 LLM 流式增量。
    pub fn sender(&self) -> mpsc::Sender<String> {
        self.delta_tx.clone()
    }

    /// 标记 LLM 请求开始时间，用于指标统计。
    pub fn mark_llm_start(&self) {
        let _ = self.llm_start.get_or_init(Instant::now);
    }

    /// 中断播放并清空已排队音频（不可恢复）。
    pub async fn pause(&self) -> Result<()> {
        let _ = self.pause_tx.send(true);
        self.tts_engine.stop().await?;
        Ok(())
    }

    /// 取消当前流并停止播放。
    pub async fn cancel(&self) -> Result<()> {
        let _ = self.cancel_tx.send(true);
        self.tts_engine.stop().await?;
        Ok(())
    }
}

/// Stream cancellation handle that does not keep the delta input channel open.
#[derive(Clone)]
pub struct StreamCancelHandle {
    cancel_tx: watch::Sender<bool>,
    pause_tx: watch::Sender<bool>,
    tts_engine: Arc<dyn TtsEngine>,
    llm_start: Arc<OnceLock<Instant>>,
}

impl StreamCancelHandle {
    /// 标记 LLM 请求开始时间，用于指标统计。
    pub fn mark_llm_start(&self) {
        let _ = self.llm_start.get_or_init(Instant::now);
    }

    /// 中断播放并清空已排队音频（不可恢复）。
    pub async fn pause(&self) -> Result<()> {
        let _ = self.pause_tx.send(true);
        self.tts_engine.stop().await?;
        Ok(())
    }

    /// 取消当前流并停止播放。
    pub async fn cancel(&self) -> Result<()> {
        let _ = self.cancel_tx.send(true);
        self.tts_engine.stop().await?;
        Ok(())
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
        let task_start = Instant::now();
        let llm_start = Arc::new(OnceLock::new());
        let (delta_tx, delta_rx) = mpsc::channel::<String>(stream_config.delta_channel);
        let (chunk_tx, chunk_rx) = mpsc::channel::<Segment>(stream_config.segment_channel);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let (pause_tx, pause_rx) = watch::channel(false);
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
            pause_rx.clone(),
            tts_engine.clone(),
            pipeline_config,
        );
        let pipeline_handle = tokio::spawn(pipeline.run());

        let tokenizer = Tokenizer::new(
            delta_rx,
            chunk_tx,
            cancel_rx.clone(),
            pause_rx.clone(),
            buffer_rx,
            task_start,
            llm_start.clone(),
            tokenizer_config,
        );
        let tokenizer_handle = tokio::spawn(tokenizer.run());

        let control = StreamControl {
            delta_tx,
            cancel_tx,
            pause_tx,
            tts_engine,
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
            cancel_tx: self.control.cancel_tx.clone(),
            pause_tx: self.control.pause_tx.clone(),
            tts_engine: self.control.tts_engine.clone(),
            llm_start: self.control.llm_start.clone(),
        }
    }

    /// 终止会话并等待后台任务结束。
    pub async fn shutdown(self) -> Result<()> {
        let _ = self.control.cancel_tx.send(true);
        self.control.tts_engine.stop().await?;
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
                        let _ = self.control.cancel_tx.send(true);
                        let _ = self.control.tts_engine.stop().await;
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
            let _ = self.control.cancel_tx.send(true);
            let _ = self.control.tts_engine.stop().await;
        }

        Ok(())
    }
}
