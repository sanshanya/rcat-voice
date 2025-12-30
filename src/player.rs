use crate::chunker::Chunk;
use crate::tts::TtsEngine;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use tracing::{info, warn};

pub struct Player {
    chunk_rx: mpsc::Receiver<Chunk>,
    cancel_rx: watch::Receiver<bool>,
    queued_ms: Arc<AtomicU64>,
    engine: Arc<dyn TtsEngine>,
}

impl Player {
    pub fn new(
        chunk_rx: mpsc::Receiver<Chunk>,
        cancel_rx: watch::Receiver<bool>,
        queued_ms: Arc<AtomicU64>,
        engine: Arc<dyn TtsEngine>,
    ) -> Self {
        Self {
            chunk_rx,
            cancel_rx,
            queued_ms,
            engine,
        }
    }

    pub async fn run(mut self) {
        let mut first = true;

        while let Some(chunk) = self.chunk_rx.recv().await {
            if *self.cancel_rx.borrow() {
                break;
            }

            if first {
                info!("=== 指标时间线 ===");

                if let Some(t1) = chunk.first_token_ts {
                    // LLM首字时延: 从请求到第一个字
                    let llm_delay = t1.duration_since(chunk.task_start);
                    let llm_abs = llm_delay;
                    info!("LLM首字时延: {:?} @ {:?}", llm_delay, llm_abs);

                    // 分段器延迟: 从第一个字到第一段被分出来
                    let chunker_delay = chunk.chunk_sent_ts.duration_since(t1);
                    let chunker_abs = chunk.chunk_sent_ts.duration_since(chunk.task_start);
                    info!("分段器延迟: {:?} @ {:?}", chunker_delay, chunker_abs);
                }

                // 首播时延: 从第一段分出来到播放开始
                let play_delay = chunk.task_start.elapsed()
                    - chunk.chunk_sent_ts.duration_since(chunk.task_start);
                let play_abs = chunk.task_start.elapsed();
                info!("首播时延: {:?} @ {:?}", play_delay, play_abs);

                info!("首段长度: {} 字符", chunk.text.len());

                first = false;
            }

            // Measure TTS processing delay
            let tts_start = Instant::now();

            match self.engine.speak(&chunk.text).await {
                Ok(_) => {
                    let tts_time = tts_start.elapsed();
                    let audio_complete_abs = chunk.task_start.elapsed();

                    // OS TTS 是同步的：合成+播放一体，这里是播放完成的时间
                    if chunk.first_token_ts.is_some() {
                        let chunk_sent_abs = chunk.chunk_sent_ts.duration_since(chunk.task_start);
                        let audio_gen_delay = audio_complete_abs - chunk_sent_abs;
                        info!(
                            "首句播放完成: {:?} @ {:?} | {} 字符",
                            audio_gen_delay,
                            audio_complete_abs,
                            chunk.text.len()
                        );
                    } else {
                        info!(
                            "音频块播放完成: {:?} @ {:?} | {} 字符",
                            tts_time,
                            audio_complete_abs,
                            chunk.text.len()
                        );
                    }

                    // Successfully spoken, remove estimate from queue
                    self.queued_ms.fetch_sub(chunk.est_ms, Ordering::Relaxed);
                }
                Err(e) => {
                    warn!("TTS engine failed: {}", e);
                }
            }
        }
    }
}
