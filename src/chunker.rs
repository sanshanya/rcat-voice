use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{mpsc, watch};

use tokio::time::Instant;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub est_ms: u64,
    pub task_start: Instant,             // t0: Task origin
    pub first_token_ts: Option<Instant>, // t1: First DeepSeek token
    pub chunk_sent_ts: Instant,          // t2: When chunk was sent
}

pub struct Chunker {
    delta_rx: mpsc::Receiver<String>,
    chunk_tx: mpsc::Sender<Chunk>,
    cancel_rx: watch::Receiver<bool>,
    queued_ms: Arc<AtomicU64>,
    task_start: Instant, // t0 reference
}

impl Chunker {
    pub fn new(
        delta_rx: mpsc::Receiver<String>,
        chunk_tx: mpsc::Sender<Chunk>,
        cancel_rx: watch::Receiver<bool>,
        queued_ms: Arc<AtomicU64>,
        task_start: Instant,
    ) -> Self {
        Self {
            delta_rx,
            chunk_tx,
            cancel_rx,
            queued_ms,
            task_start,
        }
    }

    pub async fn run(mut self) {
        let mut buf = String::new();
        let mut first = true;
        let mut first_delta_ts: Option<Instant> = None;

        // Constants for adaptive logic
        let low_water_ms: u64 = 600;
        let high_water_ms: u64 = 2500;
        // Estimate: 180ms per character (tuned for mixed/Chinese)
        let ms_per_char: u64 = 180;

        loop {
            tokio::select! {
                _ = self.cancel_rx.changed() => {
                    if *self.cancel_rx.borrow() { break; }
                }
                maybe = self.delta_rx.recv() => {
                    let Some(delta) = maybe else { break };

                    // Capture time of first token
                    if first && first_delta_ts.is_none() {
                        first_delta_ts = Some(Instant::now());
                    }

                    buf.push_str(&delta);

                    let q = self.queued_ms.load(Ordering::Relaxed);

                    // Adaptive thresholds
                    let (min_c, max_c) = if first {
                        (10usize, 20usize) // First chunk short for low TTFA
                    } else if q < low_water_ms {
                        (20, 45)           // Low buffer: flush often
                    } else if q > high_water_ms {
                        (80, 140)          // High buffer: batched flush
                    } else {
                        (40, 90)           // Normal operation
                    };

                    if should_flush(&buf, min_c, max_c) {
                        let text = buf.trim().to_string();
                        buf.clear();
                        if !text.is_empty() {
                            let est = (text.chars().count() as u64)
                                .saturating_mul(ms_per_char)
                                .clamp(250, 8000);

                            // Update shared state
                            self.queued_ms.fetch_add(est, Ordering::Relaxed);

                            if self.chunk_tx.send(Chunk{
                                text,
                                est_ms: est,
                                task_start: self.task_start,
                                first_token_ts: if first { first_delta_ts } else { None },
                                chunk_sent_ts: Instant::now(), // t2
                            }).await.is_err() {
                                break;
                            }
                            first = false;
                        }
                    }
                }
            }
        }
    }
}

fn should_flush(s: &str, min_chars: usize, max_chars: usize) -> bool {
    let n = s.chars().count();
    if n < min_chars {
        return false;
    }

    // Strong boundaries
    if s.ends_with(['。', '！', '？', '!', '?', '\n']) {
        return true;
    }
    // Weak boundaries
    if s.ends_with(['，', ',', '；', ';', '：', ':']) {
        return true;
    }

    n >= max_chars
}
