use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, watch};

use tokio::time::Instant;

use crate::internal::env;

const THRESHOLD_MAX_CHARS: usize = 400;

#[derive(Debug, Clone, Copy)]
struct FlushThresholds {
    min_chars: usize,
    soft_max: usize,
    hard_max: usize,
}

const EAGER_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 2,
    soft_max: 6,
    hard_max: 12,
};

const NORMAL_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 10,
    soft_max: 20,
    hard_max: 40,
};

const RELAX_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 20,
    soft_max: 35,
    hard_max: 80,
};

impl FlushThresholds {
    fn from_env(prefix: &str, defaults: FlushThresholds) -> Self {
        let (min_chars, soft_max, hard_max) = env::usize_threshold_triplet(
            prefix,
            defaults.min_chars,
            defaults.soft_max,
            defaults.hard_max,
            THRESHOLD_MAX_CHARS,
        );
        Self {
            min_chars,
            soft_max,
            hard_max,
        }
    }
}

#[derive(Debug, Clone)]
/// 发送到 TTS 管线的文本片段（含时间戳）。
pub struct Segment {
    /// 片段文本内容。
    pub text: String,
    /// LLM 请求起点时间戳（t0）。若未显式标记，则为会话创建时间。
    pub llm_start_ts: Instant,
    /// 首个 LLM token 时间戳（t1），仅首段设置。
    pub first_token_ts: Option<Instant>,
    /// 发送该片段前观测到的最后一个 LLM token 时间戳。
    pub last_token_ts: Option<Instant>,
    /// 片段发送到管线的时间戳（t2）。
    pub segment_sent_ts: Instant,
}

/// Tokenizer tuning parameters.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub eager_chunks: usize,
    pub relax_buffer_ms: u64,
    pub relax_log: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            eager_chunks: 1,
            relax_buffer_ms: 200,
            relax_log: false,
        }
    }
}

impl TokenizerConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(parsed) = env::get::<usize>("TOKENIZER_EAGER_CHUNKS")
            .or_else(|| env::get::<usize>("CHUNKER_EAGER_CHUNKS"))
        {
            cfg.eager_chunks = parsed;
        }
        if let Some(parsed) = env::get::<u64>("TOKENIZER_RELAX_BUFFER_MS") {
            cfg.relax_buffer_ms = parsed.clamp(0, 60_000);
        }
        cfg.relax_log = env::bool01("TOKENIZER_RELAX_LOG", cfg.relax_log);
        cfg.normalize()
    }

    pub fn normalize(mut self) -> Self {
        self.eager_chunks = self.eager_chunks.clamp(0, 32);
        self.relax_buffer_ms = self.relax_buffer_ms.clamp(0, 60_000);
        self
    }
}

pub struct Tokenizer {
    delta_rx: mpsc::Receiver<String>,
    chunk_tx: mpsc::Sender<Segment>,
    cancel_rx: watch::Receiver<bool>,
    interrupt_rx: watch::Receiver<u64>,
    buffer_ms_rx: watch::Receiver<u64>,
    session_start_ts: Instant, // t0 fallback
    llm_start: Arc<OnceLock<Instant>>,
    config: TokenizerConfig,
}

impl Tokenizer {
    pub fn new(
        delta_rx: mpsc::Receiver<String>,
        chunk_tx: mpsc::Sender<Segment>,
        cancel_rx: watch::Receiver<bool>,
        interrupt_rx: watch::Receiver<u64>,
        buffer_ms_rx: watch::Receiver<u64>,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        config: TokenizerConfig,
    ) -> Self {
        Self {
            delta_rx,
            chunk_tx,
            cancel_rx,
            interrupt_rx,
            buffer_ms_rx,
            session_start_ts,
            llm_start,
            config: config.normalize(),
        }
    }

    pub fn from_env(
        delta_rx: mpsc::Receiver<String>,
        chunk_tx: mpsc::Sender<Segment>,
        cancel_rx: watch::Receiver<bool>,
        interrupt_rx: watch::Receiver<u64>,
        buffer_ms_rx: watch::Receiver<u64>,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
    ) -> Self {
        Self::new(
            delta_rx,
            chunk_tx,
            cancel_rx,
            interrupt_rx,
            buffer_ms_rx,
            session_start_ts,
            llm_start,
            TokenizerConfig::from_env(),
        )
    }

    async fn emit_segment(
        &self,
        text: String,
        first: &mut bool,
        first_delta_ts: Option<Instant>,
        last_delta_ts: Option<Instant>,
    ) -> bool {
        let mut text = text;
        if text.trim().is_empty() {
            return true;
        }
        if text.starts_with('\u{feff}') {
            text = text.trim_start_matches('\u{feff}').to_string();
        }

        let llm_start_ts = self
            .llm_start
            .get()
            .copied()
            .unwrap_or(self.session_start_ts);
        let chunk = Segment {
            text,
            llm_start_ts,
            first_token_ts: if *first { first_delta_ts } else { None },
            last_token_ts: last_delta_ts,
            segment_sent_ts: Instant::now(),
        };

        if self.chunk_tx.send(chunk).await.is_err() {
            return false;
        }
        *first = false;
        true
    }

    pub async fn run(mut self) {
        let eager_thresholds = FlushThresholds::from_env("TOKENIZER_EAGER", EAGER_DEFAULT);
        let normal_thresholds = FlushThresholds::from_env("TOKENIZER_NORMAL", NORMAL_DEFAULT);
        let relax_thresholds = FlushThresholds::from_env("TOKENIZER_RELAX", RELAX_DEFAULT);
        let mut buf = String::new();
        let mut first = true;
        let mut first_delta_ts: Option<Instant> = None;
        let mut last_delta_ts: Option<Instant> = None;
        let mut eager_chunks_remaining = self.config.eager_chunks;
        let eager_chunks_default = eager_chunks_remaining;
        let mut cancel_closed = false;
        let mut interrupt_closed = false;

        let relax_buffer_ms = self.config.relax_buffer_ms;
        let relax_log = self.config.relax_log;
        let mut relax_active = false;
        loop {
            tokio::select! {
                res = self.cancel_rx.changed(), if !cancel_closed => {
                    if res.is_err() {
                        cancel_closed = true;
                        continue;
                    }
                    if *self.cancel_rx.borrow() { break; }
                }
                res = self.interrupt_rx.changed(), if !interrupt_closed => {
                    if res.is_err() {
                        interrupt_closed = true;
                        continue;
                    }
                    buf.clear();
                    while self.delta_rx.try_recv().is_ok() {}
                    first = true;
                    first_delta_ts = None;
                    last_delta_ts = None;
                    eager_chunks_remaining = eager_chunks_default;
                }
                maybe = self.delta_rx.recv() => {
                    let Some(delta) = maybe else {
                        let pending = std::mem::take(&mut buf);
                        let _ = self
                            .emit_segment(pending, &mut first, first_delta_ts, last_delta_ts)
                            .await;
                        break;
                    };

                    // Capture time of first token
                    if first && first_delta_ts.is_none() {
                        first_delta_ts = Some(Instant::now());
                    }
                    last_delta_ts = Some(Instant::now());

                    buf.push_str(&delta);

                    let mut buffered_ms_for_log: Option<u64> = None;
                    let (thresholds, relax_now) = if eager_chunks_remaining > 0 {
                        (eager_thresholds, false)
                    } else {
                        let buffered_ms = *self.buffer_ms_rx.borrow();
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
                            .unwrap_or_else(|| *self.buffer_ms_rx.borrow());
                        log_relax_transition(
                            relax_now,
                            &mut relax_active,
                            buffered_ms,
                            (min_c, soft_max, hard_max),
                        );
                    }
                    if let Some(cut_idx) = find_flush_index(
                        &buf,
                        min_c,
                        soft_max,
                        hard_max,
                    ) {
                        let remaining = buf.split_off(cut_idx);
                        let pending = std::mem::replace(&mut buf, remaining);
                        if !self
                            .emit_segment(pending, &mut first, first_delta_ts, last_delta_ts)
                            .await
                        {
                            break;
                        }
                        if eager_chunks_remaining > 0 {
                            eager_chunks_remaining -= 1;
                        }
                    }
                }
            }
        }
    }
}

fn log_relax_transition(
    relax_now: bool,
    relax_active: &mut bool,
    buffered_ms: u64,
    thresholds: (usize, usize, usize),
) {
    if relax_now == *relax_active {
        return;
    }
    *relax_active = relax_now;
    if relax_now {
        let (min_c, soft_max, hard_max) = thresholds;
        tracing::info!(
            "Tokenizer relax on (buffer={}ms, min={}, soft_max={}, hard_max={})",
            buffered_ms,
            min_c,
            soft_max,
            hard_max
        );
    } else {
        tracing::info!("Tokenizer relax off (buffer={}ms)", buffered_ms);
    }
}

fn find_flush_index(
    s: &str,
    min_chars: usize,
    soft_max: usize,
    hard_max: usize,
) -> Option<usize> {
    let scan = scan_boundaries(s, hard_max);
    let count = scan.total_chars;

    if count < min_chars {
        return None;
    }

    if count < soft_max {
        return scan.ends_with_boundary.then_some(s.len());
    }

    let window_chars = hard_max.saturating_sub(soft_max).max(min_chars);

    if count < hard_max {
        let window_start = count.saturating_sub(window_chars);
        return scan
            .last_boundary
            .filter(|hit| hit.char_count >= min_chars && hit.char_count >= window_start)
            .map(|hit| hit.byte_idx);
    }

    let window_start = hard_max.saturating_sub(window_chars);
    scan.last_boundary_before_hard_max
        .filter(|hit| hit.char_count >= min_chars && hit.char_count >= window_start)
        .map(|hit| hit.byte_idx)
        .or(scan.hard_cut)
}

#[derive(Debug, Clone, Copy)]
struct BoundaryHit {
    byte_idx: usize,
    char_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct BoundaryScan {
    total_chars: usize,
    ends_with_boundary: bool,
    last_boundary: Option<BoundaryHit>,
    last_boundary_before_hard_max: Option<BoundaryHit>,
    hard_cut: Option<usize>,
}

fn scan_boundaries(s: &str, hard_max: usize) -> BoundaryScan {
    let mut total_chars = 0usize;
    let mut last_boundary: Option<BoundaryHit> = None;
    let mut last_boundary_before_hard_max: Option<BoundaryHit> = None;
    let mut hard_cut: Option<usize> = None;

    for (idx, ch) in s.char_indices() {
        total_chars += 1;
        let next = idx + ch.len_utf8();
        if matches!(
            ch,
            '。' | '！' | '？' | '!' | '?' | '\n' | '，' | ',' | '；' | ';' | '：' | ':'
        ) {
            last_boundary = Some(BoundaryHit {
                byte_idx: next,
                char_count: total_chars,
            });
        }
        if hard_max > 0 && total_chars == hard_max {
            hard_cut = Some(next);
            last_boundary_before_hard_max = last_boundary;
        }
    }

    let ends_with_boundary = last_boundary.map_or(false, |hit| hit.byte_idx == s.len());
    BoundaryScan {
        total_chars,
        ends_with_boundary,
        last_boundary,
        last_boundary_before_hard_max,
        hard_cut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_requires_min_chars() {
        let s = "Hi!";
        let idx = find_flush_index(s, 5, 20, 30);
        assert_eq!(idx, None);
    }

    #[test]
    fn flush_on_strong_boundary_at_end() {
        let s = "Hello!";
        let idx = find_flush_index(s, 1, 20, 30);
        assert_eq!(idx, Some(s.len()));
    }

    #[test]
    fn flush_on_weak_boundary_when_over_max() {
        let s = "abcd,efgh";
        let idx = find_flush_index(s, 1, 5, 12);
        assert_eq!(idx, Some(5));
    }

    #[test]
    fn hard_cut_when_no_boundary_and_over_hard_limit() {
        let s = "abcdefghij";
        let idx = find_flush_index(s, 1, 5, 8);
        assert_eq!(idx, Some(8));
    }

    #[test]
    fn waits_for_hard_max_if_tail_window_has_no_boundary() {
        let s = "ab,cdef";
        let idx = find_flush_index(s, 1, 5, 8);
        assert_eq!(idx, None);

        let s2 = "ab,cdefg";
        let idx2 = find_flush_index(s2, 1, 5, 8);
        assert_eq!(idx2, Some(8));
    }

    #[test]
    fn ignores_boundary_before_min_chars() {
        let s = "ab,xxxxxxxxxx";
        let idx = find_flush_index(s, 5, 10, 20);
        assert_eq!(idx, None);

        let s2 = "ab,xxxxxxxxxxxxxxxxx";
        let idx2 = find_flush_index(s2, 5, 10, 20);
        assert_eq!(idx2, Some(20));
    }
}
