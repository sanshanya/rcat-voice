use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use tokio::time::Instant;

use crate::internal::env;
use crate::metrics::{MetricEvent, MetricEventKind, MetricsSink, default_sink};

const THRESHOLD_MAX_CHARS: usize = 400;

pub type BufferMsFn = Arc<dyn Fn() -> u64 + Send + Sync>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct FlushThresholds {
    pub(crate) min_chars: usize,
    pub(crate) soft_max: usize,
    pub(crate) hard_max: usize,
}

pub(crate) const EAGER_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 2,
    soft_max: 6,
    hard_max: 12,
};

pub(crate) const NORMAL_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 10,
    soft_max: 20,
    hard_max: 40,
};

pub(crate) const RELAX_DEFAULT: FlushThresholds = FlushThresholds {
    min_chars: 20,
    soft_max: 35,
    hard_max: 80,
};

impl FlushThresholds {
    pub(crate) fn from_env(prefix: &str, defaults: FlushThresholds) -> Self {
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
pub struct TextSegment {
    /// Turn ID（用于将日志/metrics 与单轮对话绑定）。
    ///
    /// 约定：0 表示“未知/未绑定”。
    pub turn_id: u64,
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

#[deprecated(since = "0.2.0", note = "use TextSegment")]
pub type Segment = TextSegment;

#[derive(Debug)]
pub enum DeltaMsg {
    Delta(String),
    Eof,
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
    delta_rx: mpsc::Receiver<DeltaMsg>,
    text_seg_tx: mpsc::Sender<TextSegment>,
    buffer_ms: BufferMsFn,
    session_start_ts: Instant, // t0 fallback
    llm_start: Arc<OnceLock<Instant>>,
    turn_id: u64,
    metrics: Arc<dyn MetricsSink>,
    config: TokenizerConfig,
    cancel: CancellationToken,
}

impl Tokenizer {
    pub fn new(
        delta_rx: mpsc::Receiver<DeltaMsg>,
        text_seg_tx: mpsc::Sender<TextSegment>,
        buffer_ms: BufferMsFn,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        turn_id: u64,
        cancel: CancellationToken,
        config: TokenizerConfig,
    ) -> Self {
        Self {
            delta_rx,
            text_seg_tx,
            buffer_ms,
            session_start_ts,
            llm_start,
            turn_id,
            metrics: default_sink(),
            config: config.normalize(),
            cancel,
        }
    }

    pub fn new_with_metrics(
        delta_rx: mpsc::Receiver<DeltaMsg>,
        text_seg_tx: mpsc::Sender<TextSegment>,
        buffer_ms: BufferMsFn,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        turn_id: u64,
        metrics: Arc<dyn MetricsSink>,
        cancel: CancellationToken,
        config: TokenizerConfig,
    ) -> Self {
        Self {
            delta_rx,
            text_seg_tx,
            buffer_ms,
            session_start_ts,
            llm_start,
            turn_id,
            metrics,
            config: config.normalize(),
            cancel,
        }
    }

    pub fn from_env(
        delta_rx: mpsc::Receiver<DeltaMsg>,
        text_seg_tx: mpsc::Sender<TextSegment>,
        buffer_ms: BufferMsFn,
        session_start_ts: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        turn_id: u64,
        cancel: CancellationToken,
    ) -> Self {
        Self::new(
            delta_rx,
            text_seg_tx,
            buffer_ms,
            session_start_ts,
            llm_start,
            turn_id,
            cancel,
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
        let chunk = TextSegment {
            turn_id: self.turn_id,
            text,
            llm_start_ts,
            first_token_ts: if *first { first_delta_ts } else { None },
            last_token_ts: last_delta_ts,
            segment_sent_ts: Instant::now(),
        };

        if self.text_seg_tx.send(chunk).await.is_err() {
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
        let mut llm_first_token_emitted = false;
        let mut eager_chunks_remaining = self.config.eager_chunks;
        let mut input_closed = false;

        let relax_buffer_ms = self.config.relax_buffer_ms;
        let relax_log = self.config.relax_log;
        let mut relax_active = false;
        'run: loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    break 'run;
                }
                maybe = self.delta_rx.recv() => {
                    let mut is_eof = false;
                    match maybe {
                        Some(DeltaMsg::Delta(delta)) => {
                            if input_closed {
                                continue;
                            }
                            let now = Instant::now();

                            // Capture time of first (non-empty) token.
                            if first_delta_ts.is_none() && !delta.is_empty() {
                                first_delta_ts = Some(now);
                                if !llm_first_token_emitted {
                                    self.metrics.on_event(MetricEvent {
                                        turn_id: self.turn_id,
                                        kind: MetricEventKind::LlmFirstToken,
                                        ts: now,
                                    });
                                    llm_first_token_emitted = true;
                                }
                            }
                            if !delta.is_empty() {
                                last_delta_ts = Some(now);
                            }

                            buf.push_str(&delta);
                        }
                        Some(DeltaMsg::Eof) => {
                            input_closed = true;
                            is_eof = true;
                        }
                        None => {
                            input_closed = true;
                            is_eof = true;
                        }
                    }

                    if is_eof && buf.is_empty() {
                        break 'run;
                    }

                    loop {
                        let mut buffered_ms_for_log: Option<u64> = None;
                        let (thresholds, relax_now) = if eager_chunks_remaining > 0 {
                            (eager_thresholds, false)
                        } else {
                            let buffered_ms = (self.buffer_ms)();
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
                            let buffered_ms =
                                buffered_ms_for_log.unwrap_or_else(|| (self.buffer_ms)());
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
                                let _ = self
                                    .emit_segment(pending, &mut first, first_delta_ts, last_delta_ts)
                                    .await;
                            }
                            break;
                        };
                        let remaining = buf.split_off(cut_idx);
                        let pending = std::mem::replace(&mut buf, remaining);
                        if !self
                            .emit_segment(pending, &mut first, first_delta_ts, last_delta_ts)
                            .await
                        {
                            break 'run;
                        }
                        if eager_chunks_remaining > 0 {
                            eager_chunks_remaining -= 1;
                        }
                    }

                    if is_eof {
                        break 'run;
                    }
                }
            }
        }
    }
}

pub(crate) fn log_relax_transition(
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

pub(crate) fn find_flush_index(
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
    use std::sync::{Arc, OnceLock};
    use tokio::time::{Duration, timeout};

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

    #[test]
    fn flush_with_chinese_punctuation() {
        // 中文句号"。"是3字节 (U+3002)
        // "你好世界。" = 5 chars, byte lengths: 你(3) 好(3) 世(3) 界(3) 。(3) = 15 bytes
        let s = "你好世界。";
        let idx = find_flush_index(s, 1, 20, 30);
        // Should detect 。 as strong boundary and return full string length
        assert_eq!(idx, Some(s.len())); // 15 bytes
        assert_eq!(s.len(), 15);
    }

    #[test]
    fn flush_with_emoji() {
        // "Hello👍" = 6 chars, but 👍 is 4 bytes (U+1F44D)
        // H(1) e(1) l(1) l(1) o(1) 👍(4) = 9 bytes
        let s = "Hello👍";
        assert_eq!(s.chars().count(), 6);
        assert_eq!(s.len(), 9);

        // No boundary - should not flush until hard limit
        let idx = find_flush_index(s, 1, 5, 10);
        assert_eq!(idx, None); // Below hard_max, no boundary
    }

    #[test]
    fn flush_with_mixed_multibyte() {
        // Mixed: ASCII + Chinese + punctuation
        // "Hi你好！" = 5 chars, bytes: H(1) i(1) 你(3) 好(3) ！(3) = 11 bytes
        let s = "Hi你好！";
        assert_eq!(s.chars().count(), 5);
        assert_eq!(s.len(), 11);

        let idx = find_flush_index(s, 1, 10, 20);
        // ！ is strong boundary at end
        assert_eq!(idx, Some(s.len())); // 11 bytes
    }

    #[test]
    fn hard_cut_respects_char_boundaries_multibyte() {
        // "一二三四五六七八九十" = 10 chars, 30 bytes (each 3 bytes)
        let s = "一二三四五六七八九十";
        assert_eq!(s.chars().count(), 10);
        assert_eq!(s.len(), 30);

        // hard_max=8 chars → should cut at char boundary (8*3=24 bytes)
        let idx = find_flush_index(s, 1, 5, 8);
        // hard_cut should return byte index of 8th char end
        assert_eq!(idx, Some(24)); // 8 chars * 3 bytes = 24
        assert!(s.is_char_boundary(24));
    }

    #[tokio::test]
    async fn tokenizer_flushes_multiple_segments_from_single_delta() {
        let (llm_delta_tx, llm_delta_rx) = tokio::sync::mpsc::channel::<DeltaMsg>(8);
        let (text_seg_tx, mut text_seg_rx) = tokio::sync::mpsc::channel::<TextSegment>(8);
        let buffer_ms = Arc::new(|| 0u64);

        let session_start_ts = Instant::now();
        let llm_start = Arc::new(OnceLock::new());
        let cancel = CancellationToken::new();
        let tokenizer = Tokenizer::new(
            llm_delta_rx,
            text_seg_tx,
            buffer_ms,
            session_start_ts,
            llm_start,
            123,
            cancel,
            TokenizerConfig {
                eager_chunks: 0,
                relax_buffer_ms: 0,
                relax_log: false,
            },
        );

        let handle = tokio::spawn(tokenizer.run());

        llm_delta_tx
            .send(DeltaMsg::Delta(
                "这是第一句很长很长很长！这是第二句也很长很长很长！这是第三句也很长很长很长！这是第四句也很长很长很长！"
                    .to_string(),
            ))
            .await
            .unwrap();

        let first = timeout(Duration::from_millis(1000), text_seg_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let second = timeout(Duration::from_millis(1000), text_seg_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(first.text, second.text);

        let _ = llm_delta_tx.send(DeltaMsg::Eof).await;
        drop(llm_delta_tx);
        let _ = handle.await;
    }
}
