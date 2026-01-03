use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, watch};

use tokio::time::Instant;

// 放松模式下的硬上限字符数。
const RELAXED_HARD_MAX_CHARS: usize = 120;

#[derive(Debug, Clone)]
/// 发送到 TTS 管线的文本片段（含时间戳）。
pub struct Segment {
    /// 片段文本内容。
    pub text: String,
    /// 任务起点时间戳（t0）。
    pub task_start: Instant,
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
    pub min_chars: usize,
    pub max_chars: usize,
    pub boundary_overflow: usize,
    pub relax_buffer_ms: u64,
    pub relax_scale: f32,
    pub relax_boundary_window: usize,
    pub relax_overflow: usize,
    pub relax_log: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            eager_chunks: 2,
            min_chars: 20,
            max_chars: 50,
            boundary_overflow: 20,
            relax_buffer_ms: 200,
            relax_scale: 1.5,
            relax_boundary_window: 24,
            relax_overflow: 30,
            relax_log: false,
        }
    }
}

impl TokenizerConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(value) = std::env::var("CHUNKER_EAGER_CHUNKS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.eager_chunks = parsed;
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_MIN_CHARS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.min_chars = parsed.clamp(5, 200);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_MAX_CHARS") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.max_chars = parsed.clamp(cfg.min_chars + 5, 400);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_BOUNDARY_OVERFLOW") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.boundary_overflow = parsed.clamp(0, 200);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_RELAX_BUFFER_MS") {
            if let Ok(parsed) = value.parse::<u64>() {
                cfg.relax_buffer_ms = parsed.clamp(0, 60_000);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_RELAX_SCALE") {
            if let Ok(parsed) = value.parse::<f32>() {
                cfg.relax_scale = parsed.clamp(1.0, 3.0);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_RELAX_BOUNDARY_WINDOW") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.relax_boundary_window = parsed.clamp(0, 200);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_RELAX_OVERFLOW") {
            if let Ok(parsed) = value.parse::<usize>() {
                cfg.relax_overflow = parsed.clamp(0, 400);
            }
        }
        if let Ok(value) = std::env::var("TOKENIZER_RELAX_LOG") {
            cfg.relax_log = value == "1";
        }
        cfg.normalize()
    }

    pub fn normalize(mut self) -> Self {
        self.min_chars = self.min_chars.clamp(5, 200);
        self.max_chars = self.max_chars.clamp(self.min_chars + 5, 400);
        self.boundary_overflow = self.boundary_overflow.clamp(0, 200);
        self.relax_buffer_ms = self.relax_buffer_ms.clamp(0, 60_000);
        self.relax_scale = self.relax_scale.clamp(1.0, 3.0);
        self.relax_boundary_window = self.relax_boundary_window.clamp(0, 200);
        self.relax_overflow = self.relax_overflow.clamp(0, 400);
        self
    }
}

pub struct Tokenizer {
    delta_rx: mpsc::Receiver<String>,
    chunk_tx: mpsc::Sender<Segment>,
    cancel_rx: watch::Receiver<bool>,
    pause_rx: watch::Receiver<bool>,
    buffer_ms_rx: watch::Receiver<u64>,
    task_start: Instant, // t0 基准
    llm_start: Arc<OnceLock<Instant>>,
    config: TokenizerConfig,
}

impl Tokenizer {
    pub fn new(
        delta_rx: mpsc::Receiver<String>,
        chunk_tx: mpsc::Sender<Segment>,
        cancel_rx: watch::Receiver<bool>,
        pause_rx: watch::Receiver<bool>,
        buffer_ms_rx: watch::Receiver<u64>,
        task_start: Instant,
        llm_start: Arc<OnceLock<Instant>>,
        config: TokenizerConfig,
    ) -> Self {
        Self {
            delta_rx,
            chunk_tx,
            cancel_rx,
            pause_rx,
            buffer_ms_rx,
            task_start,
            llm_start,
            config: config.normalize(),
        }
    }

    pub fn from_env(
        delta_rx: mpsc::Receiver<String>,
        chunk_tx: mpsc::Sender<Segment>,
        cancel_rx: watch::Receiver<bool>,
        pause_rx: watch::Receiver<bool>,
        buffer_ms_rx: watch::Receiver<u64>,
        task_start: Instant,
        llm_start: Arc<OnceLock<Instant>>,
    ) -> Self {
        Self::new(
            delta_rx,
            chunk_tx,
            cancel_rx,
            pause_rx,
            buffer_ms_rx,
            task_start,
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

        let task_start = self
            .llm_start
            .get()
            .copied()
            .unwrap_or(self.task_start);
        let chunk = Segment {
            text,
            task_start,
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
        let mut buf = String::new();
        let mut first = true;
        let mut first_delta_ts: Option<Instant> = None;
        let mut last_delta_ts: Option<Instant> = None;
        let mut eager_chunks_remaining = self.config.eager_chunks;
        let eager_chunks_default = eager_chunks_remaining;
        let mut cancel_closed = false;
        let mut pause_closed = false;

        let min_chars = self.config.min_chars;
        let max_chars = self.config.max_chars;
        let normal_overflow = self.config.boundary_overflow;
        let hard_max = max_chars.min(600);
        let relax_buffer_ms = self.config.relax_buffer_ms;
        let relax_scale = self.config.relax_scale;
        let relax_boundary_window = self.config.relax_boundary_window;
        let relax_overflow = self.config.relax_overflow;
        let relaxed_max_chars = ((max_chars as f32) * relax_scale).round() as usize;
        let relaxed_max_chars = relaxed_max_chars.clamp(max_chars, RELAXED_HARD_MAX_CHARS);
        let relaxed_hard_max = RELAXED_HARD_MAX_CHARS;
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
                res = self.pause_rx.changed(), if !pause_closed => {
                    if res.is_err() {
                        pause_closed = true;
                        continue;
                    }
                    if *self.pause_rx.borrow() {
                        buf.clear();
                        first = true;
                        first_delta_ts = None;
                        last_delta_ts = None;
                        eager_chunks_remaining = eager_chunks_default;
                    }
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

                    let (min_c, max_c, limit, relax_now) = if eager_chunks_remaining > 0 {
                        (10usize, 20usize, 20usize, false) // Eager chunks for low TTFA/second-play delay
                    } else {
                        let buffered_ms = *self.buffer_ms_rx.borrow();
                        let relax_now = relax_buffer_ms > 0 && buffered_ms >= relax_buffer_ms;
                        let max_c = if relax_now { relaxed_max_chars } else { max_chars };
                        let limit = if relax_now { relaxed_hard_max } else { hard_max };
                        (min_chars, max_c, limit, relax_now)
                    };
                    if relax_log && relax_now != relax_active {
                        relax_active = relax_now;
                        let buffered_ms = *self.buffer_ms_rx.borrow();
                        if relax_active {
                            tracing::info!(
                                "Tokenizer relax on (buffer={}ms, max_chars={}, hard_max={})",
                                buffered_ms,
                                max_c,
                                limit
                            );
                        } else {
                            tracing::info!(
                                "Tokenizer relax off (buffer={}ms)",
                                buffered_ms
                            );
                        }
                    }
                    let trigger_strong_boundary = !relax_now;
                    let trigger_weak_boundary = !relax_now;
                    let strong_min_chars = if relax_now { 1 } else { min_chars };
                    let weak_min_chars = min_chars;
                    let hard_limit = if relax_now {
                        relaxed_hard_max.saturating_add(relax_overflow)
                    } else {
                        limit.saturating_add(normal_overflow)
                    };
                    if let Some(cut_idx) = find_flush_index(
                        &buf,
                        min_c,
                        max_c,
                        limit,
                        hard_limit,
                        strong_min_chars,
                        weak_min_chars,
                        trigger_strong_boundary,
                        trigger_weak_boundary,
                        relax_now,
                        relax_boundary_window,
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

fn find_flush_index(
    s: &str,
    min_chars: usize,
    max_chars: usize,
    hard_max: usize,
    hard_limit: usize,
    strong_min_chars: usize,
    weak_min_chars: usize,
    trigger_strong_boundary: bool,
    trigger_weak_boundary: bool,
    relax_mode: bool,
    relax_boundary_window: usize,
) -> Option<usize> {
    let mut count = 0usize;
    let mut last_strong: Option<(usize, usize)> = None;
    let mut last_weak: Option<(usize, usize)> = None;
    let mut hard_cut: Option<usize> = None;

    for (idx, ch) in s.char_indices() {
        count += 1;
        if count == hard_max {
            hard_cut = Some(idx + ch.len_utf8());
        }
        if matches!(ch, '。' | '！' | '？' | '!' | '?' | '\n') {
            last_strong = Some((idx + ch.len_utf8(), count));
        } else if matches!(ch, '，' | ',' | '；' | ';' | '：' | ':') {
            last_weak = Some((idx + ch.len_utf8(), count));
        }
    }

    if count < min_chars {
        return None;
    }

    let ends_with_strong = last_strong.map_or(false, |(idx, _)| idx == s.len());
    if trigger_strong_boundary && ends_with_strong {
        return Some(s.len());
    }
    let ends_with_weak = last_weak.map_or(false, |(idx, _)| idx == s.len());
    if trigger_weak_boundary && ends_with_weak {
        return Some(s.len());
    }

    if count < max_chars {
        return None;
    }
    if relax_mode && count < hard_max {
        return None;
    }

    if relax_mode {
        let window_start = count.saturating_sub(relax_boundary_window);
        let near_strong = last_strong
            .filter(|(_, boundary_count)| *boundary_count >= window_start);
        let near_weak = last_weak
            .filter(|(_, boundary_count)| *boundary_count >= window_start);
        if count >= hard_max {
            if let Some((idx, boundary_count)) = near_strong {
                if boundary_count >= strong_min_chars {
                    return Some(idx);
                }
            }
            if let Some((idx, boundary_count)) = near_weak {
                if boundary_count >= weak_min_chars {
                    return Some(idx);
                }
            }
            if count < hard_limit {
                return None;
            }
            if let Some((idx, boundary_count)) = last_strong {
                if boundary_count >= strong_min_chars {
                    return Some(idx);
                }
            }
            if let Some((idx, boundary_count)) = last_weak {
                if boundary_count >= weak_min_chars {
                    return Some(idx);
                }
            }
            if let Some((idx, _)) = last_strong.or(last_weak) {
                return Some(idx);
            }
            return hard_cut.or_else(|| Some(s.len()));
        }
    } else if count >= hard_max {
        if let Some((idx, boundary_count)) = last_strong {
            if boundary_count >= strong_min_chars {
                return Some(idx);
            }
        }
        if let Some((idx, boundary_count)) = last_weak {
            if boundary_count >= weak_min_chars {
                return Some(idx);
            }
        }
        if let Some((idx, _)) = last_strong.or(last_weak) {
            return Some(idx);
        }
        return hard_cut.or_else(|| Some(s.len()));
    }

    if let Some((idx, boundary_count)) = last_strong {
        if boundary_count >= strong_min_chars {
            return Some(idx);
        }
    }
    if let Some((idx, boundary_count)) = last_weak {
        if boundary_count >= weak_min_chars {
            return Some(idx);
        }
    }

    None
}
