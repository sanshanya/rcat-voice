use crate::internal::env;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::time::Instant;

const DEFAULT_MAX_TURNS: usize = 1024;

/// Atomic metric event kinds (turn-scoped).
#[derive(Debug, Clone)]
pub enum MetricEventKind {
    /// End-of-turn boundary confirmed (SpeechEnd committed).
    TurnEnd,
    /// LLM request start timestamp (baseline).
    LlmStart,
    /// First token (first non-empty delta) received from LLM.
    LlmFirstToken,
    /// First text segment (first chunk) sent to the TTS pipeline.
    TtsFirstSegmentSent,
    /// First playable audio timestamp from TTS (play-domain estimate when available).
    TtsFirstAudio,
    /// ASR inference latency for a single segment.
    AsrInference { infer_ms: u64 },
}

/// A single metric event bound to a `turn_id`.
#[derive(Debug, Clone)]
pub struct MetricEvent {
    pub turn_id: u64,
    pub kind: MetricEventKind,
    pub ts: Instant,
}

/// Metric sink interface.
///
/// This trait is synchronous so it can be called from `spawn_blocking` threads.
pub trait MetricsSink: Send + Sync {
    fn on_event(&self, event: MetricEvent);
}

#[derive(Debug, Default)]
pub struct NoopMetricsSink;

impl MetricsSink for NoopMetricsSink {
    fn on_event(&self, _event: MetricEvent) {}
}

#[derive(Debug, Default, Clone)]
struct TurnMetricsState {
    turn_end: Option<Instant>,
    llm_start: Option<Instant>,
    llm_first_token: Option<Instant>,
    tts_first_segment_sent: Option<Instant>,
    tts_first_audio: Option<Instant>,
    logged_llm_ttft: bool,
    logged_tts_ttfa: bool,
    logged_llm_to_tts_first_audio: bool,
    logged_e2e_ttfa: bool,
    asr_infer_ms_total: u64,
    asr_infer_ms_max: u64,
    asr_infer_count: u64,
}

/// Default metrics sink: logs derived timings to `tracing` when enabled via env vars.
///
/// Enable with one of:
/// - `VOICE_TTS_METRICS=1`
/// - `VOICE_STREAM_METRICS=1`
/// - `STREAM_METRICS=1`
/// - `ASR_INFER_LOG=1` (ASR-only)
pub struct TracingMetricsSink {
    enabled: bool,
    asr_infer_log: bool,
    max_turns: usize,
    state: Mutex<HashMap<u64, TurnMetricsState>>,
}

impl TracingMetricsSink {
    pub fn from_env() -> Self {
        let enabled = env::bool01("VOICE_TTS_METRICS", false)
            || env::bool01("VOICE_STREAM_METRICS", false)
            || env::bool01("STREAM_METRICS", false);
        let asr_infer_log = env::bool01("ASR_INFER_LOG", false) || enabled;
        let max_turns =
            env::usize_clamped("VOICE_METRICS_MAX_TURNS", DEFAULT_MAX_TURNS, 16, 1_000_000);
        Self {
            enabled,
            asr_infer_log,
            max_turns,
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl MetricsSink for TracingMetricsSink {
    fn on_event(&self, event: MetricEvent) {
        if !self.enabled {
            match &event.kind {
                MetricEventKind::AsrInference { .. } if self.asr_infer_log => {}
                _ => return,
            }
        }

        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.entry(event.turn_id).or_default();

        match event.kind {
            MetricEventKind::TurnEnd => {
                if entry.turn_end.is_none() {
                    entry.turn_end = Some(event.ts);
                }
            }
            MetricEventKind::LlmStart => {
                if entry.llm_start.is_none() {
                    entry.llm_start = Some(event.ts);
                }
            }
            MetricEventKind::LlmFirstToken => {
                if entry.llm_first_token.is_none() {
                    entry.llm_first_token = Some(event.ts);
                }
            }
            MetricEventKind::TtsFirstSegmentSent => {
                if entry.tts_first_segment_sent.is_none() {
                    entry.tts_first_segment_sent = Some(event.ts);
                }
            }
            MetricEventKind::TtsFirstAudio => {
                if entry.tts_first_audio.is_none() {
                    entry.tts_first_audio = Some(event.ts);
                }
            }
            MetricEventKind::AsrInference { infer_ms } => {
                entry.asr_infer_count = entry.asr_infer_count.saturating_add(1);
                entry.asr_infer_ms_total = entry.asr_infer_ms_total.saturating_add(infer_ms);
                entry.asr_infer_ms_max = entry.asr_infer_ms_max.max(infer_ms);

                if self.asr_infer_log {
                    tracing::info!(turn_id = event.turn_id, infer_ms, "指标: ASR 推理耗时");
                }
            }
        }

        if self.enabled {
            maybe_log_derived(event.turn_id, entry);
        }

        evict_if_needed(&mut guard, self.max_turns, event.turn_id);
    }
}

fn maybe_log_derived(turn_id: u64, state: &mut TurnMetricsState) {
    if !state.logged_llm_ttft {
        if let (Some(start), Some(first_token)) = (state.llm_start, state.llm_first_token) {
            let ttft = first_token.saturating_duration_since(start);
            tracing::info!(
                turn_id,
                llm_ttft_ms = ttft.as_millis() as u64,
                "指标: LLM 首 token 时延 (TTFT)"
            );
            state.logged_llm_ttft = true;
        }
    }

    if !state.logged_tts_ttfa {
        if let (Some(sent), Some(first_audio)) =
            (state.tts_first_segment_sent, state.tts_first_audio)
        {
            let ttfa = first_audio.saturating_duration_since(sent);
            tracing::info!(
                turn_id,
                tts_ttfa_ms = ttfa.as_millis() as u64,
                "指标: 音频生成延迟 (TTS_TTFA, 首段送入 TTS→首音)"
            );
            state.logged_tts_ttfa = true;
        }
    }

    if !state.logged_e2e_ttfa {
        if let (Some(turn_end), Some(first_audio)) = (state.turn_end, state.tts_first_audio) {
            let e2e = first_audio.saturating_duration_since(turn_end);
            tracing::info!(
                turn_id,
                e2e_ttfa_ms = e2e.as_millis() as u64,
                e2e_baseline = "turn_end",
                "指标: 端到端延迟 (E2E_TTFA, 用户说完→首音)"
            );
            state.logged_e2e_ttfa = true;
            // Prefer E2E TTFA over LLM->TTS when turn_end exists.
            state.logged_llm_to_tts_first_audio = true;
        }
    }

    if !state.logged_llm_to_tts_first_audio && !state.logged_e2e_ttfa {
        if let (Some(llm_start), Some(first_audio)) = (state.llm_start, state.tts_first_audio) {
            let since_llm = first_audio.saturating_duration_since(llm_start);
            tracing::info!(
                turn_id,
                e2e_ttfa_ms = since_llm.as_millis() as u64,
                e2e_baseline = "llm_start",
                "指标: 端到端延迟 (E2E_TTFA, LLM 起点→首音)"
            );
            state.logged_llm_to_tts_first_audio = true;
        }
    }
}

fn evict_if_needed(map: &mut HashMap<u64, TurnMetricsState>, max_turns: usize, keep_turn_id: u64) {
    if max_turns == 0 || map.len() <= max_turns {
        return;
    }

    let overflow = map.len().saturating_sub(max_turns);
    if overflow == 0 {
        return;
    }

    let mut keys: Vec<u64> = map
        .keys()
        .copied()
        .filter(|&k| k != 0 && k != keep_turn_id)
        .collect();
    keys.sort_unstable();
    for k in keys.into_iter().take(overflow) {
        map.remove(&k);
    }
}

static DEFAULT_SINK: OnceLock<Arc<dyn MetricsSink>> = OnceLock::new();

/// Default metrics sink instance (env-controlled tracing logger).
pub fn default_sink() -> Arc<dyn MetricsSink> {
    DEFAULT_SINK
        .get_or_init(|| Arc::new(TracingMetricsSink::from_env()))
        .clone()
}
