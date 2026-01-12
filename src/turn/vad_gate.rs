//! VAD-based turn detection (pure silence gating).
//!
//! 纯 VAD 静音门控端点检测，不使用 Smart Turn 模型。

use super::types::{AudioFrameRef, TurnBoundaryDetector, TurnDetectorConfig, TurnEvent};
use crate::asr::VadEvent;
use smallvec::SmallVec;
use tokio::time::Instant;

/// 基于 VAD 的端点检测器
///
/// 使用 VadEvent 判断语音活动，通过静音累计触发端点。
pub struct VadGateTurnDetector {
    config: TurnDetectorConfig,
    /// 是否在说话
    speaking: bool,
    /// 语音开始时刻
    speech_start_ts: Option<Instant>,
    /// 静音开始时刻
    silence_start_ts: Option<Instant>,
    /// 是否已提交（防止重复 TurnCommitted）
    committed: bool,
}

impl VadGateTurnDetector {
    pub fn new(config: TurnDetectorConfig) -> Self {
        Self {
            config,
            speaking: false,
            speech_start_ts: None,
            silence_start_ts: None,
            committed: false,
        }
    }

    pub fn from_env() -> Self {
        use crate::internal::env;
        // Compat: allow both TURN_* and legacy SMART_TURN_* env vars.
        let min_silence_ms = env::get::<u64>("TURN_MIN_SILENCE_MS")
            .or_else(|| env::get::<u64>("SMART_TURN_MIN_SILENCE_MS"))
            .unwrap_or(250)
            .clamp(50, 2000);
        let commit_ms = env::get::<u64>("TURN_COMMIT_MS")
            .or_else(|| env::get::<u64>("SMART_TURN_COMMIT_MS"))
            .unwrap_or(100)
            .clamp(0, 1000);
        let min_force_end_ms = min_silence_ms.saturating_add(commit_ms);
        let force_end_ms = env::get::<u64>("TURN_FORCE_END_MS")
            .or_else(|| env::get::<u64>("SMART_TURN_FORCE_END_MS"))
            .unwrap_or(2000)
            .clamp(min_force_end_ms, 60_000);
        let eval_interval_ms = env::get::<u64>("TURN_EVAL_INTERVAL_MS")
            .or_else(|| env::get::<u64>("SMART_TURN_EVAL_INTERVAL_MS"))
            .unwrap_or(80)
            .clamp(10, 500);
        let silence_threshold = env::get::<u16>("TURN_SILENCE_THRESHOLD")
            .or_else(|| env::get::<u16>("SMART_TURN_SILENCE_ABS"))
            .unwrap_or(200)
            .clamp(0, 20_000);
        let config = TurnDetectorConfig {
            min_silence_ms,
            commit_ms,
            force_end_ms,
            eval_interval_ms,
            silence_threshold,
        };
        Self::new(config)
    }

    fn trailing_silence_ms(&self, now: Instant) -> u64 {
        self.silence_start_ts
            .map(|start| now.saturating_duration_since(start).as_millis() as u64)
            .unwrap_or(0)
    }
}

impl TurnBoundaryDetector for VadGateTurnDetector {
    fn push_audio(&mut self, _frame: AudioFrameRef<'_>, _out: &mut SmallVec<[TurnEvent; 4]>) {
        // VadGateTurnDetector 不直接处理音频，而是依赖 VadEvent
    }

    fn push_vad(&mut self, event: VadEvent, out: &mut SmallVec<[TurnEvent; 4]>) {
        if self.committed {
            return;
        }
        match event {
            VadEvent::SpeechStart { ts } => {
                self.speaking = true;
                self.speech_start_ts = Some(ts);
                self.silence_start_ts = None;

                out.push(TurnEvent::speech_start(ts));
            }
            VadEvent::SpeechEnd { ts, .. } => {
                self.speaking = false;
                self.silence_start_ts = Some(ts);

                out.push(TurnEvent::speech_end(ts));
            }
        }
    }

    fn tick(&mut self, now: Instant, out: &mut SmallVec<[TurnEvent; 4]>) {
        if self.committed {
            return;
        }
        // 只在有过语音且当前静音时评估
        if self.speech_start_ts.is_none() || self.speaking || self.silence_start_ts.is_none() {
            return;
        }

        let silence_ms = self.trailing_silence_ms(now);

        // 强制结束检查
        if silence_ms >= self.config.force_end_ms {
            out.push(TurnEvent::turn_committed(now));
            self.committed = true;
            return;
        }

        // 正常端点检查：min_silence + commit
        let commit_threshold = self
            .config
            .min_silence_ms
            .saturating_add(self.config.commit_ms);
        if silence_ms >= commit_threshold {
            out.push(TurnEvent::turn_committed(now));
            self.committed = true;
        }
    }

    fn reset(&mut self) {
        self.speaking = false;
        self.speech_start_ts = None;
        self.silence_start_ts = None;
        self.committed = false;
    }
}
