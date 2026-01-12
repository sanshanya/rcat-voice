//! Turn detection types and traits.
//!
//! 端点检测核心类型，用于统一静音门控、VAD、Smart Turn 等逻辑。

use crate::asr::VadEvent;
use smallvec::SmallVec;
use tokio::time::Instant;

/// 端点事件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEventKind {
    /// 语音开始（从静音进入说话）
    SpeechStart,
    /// 语音结束（从说话进入静音）
    SpeechEnd,
    /// 轮次确认（满足端点条件，可提交）
    TurnCommitted,
}

/// 端点事件
#[derive(Debug, Clone)]
pub struct TurnEvent {
    pub kind: TurnEventKind,
    pub ts: Instant,
}

impl TurnEvent {
    pub fn speech_start(ts: Instant) -> Self {
        Self {
            kind: TurnEventKind::SpeechStart,
            ts,
        }
    }

    pub fn speech_end(ts: Instant) -> Self {
        Self {
            kind: TurnEventKind::SpeechEnd,
            ts,
        }
    }

    pub fn turn_committed(ts: Instant) -> Self {
        Self {
            kind: TurnEventKind::TurnCommitted,
            ts,
        }
    }
}

/// 音频帧引用（零拷贝输入）
#[derive(Debug, Clone)]
pub struct AudioFrameRef<'a> {
    pub samples: &'a [i16],
    pub sample_rate: u32,
    pub channels: u16,
    pub ts: Instant,
}

/// 端点检测器配置
#[derive(Debug, Clone)]
pub struct TurnDetectorConfig {
    /// 最小静音时长（ms），超过后开始评估端点
    pub min_silence_ms: u64,
    /// 确认延迟（ms），端点候选后需持续静音的时长
    pub commit_ms: u64,
    /// 强制结束时长（ms），超过后无条件提交
    pub force_end_ms: u64,
    /// 评估间隔（ms）
    pub eval_interval_ms: u64,
    /// 静音阈值（能量，fallback 用）
    pub silence_threshold: u16,
}

impl Default for TurnDetectorConfig {
    fn default() -> Self {
        Self {
            min_silence_ms: 250,
            commit_ms: 100,
            force_end_ms: 2000,
            eval_interval_ms: 80,
            silence_threshold: 200,
        }
    }
}

/// 端点检测器 trait
///
/// 统一接口用于不同的端点检测策略：
/// - 纯 VAD 静音门控 (`VadGateTurnDetector`)
/// - Smart Turn 模型 (`SmartTurnDetector`)
/// - 组合策略
pub trait TurnBoundaryDetector: Send {
    /// 推送音频帧（用于需要处理音频的检测器）
    fn push_audio(&mut self, frame: AudioFrameRef<'_>, out: &mut SmallVec<[TurnEvent; 4]>);

    /// 推送 VAD 事件（用于基于 VAD 的检测器）
    fn push_vad(&mut self, event: VadEvent, out: &mut SmallVec<[TurnEvent; 4]>);

    /// 时间推进（用于静音累计等基于时间的逻辑）
    ///
    /// `now` 应使用 `frame.ts` 而非 `Instant::now()` 以确保确定性
    fn tick(&mut self, now: Instant, out: &mut SmallVec<[TurnEvent; 4]>);

    /// 重置状态（新轮次开始）
    fn reset(&mut self);
}
