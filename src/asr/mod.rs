use tokio::time::Instant;

/// 单个 ASR 片段的识别结果（可能是中间结果或最终结果）。
#[derive(Debug, Clone)]
pub struct AsrSegment {
    pub text: String,
    pub finished: bool,
    pub idx: usize,
    pub start: f32,
    pub end: f32,
    pub channel: Option<u16>,
}

/// VAD 事件：语音活动检测边沿事件。
///
/// `ts` 表示检测时刻 (detection time)，不是音频边界时刻。
#[derive(Debug, Clone)]
pub enum VadEvent {
    /// 检测到语音开始
    SpeechStart { ts: Instant },
    /// 检测到语音结束
    SpeechEnd { ts: Instant, duration_ms: u32 },
}

/// VAD 状态快照（用于状态查询，避免丢边沿）。
#[derive(Debug, Clone)]
pub struct VadState {
    /// 当前是否在说话
    pub speaking: bool,
    /// 上次状态变化时刻
    pub last_change_ts: Instant,
    /// 单调递增序列号（用于检测变化）
    pub seq: u64,
}

impl Default for VadState {
    fn default() -> Self {
        Self {
            speaking: false,
            last_change_ts: Instant::now(),
            seq: 0,
        }
    }
}

pub mod utils;

#[cfg(feature = "asr-sherpa")]
pub mod sherpa;

#[cfg(feature = "asr-sherpa")]
pub use sherpa::{
    AsrModelDtype, SherpaAsrConfig, SherpaAsrModel, SherpaAsrStream, SherpaVadConfig,
};

/// Convenience: build an ASR stream from env vars.
#[cfg(feature = "asr-sherpa")]
pub fn build_from_env() -> anyhow::Result<SherpaAsrStream> {
    SherpaAsrStream::from_env()
}
