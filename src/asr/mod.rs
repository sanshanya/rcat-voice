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

pub mod utils;

#[cfg(feature = "asr-sherpa")]
pub mod sherpa;

#[cfg(feature = "asr-sherpa")]
pub use sherpa::{SherpaAsrConfig, SherpaAsrModel, SherpaAsrStream, SherpaVadConfig};

/// Convenience: build an ASR stream from env vars.
#[cfg(feature = "asr-sherpa")]
pub fn build_from_env() -> anyhow::Result<SherpaAsrStream> {
    SherpaAsrStream::from_env()
}
