use anyhow::{Result, bail};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::warn;

use crate::audio::AudioBackend;

pub mod os;
pub mod remote;
#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub mod gpt_sovits;
#[cfg(feature = "gpt-sovits-onnx")]
pub mod gpt_sovits_onnx;
pub use os::OsTts;
pub use remote::RemoteTts;
#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub use gpt_sovits::{GptSovitsChunkPolicy, GptSovitsConfig, GptSovitsTts, build as build_gpt_sovits_tts};
#[cfg(feature = "gpt-sovits-onnx")]
pub use gpt_sovits_onnx::{
    GptSovitsOnnxConfig,
    GptSovitsOnnxSampling,
    GptSovitsOnnxTts,
    build as build_gpt_sovits_onnx_tts,
};

/// 单个 TTS 片段的生成与播放指标。
pub struct TtsMetrics {
    /// 生成开始时间戳。
    pub start_ts: Instant,
    /// 首个音频样本时间戳（流式时可用）。
    pub first_audio_ts: Option<Instant>,
    /// 生成完成时间戳。
    pub gen_done_ts: Instant,
    /// 播放完成时间戳。
    pub play_done_ts: Instant,
    /// 实际播放完成时间戳的可选回传通道。
    pub play_done_rx: Option<oneshot::Receiver<Instant>>,
}

/// 预先合成的音频与时间信息（用于合成/播放解耦）。
pub struct SynthesizedAudio {
    pub samples: Vec<f32>,
    pub start_ts: Instant,
    pub gen_done_ts: Instant,
}

/// 流式文本转语音引擎接口。
#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// 合成并播放给定的文本片段。
    async fn speak(&self, text: &str) -> Result<TtsMetrics>;
    /// 中断播放并清空已排队音频。
    async fn stop(&self) -> Result<()>;
    /// 是否支持合成与播放解耦。
    fn supports_synthesis_queue(&self) -> bool {
        false
    }
    /// 仅合成音频（不播放）。
    async fn synthesize(&self, _text: &str) -> Result<Option<SynthesizedAudio>> {
        Ok(None)
    }
    /// 播放预合成音频。
    async fn play_samples(&self, _audio: SynthesizedAudio) -> Result<Option<TtsMetrics>> {
        Ok(None)
    }
    /// 返回已缓存待播音频的毫秒数（若后端支持）。
    fn buffered_ms(&self) -> Option<u64> {
        None
    }
}

/// TTS backend selection for the builder API.
#[derive(Debug, Clone)]
pub enum TtsBackend {
    Os,
    Remote,
    #[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
    GptSovits(GptSovitsConfig),
    #[cfg(feature = "gpt-sovits-onnx")]
    GptSovitsOnnx(GptSovitsOnnxConfig),
}

/// Builder for `TtsEngine` with explicit backend and audio wiring.
pub struct TtsEngineBuilder {
    backend: TtsBackend,
    audio: Option<Arc<dyn AudioBackend>>,
}

impl TtsEngineBuilder {
    pub fn new(backend: TtsBackend) -> Self {
        Self { backend, audio: None }
    }

    pub fn audio_backend(mut self, audio: Arc<dyn AudioBackend>) -> Self {
        self.audio = Some(audio);
        self
    }

    pub fn build(self) -> Result<Arc<dyn TtsEngine>> {
        match self.backend {
            TtsBackend::Os => Ok(Arc::new(OsTts::new())),
            TtsBackend::Remote => Ok(Arc::new(RemoteTts::new()?)),
            #[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
            TtsBackend::GptSovits(config) => {
                let audio = self
                    .audio
                    .ok_or_else(|| anyhow::Error::msg("GptSovits backend requires an AudioBackend"))?;
                Ok(Arc::new(gpt_sovits::GptSovitsTts::from_config(config, audio)?))
            }
            #[cfg(feature = "gpt-sovits-onnx")]
            TtsBackend::GptSovitsOnnx(config) => {
                let audio = self
                    .audio
                    .ok_or_else(|| {
                        anyhow::Error::msg("GptSovitsOnnx backend requires an AudioBackend")
                    })?;
                Ok(Arc::new(gpt_sovits_onnx::GptSovitsOnnxTts::from_config(
                    config, audio,
                )?))
            }
        }
    }
}

pub fn build_from_env() -> Result<Arc<dyn TtsEngine>> {
    let backend = std::env::var("TTS_BACKEND")
        .unwrap_or_else(|_| default_backend().to_string())
        .to_lowercase();
    match backend.as_str() {
        "os" => Ok(Arc::new(OsTts::new())),
        "gpt-sovits" => build_gpt_sovits_from_env(),
        "gpt-sovits-onnx" => build_gpt_sovits_onnx_from_env(),
        "remote" => {
            warn!("TTS_BACKEND=remote selected, but RemoteTts is not implemented yet.");
            Ok(Arc::new(RemoteTts::new()?))
        }
        _ => bail!("Unknown TTS_BACKEND: {backend}"),
    }
}

fn default_backend() -> &'static str {
    if cfg!(all(feature = "gpt-sovits", target_os = "windows")) {
        "gpt-sovits"
    } else {
        "os"
    }
}

fn build_gpt_sovits_from_env() -> Result<Arc<dyn TtsEngine>> {
    #[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
    {
        let audio = crate::audio::build_from_env()?;
        gpt_sovits::build(audio)
    }
    #[cfg(not(all(feature = "gpt-sovits", target_os = "windows")))]
    {
        Err(anyhow::anyhow!(
            "TTS_BACKEND=gpt-sovits requires Windows and the `gpt-sovits` feature (use `--features gpt-sovits` or `--all-features`)"
        ))
    }
}

fn build_gpt_sovits_onnx_from_env() -> Result<Arc<dyn TtsEngine>> {
    #[cfg(feature = "gpt-sovits-onnx")]
    {
        let audio = crate::audio::build_from_env()?;
        gpt_sovits_onnx::build(audio)
    }
    #[cfg(not(feature = "gpt-sovits-onnx"))]
    {
        Err(anyhow::anyhow!(
            "TTS_BACKEND=gpt-sovits-onnx requires the `gpt-sovits-onnx` feature (use `--features gpt-sovits-onnx` or `--all-features`)"
        ))
    }
}
