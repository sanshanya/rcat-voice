//! Streaming text-to-speech pipeline with pluggable backends.
//!
//! Basic usage (OS TTS backend, no extra features required):
//! ```no_run
//! use rcat_voice::prelude::*;
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let tts = TtsEngineBuilder::new(TtsBackend::Os).build()?;
//! let session = StreamSession::builder(tts.clone()).build();
//! let control = session.control();
//! control.mark_llm_start();
//! control.sender().send("Hello,".to_string()).await?;
//! control.sender().send(" world!".to_string()).await?;
//! session.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod asr;
pub mod audio;
pub mod generator;
mod internal;
pub mod metrics;
pub mod pipeline;
#[cfg(any(feature = "tts-remote", feature = "tts-worker"))]
pub mod remote_tts_protocol;
pub mod streaming;
pub mod tokenizer;
pub mod turn;
#[cfg(feature = "tts-worker")]
pub mod worker;

pub mod prelude {
    pub use crate::asr::AsrSegment;
    #[cfg(feature = "asr-sherpa")]
    pub use crate::asr::{
        AsrModelDtype, SherpaAsrConfig, SherpaAsrModel, SherpaAsrStream, SherpaVadConfig,
    };
    pub use crate::audio::{AudioBackend, AudioBackendKind, AudioConfig, RodioConfig};
    #[cfg(feature = "asr-mic")]
    pub use crate::audio::{MicConfig, MicStream};
    #[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
    pub use crate::generator::{GptSovitsChunkPolicy, GptSovitsConfig};
    #[cfg(feature = "gpt-sovits-onnx")]
    pub use crate::generator::{GptSovitsOnnxConfig, GptSovitsOnnxSampling};
    pub use crate::generator::{
        SynthesizedAudio, TtsBackend, TtsEngine, TtsEngineBuilder, TtsMetrics,
    };
    pub use crate::metrics::{
        MetricEvent, MetricEventKind, MetricsSink, NoopMetricsSink, TracingMetricsSink,
        default_sink,
    };
    pub use crate::pipeline::PipelineConfig;
    pub use crate::streaming::{StreamConfig, StreamControl, StreamSession, StreamSessionBuilder};
    pub use crate::tokenizer::{Segment, TokenizerConfig};
    #[cfg(feature = "turn-smart")]
    pub use crate::turn::{SmartTurnConfig, SmartTurnDecision, SmartTurnDetector, SmartTurnModel};
}
