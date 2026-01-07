#[cfg(any(feature = "tts-remote", feature = "tts-worker"))]
use serde::{Deserialize, Serialize};

fn default_model() -> String {
    "gpt-sovits".to_string()
}

fn default_voice() -> String {
    "default".to_string()
}

fn default_response_format() -> String {
    "pcm16".to_string()
}

fn default_stream() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SpeechRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    /// OpenAI-compatible naming: response_format.
    /// We currently support streaming `pcm16` only.
    #[serde(default = "default_response_format")]
    pub response_format: String,
    /// If true, server streams raw audio bytes using chunked transfer.
    #[serde(default = "default_stream")]
    pub stream: bool,
    /// Desired sample rate (Hz) for PCM.
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// Desired channels for PCM.
    #[serde(default)]
    pub channels: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ErrorBody {
    pub message: String,
    #[serde(default)]
    pub r#type: Option<String>,
}

