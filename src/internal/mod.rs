pub(crate) mod env;

#[cfg(any(
    feature = "asr-sherpa",
    feature = "gpt-sovits-onnx",
    feature = "turn-smart",
    all(feature = "gpt-sovits", target_os = "windows")
))]
pub(crate) mod models;

#[cfg(any(feature = "gpt-sovits-onnx", feature = "turn-smart"))]
pub(crate) mod ort_log;

#[cfg(any(
    feature = "asr-sherpa",
    feature = "gpt-sovits-onnx",
    feature = "turn-smart",
    all(feature = "gpt-sovits", target_os = "windows")
))]
pub(crate) mod model_locator;

#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub(crate) mod timing;
