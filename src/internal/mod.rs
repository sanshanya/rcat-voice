pub(crate) mod env;
pub(crate) mod metrics;

#[cfg(any(
    feature = "asr-sherpa",
    feature = "gpt-sovits-onnx",
    feature = "turn-smart",
    all(feature = "gpt-sovits", target_os = "windows")
))]
pub(crate) mod model_locator;

#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub(crate) mod timing;
