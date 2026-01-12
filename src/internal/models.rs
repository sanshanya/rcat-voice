use crate::internal::env;
use std::path::{Path, PathBuf};

pub(crate) const RCAT_MODELS_DIR_ENV: &str = "RCAT_MODELS_DIR";

fn looks_like_rcat_models_root(root: &Path) -> bool {
    if !root.is_dir() {
        return false;
    }
    ["ASR", "TTS", "TURN", "VAD"]
        .iter()
        .any(|name| root.join(name).is_dir())
}

fn auto_detect_models_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;

    // Typical layout:
    // - monorepo root: `./models`
    // - rcat-voice subproject: `../models`
    let candidates = [cwd.join("models"), cwd.join("..").join("models")];

    candidates
        .into_iter()
        .find(|p| looks_like_rcat_models_root(p))
}

/// Return the unified models root directory if configured.
///
/// Preferred: set `RCAT_MODELS_DIR=/path/to/models` (contains `ASR/`, `TTS/`, `TURN/`, `VAD/`).
pub(crate) fn models_root_dir() -> Option<PathBuf> {
    env::string(RCAT_MODELS_DIR_ENV)
        .map(PathBuf::from)
        .or_else(auto_detect_models_root)
}

fn category_dir(category: &str) -> Option<PathBuf> {
    let root = models_root_dir()?;
    let candidate = root.join(category);
    candidate.is_dir().then_some(candidate)
}

#[cfg(feature = "asr-sherpa")]
pub(crate) fn asr_dir() -> Option<PathBuf> {
    category_dir("ASR").or_else(models_root_dir)
}

#[cfg(feature = "asr-sherpa")]
pub(crate) fn vad_dir() -> Option<PathBuf> {
    category_dir("VAD").or_else(models_root_dir)
}

#[cfg(feature = "turn-smart")]
pub(crate) fn turn_dir() -> Option<PathBuf> {
    category_dir("TURN").or_else(models_root_dir)
}

#[cfg(any(feature = "gpt-sovits-onnx", all(feature = "gpt-sovits", target_os = "windows")))]
fn tts_dir() -> Option<PathBuf> {
    category_dir("TTS").or_else(models_root_dir)
}

#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub(crate) fn tts_gpt_sovits_dir() -> Option<PathBuf> {
    let root = tts_dir()?;
    let candidate = root.join("gpt-sovits");
    candidate.is_dir().then_some(candidate)
}

#[cfg(feature = "gpt-sovits-onnx")]
pub(crate) fn tts_gpt_sovits_onnx_dir() -> Option<PathBuf> {
    let root = tts_dir()?;
    let candidate = root.join("gpt-sovits-onnx");
    candidate.is_dir().then_some(candidate)
}
