#[cfg(any(
    feature = "asr-sherpa",
    feature = "gpt-sovits-onnx",
    all(feature = "gpt-sovits", target_os = "windows")
))]
use anyhow::{Result, bail};
#[cfg(any(
    feature = "asr-sherpa",
    feature = "gpt-sovits-onnx",
    all(feature = "gpt-sovits", target_os = "windows")
))]
use std::path::{Path, PathBuf};

#[cfg(feature = "asr-sherpa")]
pub(crate) fn first_existing_dir(root: &Path, label: &str, candidates: &[&str]) -> Result<PathBuf> {
    for rel in candidates {
        let base = root.join(rel);
        if base.is_dir() {
            return Ok(base);
        }
    }
    let tried = candidates
        .iter()
        .map(|rel| format!("`{}`", root.join(rel).display()))
        .collect::<Vec<_>>()
        .join(", ");
    bail!("{label} not found (tried: {tried})");
}

#[cfg(feature = "asr-sherpa")]
pub(crate) fn first_existing_file(base: &Path, label: &str, candidates: &[&str]) -> Result<PathBuf> {
    for rel in candidates {
        let path = base.join(rel);
        if path.is_file() {
            return Ok(path);
        }
    }
    let tried = candidates
        .iter()
        .map(|rel| format!("`{}`", base.join(rel).display()))
        .collect::<Vec<_>>()
        .join(", ");
    bail!("{label} not found (tried: {tried})");
}

#[cfg(any(feature = "gpt-sovits-onnx", all(feature = "gpt-sovits", target_os = "windows")))]
pub(crate) fn first_existing_file_rel<S: AsRef<str>>(
    base: &Path,
    label: &str,
    candidates: &[S],
) -> Result<PathBuf> {
    for rel in candidates {
        let path = base.join(rel.as_ref());
        if path.is_file() {
            return Ok(path);
        }
    }
    let tried = candidates
        .iter()
        .map(|rel| format!("`{}`", base.join(rel.as_ref()).display()))
        .collect::<Vec<_>>()
        .join(", ");
    bail!("{label} not found (tried: {tried})");
}

#[cfg(feature = "gpt-sovits-onnx")]
pub(crate) fn optional_existing_file_rel<S: AsRef<str>>(
    base: &Path,
    candidates: &[S],
) -> Option<PathBuf> {
    for rel in candidates {
        let path = base.join(rel.as_ref());
        if path.is_file() {
            return Some(path);
        }
    }
    None
}
