use anyhow::{Result, bail};
#[cfg(feature = "turn-smart")]
use anyhow::Context;
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

#[cfg(any(feature = "gpt-sovits-onnx", all(feature = "gpt-sovits", target_os = "windows")))]
pub(crate) fn optional_existing_file_rel<S: AsRef<str>>(base: &Path, candidates: &[S]) -> Option<PathBuf> {
    for rel in candidates {
        let path = base.join(rel.as_ref());
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

#[cfg(feature = "turn-smart")]
pub(crate) fn resolve_unique_file_in_dir<F>(
    dir: &Path,
    label: &str,
    ext: &str,
    mut name_predicate: F,
) -> Result<PathBuf>
where
    F: FnMut(&str) -> bool,
{
    if !dir.is_dir() {
        bail!("{label} must be a directory, got: {}", dir.display());
    }

    let mut candidates = Vec::<PathBuf>::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read {label} directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
        {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if name_predicate(&file_name) {
            candidates.push(path);
        }
    }

    match candidates.len() {
        0 => bail!(
            "{label} points to a directory but no matching *.{ext} file was found: {}",
            dir.display()
        ),
        1 => Ok(candidates.swap_remove(0)),
        _ => {
            candidates.sort();
            let list = candidates
                .iter()
                .map(|p| format!("- {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n");
            bail!(
                "{label} points to a directory with multiple candidates. Please set {label} to an explicit file path.\n{list}"
            );
        }
    }
}
