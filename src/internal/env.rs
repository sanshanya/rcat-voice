#[cfg(feature = "gpt-sovits-onnx")]
use anyhow::{Result, anyhow};
#[cfg(feature = "gpt-sovits-onnx")]
use std::path::PathBuf;

pub(crate) fn get<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.parse().ok()
}

pub(crate) fn string(key: &str) -> Option<String> {
    let value = std::env::var(key).ok()?;
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn bool01(key: &str, default: bool) -> bool {
    let Some(raw) = std::env::var(key).ok() else {
        return default;
    };
    match raw.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => true,
        "0" | "false" | "no" | "n" | "off" => false,
        _ => default,
    }
}

pub(crate) fn u64_clamped(key: &str, default: u64, min: u64, max: u64) -> u64 {
    get::<u64>(key).unwrap_or(default).clamp(min, max)
}

pub(crate) fn usize_clamped(key: &str, default: usize, min: usize, max: usize) -> usize {
    get::<usize>(key).unwrap_or(default).clamp(min, max)
}

pub(crate) fn usize_threshold_triplet(
    prefix: &str,
    default_min_chars: usize,
    default_soft_max: usize,
    default_hard_max: usize,
    max_chars: usize,
) -> (usize, usize, usize) {
    let max_chars = max_chars.max(1);
    let min_chars = usize_clamped(
        &format!("{prefix}_MIN_CHARS"),
        default_min_chars,
        1,
        max_chars,
    );
    let soft_max = usize_clamped(
        &format!("{prefix}_SOFT_MAX_CHARS"),
        default_soft_max,
        1,
        max_chars,
    )
    .max(min_chars);
    let hard_max = usize_clamped(
        &format!("{prefix}_HARD_MAX_CHARS"),
        default_hard_max,
        1,
        max_chars,
    )
    .max(soft_max);
    (min_chars, soft_max, hard_max)
}

pub(crate) fn u32_clamped(key: &str, default: u32, min: u32, max: u32) -> u32 {
    get::<u32>(key).unwrap_or(default).clamp(min, max)
}

pub(crate) fn u16_clamped(key: &str, default: u16, min: u16, max: u16) -> u16 {
    get::<u16>(key).unwrap_or(default).clamp(min, max)
}

#[cfg(feature = "asr-sherpa")]
pub(crate) fn i32_clamped(key: &str, default: i32, min: i32, max: i32) -> i32 {
    get::<i32>(key).unwrap_or(default).clamp(min, max)
}

#[cfg(any(feature = "asr-sherpa", feature = "turn-smart"))]
pub(crate) fn f32_clamped(key: &str, default: f32, min: f32, max: f32) -> f32 {
    get::<f32>(key).unwrap_or(default).clamp(min, max)
}

#[cfg(all(feature = "gpt-sovits", target_os = "windows"))]
pub(crate) fn i64_clamped(key: &str, default: i64, min: i64, max: i64) -> i64 {
    get::<i64>(key).unwrap_or(default).clamp(min, max)
}

#[cfg(feature = "gpt-sovits-onnx")]
pub(crate) fn path_opt(key: &str) -> Result<Option<PathBuf>> {
    let Some(value) = string(key) else {
        return Ok(None);
    };
    let path = PathBuf::from(&value);
    if path.exists() {
        Ok(Some(path))
    } else {
        Err(anyhow!("{key} not found: {}", path.display()).into())
    }
}
