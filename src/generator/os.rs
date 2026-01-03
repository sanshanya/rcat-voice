use super::{Result, TtsEngine, TtsMetrics};
use async_trait::async_trait;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::Instant;
#[cfg(target_os = "windows")]
use tracing::warn;

/// OS-native TTS implementation using system commands.
#[derive(Clone)]
pub struct OsTts {
    /// Keep track of the current child process to kill it on stop/cancel.
    current_child: Arc<Mutex<Option<tokio::process::Child>>>,
}

impl OsTts {
    pub fn new() -> Self {
        Self {
            current_child: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl TtsEngine for OsTts {
    async fn speak(&self, text: &str) -> Result<TtsMetrics> {
        let start_ts = Instant::now();
        #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
        let mut child = spawn_tts_process(text)?;

        // Windows：将文本写入 stdin。
        #[cfg(target_os = "windows")]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(mut stdin) = child.stdin.take() {
                if let Err(e) = stdin.write_all(text.as_bytes()).await {
                    warn!("Failed to write to TTS stdin: {}", e);
                }
            }
        }

        // 保存子进程句柄，便于取消。
        *self.current_child.lock().await = Some(child);

        // 等待进程结束。
        {
            let mut guard: tokio::sync::MutexGuard<'_, Option<tokio::process::Child>> =
                self.current_child.lock().await;
            if let Some(child) = guard.as_mut() {
                let _ = child.wait().await?;
            }
            *guard = None;
        }

        let done_ts = Instant::now();
        Ok(TtsMetrics {
            start_ts,
            first_audio_ts: Some(start_ts),
            gen_done_ts: done_ts,
            play_done_ts: done_ts,
            play_done_rx: None,
        })
    }

    async fn stop(&self) -> Result<()> {
        let mut guard: tokio::sync::MutexGuard<'_, Option<tokio::process::Child>> =
            self.current_child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
        }
        Ok(())
    }
}

fn spawn_tts_process(_text: &str) -> Result<tokio::process::Child> {
    #[cfg(target_os = "macos")]
    {
        let mut c = Command::new("say");
        c.arg(_text);
        c.stdout(Stdio::null()).stderr(Stdio::null());
        Ok(c.spawn()?)
    }

    #[cfg(target_os = "linux")]
    {
        // Fallback or multiple choices could be added here.
        let mut c = Command::new("spd-say");
        c.arg("-w"); // Wait for completion.
        c.arg(_text);
        c.stdout(Stdio::null()).stderr(Stdio::null());
        Ok(c.spawn()?)
    }

    #[cfg(target_os = "windows")]
    {
        let mut c = Command::new("powershell");
        c.args([
            "-NoProfile",
            "-Command",
            "Add-Type -AssemblyName System.Speech; \
             $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; \
             $t=[Console]::In.ReadToEnd(); $s.Speak($t);",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        Ok(c.spawn()?)
    }
}
