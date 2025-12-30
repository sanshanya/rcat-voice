use anyhow::Result;
use async_trait::async_trait;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::warn;

#[async_trait]
pub trait TtsEngine: Send + Sync {
    async fn speak(&self, text: &str) -> Result<()>;
    async fn stop(&self) -> Result<()>;
}

/// OS-native TTS implementation using system commands
#[derive(Clone)]
pub struct OsTts {
    /// Keep track of the current child process to kill it on stop/cancel
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
    async fn speak(&self, text: &str) -> Result<()> {
        let mut child = spawn_tts_process(text)?;

        // Window specific: write text to stdin
        #[cfg(target_os = "windows")]
        {
            use tokio::io::AsyncWriteExt;
            if let Some(mut stdin) = child.stdin.take() {
                if let Err(e) = stdin.write_all(text.as_bytes()).await {
                    warn!("Failed to write to TTS stdin: {}", e);
                }
            }
        }

        // Store reference to child for cancellation
        *self.current_child.lock().await = Some(child);

        // Wait for process to complete
        // We poll loosely or just wait. Since we need to support external cancellation (stop),
        // we'll rely on the `stop()` method to kill this process, but `speak` itself
        // will block naturally until the child exits (playback finishes).
        {
            let mut guard: tokio::sync::MutexGuard<'_, Option<tokio::process::Child>> =
                self.current_child.lock().await;
            if let Some(child) = guard.as_mut() {
                let _ = child.wait().await?;
            }
            // Cleanup after finish
            *guard = None;
        }

        Ok(())
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
        // Fallback or multiple choices could be added here
        let mut c = Command::new("spd-say");
        c.arg("-w"); // Wait for completion
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
