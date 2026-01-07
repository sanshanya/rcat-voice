use super::{Result, TtsEngine, TtsMetrics};
use anyhow::anyhow;
use async_trait::async_trait;

#[cfg(feature = "tts-remote")]
use anyhow::bail;
#[cfg(feature = "tts-remote")]
use std::sync::Arc;
#[cfg(feature = "tts-remote")]
use tokio::time::Instant;

#[cfg(feature = "tts-remote")]
use crate::audio::{AudioBackend, AudioStreamSegment, CancelToken};
#[cfg(feature = "tts-remote")]
use crate::internal::env;

pub struct RemoteTts {
    #[cfg(feature = "tts-remote")]
    inner: RemoteTtsInner,
}

#[cfg(feature = "tts-remote")]
struct RemoteTtsInner {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    voice: String,
    audio: Arc<dyn AudioBackend>,
    cancel: CancelToken,
}

impl RemoteTts {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "tts-remote")]
        {
            Ok(Self {
                inner: RemoteTtsInner::from_env()?,
            })
        }
        #[cfg(not(feature = "tts-remote"))]
        {
            Err(anyhow!(
                "TTS_BACKEND=remote requires the `tts-remote` feature (enable `--features tts-remote`)"
            ))
        }
    }
}

#[cfg(feature = "tts-remote")]
impl RemoteTtsInner {
    fn from_env() -> Result<Self> {
        let base_url = env::string("TTS_REMOTE_BASE_URL")
            .or_else(|| env::string("TTS_REMOTE_URL"))
            .unwrap_or_else(|| "http://127.0.0.1:7878".to_string());

        let endpoint = if base_url.trim_end_matches('/').ends_with("/v1/audio/speech") {
            base_url.trim().to_string()
        } else {
            format!("{}/v1/audio/speech", base_url.trim_end_matches('/'))
        };

        let api_key = env::string("TTS_REMOTE_API_KEY")
            .or_else(|| env::string("TTS_API_KEY"))
            .filter(|v| !v.trim().is_empty());

        let model = env::string("TTS_REMOTE_MODEL")
            .or_else(|| env::string("TTS_MODEL"))
            .unwrap_or_else(|| "gpt-sovits".to_string());

        let voice = env::string("TTS_REMOTE_VOICE")
            .or_else(|| env::string("TTS_VOICE"))
            .unwrap_or_else(|| "default".to_string());

        let audio = crate::audio::build_from_env()?;

        let client = reqwest::Client::builder().build()?;

        Ok(Self {
            client,
            endpoint,
            api_key,
            model,
            voice,
            audio,
            cancel: CancelToken::new(),
        })
    }

    async fn speak_pcm16_stream(&self, text: &str) -> Result<TtsMetrics> {
        use futures_util::StreamExt;

        if self.audio.sample_rate() != 32_000 || self.audio.channels() != 1 {
            bail!(
                "Remote TTS currently expects 32000Hz mono output; set AUDIO_SAMPLE_RATE=32000 and AUDIO_CHANNELS=1"
            );
        }

        let start_ts = Instant::now();
        let cancel_scope = self.cancel.scope();

        let mut segment = AudioStreamSegment::new(self.audio.as_ref());

        let request = crate::remote_tts_protocol::SpeechRequest {
            model: self.model.clone(),
            input: text.to_string(),
            voice: self.voice.clone(),
            response_format: "pcm16".to_string(),
            stream: true,
            sample_rate: Some(self.audio.sample_rate()),
            channels: Some(self.audio.channels()),
        };

        let mut builder = self.client.post(&self.endpoint).json(&request);
        if let Some(key) = self.api_key.as_deref() {
            builder = builder.bearer_auth(key);
        }

        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!("Remote TTS failed (status={}): {}", status.as_u16(), body);
        }

        let mut pending_lo: Option<u8> = None;
        let mut sample_buf: Vec<f32> = Vec::new();

        let mut stream = response.bytes_stream();
        while let Some(chunk_res) = stream.next().await {
            if cancel_scope.is_cancelled() {
                break;
            }

            let chunk = chunk_res?;
            if chunk.is_empty() {
                continue;
            }

            decode_pcm16le_to_f32(&chunk, &mut pending_lo, &mut sample_buf);
            if sample_buf.is_empty() {
                continue;
            }

            let accepted = segment.push(&sample_buf, &cancel_scope);
            sample_buf.clear();
            if !accepted {
                break;
            }
        }

        let gen_done_ts = Instant::now();
        let (first_audio_ts, playback) = segment.finish(cancel_scope.is_cancelled());

        Ok(TtsMetrics {
            start_ts,
            first_audio_ts,
            gen_done_ts,
            play_done_ts: playback.play_done_ts,
            play_done_rx: playback.play_done_rx,
        })
    }
}

#[cfg(feature = "tts-remote")]
fn decode_pcm16le_to_f32(input: &[u8], pending_lo: &mut Option<u8>, out: &mut Vec<f32>) {
    let mut i = 0usize;
    if let Some(lo) = pending_lo.take() {
        if let Some(&hi) = input.get(0) {
            let v = i16::from_le_bytes([lo, hi]);
            out.push(v as f32 / 32768.0);
            i = 1;
        } else {
            *pending_lo = Some(lo);
            return;
        }
    }

    while i + 1 < input.len() {
        let lo = input[i];
        let hi = input[i + 1];
        let v = i16::from_le_bytes([lo, hi]);
        out.push(v as f32 / 32768.0);
        i += 2;
    }

    if i < input.len() {
        *pending_lo = Some(input[i]);
    }
}

#[async_trait]
impl TtsEngine for RemoteTts {
    async fn speak(&self, text: &str) -> Result<TtsMetrics> {
        #[cfg(feature = "tts-remote")]
        {
            self.inner.speak_pcm16_stream(text).await
        }
        #[cfg(not(feature = "tts-remote"))]
        {
            let _ = text;
            Err(anyhow!(
                "TTS_BACKEND=remote requires the `tts-remote` feature (enable `--features tts-remote`)"
            ))
        }
    }

    async fn stop(&self) -> Result<()> {
        #[cfg(feature = "tts-remote")]
        {
            self.inner.cancel.cancel();
            self.inner.audio.stop();
            Ok(())
        }
        #[cfg(not(feature = "tts-remote"))]
        {
            Ok(())
        }
    }

    fn buffered_ms(&self) -> Option<u64> {
        #[cfg(feature = "tts-remote")]
        {
            self.inner.audio.buffered_ms()
        }
        #[cfg(not(feature = "tts-remote"))]
        {
            None
        }
    }
}
