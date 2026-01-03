use super::{Result, TtsEngine, TtsMetrics};
use async_trait::async_trait;
use anyhow::anyhow;

pub struct RemoteTts;

impl RemoteTts {
    pub fn new() -> Result<Self> {
        Err(anyhow!("Remote TTS backend is not implemented yet"))
    }
}

#[async_trait]
impl TtsEngine for RemoteTts {
    async fn speak(&self, _text: &str) -> Result<TtsMetrics> {
        Err(anyhow!("Remote TTS backend is not implemented yet"))
    }

    async fn stop(&self) -> Result<()> {
        Ok(())
    }
}
