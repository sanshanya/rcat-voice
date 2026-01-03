use anyhow::{Result, bail};

pub struct WasapiBackend;

impl WasapiBackend {
    pub fn new() -> Result<Self> {
        bail!("WASAPI backend is not implemented yet");
    }
}
