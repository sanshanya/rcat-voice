use anyhow::{Result, bail};

pub struct SystemBackend;

impl SystemBackend {
    pub fn new() -> Result<Self> {
        bail!("System audio backend is not implemented yet");
    }
}
