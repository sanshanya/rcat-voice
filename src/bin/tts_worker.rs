#[cfg(not(feature = "tts-worker"))]
fn main() {
    eprintln!("This binary requires `--features tts-worker`.");
}

#[cfg(feature = "tts-worker")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rcat_voice::worker::run_from_env().await
}

