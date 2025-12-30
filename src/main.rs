use anyhow::Result;
use rcat_voice::chunker::{Chunk, Chunker};
use rcat_voice::deepseek::sse_stream_chat;
use rcat_voice::player::Player;
use rcat_voice::tts::OsTts;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let task_start = Instant::now(); // t0: Task origin

    // Configuration
    // Use /v1 for strict OpenAI compatibility mode as requested
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
    let api_key = std::env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| "sk-XXXXXXX".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

    info!("Starting rcat-voice pipeline...");
    info!("Base URL: {}", base_url);
    info!("Model: {}", model);

    // Channels & State
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    let (delta_tx, delta_rx) = mpsc::channel::<String>(128);
    let (chunk_tx, chunk_rx) = mpsc::channel::<Chunk>(64);
    let queued_ms = Arc::new(AtomicU64::new(0));

    // Components
    let tts_engine = Arc::new(OsTts::new());

    // Task 1: Player (Consumer of Chunks)
    let player = Player::new(
        chunk_rx,
        cancel_rx.clone(),
        queued_ms.clone(),
        tts_engine.clone(),
    );
    let player_handle = tokio::spawn(player.run());

    // Task 2: Chunker (Consumer of Deltas, Producer of Chunks)
    let chunker = Chunker::new(
        delta_rx,
        chunk_tx,
        cancel_rx.clone(),
        queued_ms.clone(),
        task_start,
    );
    let chunker_handle = tokio::spawn(chunker.run());

    // Task 3: DeepSeek SSE Client (Producer of Deltas)
    let messages = vec![
        json!({"role":"user","content":"请用两三句话解释为什么首段短、后续段长更适合流式TTS。"}),
    ];
    let sse_handle = tokio::spawn(sse_stream_chat(
        base_url,
        api_key,
        model,
        messages, // Added messages
        delta_tx,
        cancel_rx.clone(),
    ));

    // Simulation: Cancel after 45 seconds (or wait for user input in real app)
    // For POC, we keep it running long enough to hear the output.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(60)).await;
        // info!("Auto-cancelling after 60s timeout...");
        // let _ = cancel_tx.send(true);
    });

    // Wait for SSE to finish (or error)
    if let Err(e) = sse_handle.await? {
        info!("SSE task failed or finished with error: {:?}", e);
    } else {
        info!("SSE task finished successfully.");
    }

    // Wait for others to drain if needed, or close app
    // In a real app we might wait for the player queue to drain.
    // Here we give a small buffer for the player to finish the last chunks if SSE is done.
    // If SSE broke, chunker will close delta_rx, which closes chunk_tx, which closes chunk_rx, player finishes.
    chunker_handle.await?;
    player_handle.await?;

    Ok(())
}
