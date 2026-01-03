use anyhow::Result;
use rcat_voice::generator;
use rcat_voice::streaming::StreamSession;
use tokio::time::{Duration, sleep};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let tts_engine = generator::build_from_env()?;
    let session = StreamSession::from_env(tts_engine);
    let control = session.control();
    control.mark_llm_start();

    let sender = control.sender();
    let text = "首段短有利于快速生成首个音频块，降低首包延迟。后续段长则能利用上下文保持连贯性。";
    let sender_clone = sender.clone();
    let send_task = tokio::spawn(async move {
        let chars: Vec<char> = text.chars().collect();
        for chunk in chars.chunks(3) {
            let part: String = chunk.iter().collect();
            if sender_clone.send(part).await.is_err() {
                break;
            }
            sleep(Duration::from_millis(80)).await;
        }
    });


    let _ = send_task.await;
    drop(sender);
    let drain_ms = std::env::var("STREAM_SIM_DRAIN_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000)
        .min(60_000);
    if drain_ms > 0 {
        sleep(Duration::from_millis(drain_ms)).await;
    }
    session.shutdown().await?;

    Ok(())
}
