use anyhow::Result;
use rcat_voice::generator::TtsEngine;
use rcat_voice::pipeline::{Pipeline, PipelineConfig};
use rcat_voice::tokenizer::{DeltaMsg, TextSegment, Tokenizer, TokenizerConfig};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant, sleep};
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    info!("Starting rcat-voice pipeline...");

    let tts_engine: Arc<dyn TtsEngine> = rcat_voice::generator::build_from_env()?;
    let rounds = std::env::var("LLM_ROUNDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 10);

    let cancelled = Arc::new(AtomicBool::new(false));
    for round in 1..=rounds {
        info!("=== LLM round {}/{} ===", round, rounds);
        let was_cancelled = run_round(round, tts_engine.clone(), cancelled.clone()).await?;
        if was_cancelled {
            info!("Cancel requested; continuing to next round.");
        }
    }

    Ok(())
}

async fn run_round(
    round: usize,
    tts_engine: Arc<dyn TtsEngine>,
    cancelled: Arc<AtomicBool>,
) -> Result<bool> {
    let session_start_ts = Instant::now(); // t0 fallback：会话创建时间
    let llm_start = Arc::new(OnceLock::new());

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let (llm_delta_tx, llm_delta_rx) = mpsc::channel::<DeltaMsg>(256);
    let (text_seg_tx, text_seg_rx) = mpsc::channel::<TextSegment>(64);
    let buffer_engine = tts_engine.clone();
    let buffer_ms = Arc::new(move || buffer_engine.buffered_ms().unwrap_or(0));
    let cancel = CancellationToken::new();
    let cancel_tx_ctrlc = cancel_tx.clone();
    let cancel_engine = tts_engine.clone();
    let cancel_flag = cancelled.clone();
    let cancel_ctrlc = cancel.clone();
    let cancel_handle = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_flag.store(true, Ordering::Release);
            let _ = cancel_tx_ctrlc.send(true);
            let _ = cancel_engine.stop().await;
            cancel_ctrlc.cancel();
        }
    });

    let auto_cancel_handle = if round == 3 {
        let cancel_tx_auto = cancel_tx.clone();
        let cancel_engine = tts_engine.clone();
        let cancel_flag = cancelled.clone();
        let delay_ms = std::env::var("AUTO_CANCEL_DELAY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1500)
            .clamp(100, 30_000);
        let cancel_auto = cancel.clone();
        Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            cancel_flag.store(true, Ordering::Release);
            let _ = cancel_tx_auto.send(true);
            let _ = cancel_engine.stop().await;
            cancel_auto.cancel();
        }))
    } else {
        None
    };

    // Task 1: Player (Consumer of TextSegments)
    let pipeline = Pipeline::new(text_seg_rx, tts_engine.clone(), PipelineConfig::from_env())
        .with_cancel_token(cancel.clone());
    let pipeline_handle = tokio::spawn(pipeline.run());

    // Task 2: Tokenizer (Consumer of Deltas, Producer of TextSegments)
    let tokenizer = Tokenizer::new(
        llm_delta_rx,
        text_seg_tx,
        buffer_ms,
        session_start_ts,
        llm_start.clone(),
        round as u64,
        cancel.clone(),
        TokenizerConfig::from_env(),
    );
    let tokenizer_handle = tokio::spawn(tokenizer.run());

    // Task 3: Simulated LLM stream (Producer of Deltas)
    let simulated_text = std::env::var("LLM_SIM_TEXT")
        .unwrap_or_else(|_| "请用两三句话解释为什么首段短、后续段长更适合流式TTS。".to_string());
    let chunk_chars = std::env::var("LLM_SIM_CHUNK_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, 20);
    let delay_ms = std::env::var("LLM_SIM_DELAY_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(80)
        .clamp(10, 2000);
    let sim_handle = tokio::spawn(simulate_stream(
        simulated_text,
        chunk_chars,
        Duration::from_millis(delay_ms),
        llm_delta_tx.clone(),
        cancel_rx.clone(),
        llm_start.clone(),
    ));

    if let Err(e) = sim_handle.await? {
        info!("Sim stream finished with error (round {}): {:?}", round, e);
    } else {
        info!("Sim stream finished successfully (round {}).", round);
    }

    // Close delta input so tokenizer can drain and exit.
    let _ = llm_delta_tx.send(DeltaMsg::Eof).await;
    drop(llm_delta_tx);

    // Wait for others to drain if needed, or close app
    // In a real app we might wait for the player queue to drain.
    // Here we give a small buffer for the player to finish the last chunks if SSE is done.
    // If SSE broke, tokenizer will close delta_rx, which closes chunk_tx, which closes chunk_rx, pipeline finishes.
    tokenizer_handle.await?;
    pipeline_handle.await?;
    tts_engine.stop().await?;
    cancel_handle.abort();
    let _ = cancel_handle.await;
    if let Some(handle) = auto_cancel_handle {
        handle.abort();
        let _ = handle.await;
    }

    Ok(cancelled.swap(false, Ordering::AcqRel))
}

async fn simulate_stream(
    text: String,
    chunk_chars: usize,
    delay: Duration,
    llm_delta_tx: mpsc::Sender<DeltaMsg>,
    mut cancel_rx: watch::Receiver<bool>,
    llm_start: Arc<OnceLock<Instant>>,
) -> Result<()> {
    let _ = llm_start.get_or_init(Instant::now);
    let chars: Vec<char> = text.chars().collect();
    for chunk in chars.chunks(chunk_chars) {
        if *cancel_rx.borrow() {
            break;
        }
        let part: String = chunk.iter().collect();
        if llm_delta_tx.send(DeltaMsg::Delta(part)).await.is_err() {
            break;
        }
        tokio::select! {
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    break;
                }
            }
            _ = sleep(delay) => {}
        }
    }
    Ok(())
}
