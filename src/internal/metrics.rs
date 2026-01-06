use crate::generator::TtsMetrics;
use crate::tokenizer::Segment;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep_until};
use tracing::info;

pub(crate) fn log_segment_intro(segment: &Segment, intro_logged: &mut bool) {
    if *intro_logged {
        return;
    }
    *intro_logged = true;

    info!("=== 指标时间线 ===");

    let Some(t1) = segment.first_token_ts else {
        return;
    };

    let llm_delay = t1.saturating_duration_since(segment.llm_start_ts);
    info!("LLM首字时延: {}", fmt_duration(llm_delay));

    let chunker_delay = segment.segment_sent_ts.saturating_duration_since(t1);
    info!("分段器延迟: {}", fmt_duration(chunker_delay));
}

pub(crate) fn log_playback_metrics(
    segment: &Segment,
    metrics: TtsMetrics,
    last_play_done_ts: &Arc<StdMutex<Option<tokio::time::Instant>>>,
    play_done_tasks: &mut JoinSet<()>,
) {
    let is_first_chunk = segment.first_token_ts.is_some();
    let chunk_chars = segment.text.chars().count();
    let first_audio_ts = metrics.first_audio_ts.unwrap_or(metrics.start_ts);

    if is_first_chunk {
        let first_audio_delay = first_audio_ts.saturating_duration_since(segment.segment_sent_ts);
        info!("首播时延: {}", fmt_duration(first_audio_delay));
    }

    if let Some(play_done_rx) = metrics.play_done_rx {
        let llm_start_ts = segment.llm_start_ts;
        let last_done = last_play_done_ts.clone();
        play_done_tasks.spawn(async move {
            if let Ok(ts) = play_done_rx.await {
                let play_done_abs = ts.saturating_duration_since(llm_start_ts);
                info!("音频块播放完成: {} | {} 字符", fmt_duration(play_done_abs), chunk_chars);
                let mut guard = last_done.lock().expect("playback done lock poisoned");
                if guard.map_or(true, |prev| ts > prev) {
                    *guard = Some(ts);
                }
            }
        });
        return;
    }

    let ts = metrics.play_done_ts;
    let play_done_abs = ts.saturating_duration_since(segment.llm_start_ts);
    info!("音频块播放完成: {} | {} 字符", fmt_duration(play_done_abs), chunk_chars);
    if let Ok(mut guard) = last_play_done_ts.lock() {
        if guard.map_or(true, |prev| ts > prev) {
            *guard = Some(ts);
        }
    }
}

pub(crate) async fn await_playback_drain(
    play_done_tasks: &mut JoinSet<()>,
    last_play_done_ts: &Arc<StdMutex<Option<tokio::time::Instant>>>,
) {
    while play_done_tasks.join_next().await.is_some() {}
    let done_ts = last_play_done_ts
        .lock()
        .map(|guard| *guard)
        .unwrap_or(None);
    let Some(done_ts) = done_ts else {
        return;
    };
    let now = Instant::now();
    if done_ts > now {
        sleep_until(done_ts).await;
    }
}

fn fmt_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2}s")
    } else {
        format!("{}ms", duration.as_millis())
    }
}
