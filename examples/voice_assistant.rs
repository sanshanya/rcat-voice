#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use anyhow::{Context, Result, bail};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use futures::StreamExt;
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use rcat_voice::{
    generator,
    streaming::{StreamCancelHandle, StreamSession},
};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use std::io::{self, Write};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use std::sync::Arc;
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use tokio::sync::{mpsc, watch};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use tokio::task::JoinHandle;

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic", feature = "turn-smart"))]
use rcat_voice::turn::SmartTurnDetector;

#[cfg(not(feature = "asr-sherpa"))]
fn main() {
    eprintln!("This example requires `--features asr-sherpa`");
}

#[cfg(all(feature = "asr-sherpa", not(feature = "asr-mic")))]
fn main() {
    eprintln!(
        "This example requires `--features asr-sherpa,asr-mic` (on Linux you also need ALSA dev packages)."
    );
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
#[tokio::main]
async fn main() -> Result<()> {
    use cpal::traits::{DeviceTrait, StreamTrait};
    use crossbeam_queue::ArrayQueue;
    use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(feature = "turn-smart")]
    use std::time::Instant;
    use tokio::time::{Duration, MissedTickBehavior};
    use tracing::{debug, info, warn};

    tracing_subscriber::fmt().with_env_filter("info").init();

    let base_url = Arc::new(
        std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string()),
    );
    let api_key = Arc::new(
        std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is required for voice_assistant example")?,
    );
    let model = Arc::new(
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string()),
    );

    let system_prompt = std::env::var("VOICE_SYSTEM_PROMPT").unwrap_or_else(|_| {
        "你是一个低延迟语音助手。回答要简洁、口语化；遇到不确定就直接说不确定。".to_string()
    });
    let history_max_messages = std::env::var("VOICE_MAX_HISTORY_MESSAGES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(4, 200);

    let feed_ms = std::env::var("ASR_FEED_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(20)
        .clamp(5, 200);

    // For the integrated demo we allow a larger ring buffer so we can tolerate brief stalls
    // (e.g. cancellation/join, model warmup).
    let ring_seconds = std::env::var("ASR_MIC_RING_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30)
        .clamp(1, 120);

    let drop_warn_samples = std::env::var("ASR_MIC_DROP_WARN_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(100)
        .clamp(1, 1_000_000);

    let device_hint = std::env::var("ASR_MIC_DEVICE").ok();

    let host = cpal::default_host();
    let device = select_input_device(&host, device_hint)?;
    let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

    let supported_config = device
        .default_input_config()
        .context("failed to get default input config")?;
    let sample_format = supported_config.sample_format();
    let mut config: cpal::StreamConfig = supported_config.into();

    if let Ok(value) = std::env::var("ASR_MIC_BUFFER_FRAMES") {
        if let Ok(frames) = value.parse::<u32>() {
            if frames > 0 {
                config.buffer_size = cpal::BufferSize::Fixed(frames);
            }
        }
    }

    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    if sample_rate == 0 || channels == 0 {
        bail!("Invalid input audio format: {}Hz/{}ch", sample_rate, channels);
    }

    let ring_capacity = (sample_rate as usize)
        .saturating_mul(channels as usize)
        .saturating_mul(ring_seconds as usize)
        .max(1024);

    let queue: Arc<ArrayQueue<i16>> = Arc::new(ArrayQueue::new(ring_capacity));
    let dropped = Arc::new(AtomicU64::new(0));

    let stream = build_cpal_stream(&device, &config, sample_format, queue.clone(), dropped.clone())
        .context("failed to build input stream")?;
    stream.play().context("failed to start input stream")?;

    info!(
        "voice_assistant: device={} format={:?} input={}Hz/{}ch feed_ms={} ring={}s cap_samples={}",
        device_name,
        sample_format,
        sample_rate,
        channels,
        feed_ms,
        ring_seconds,
        ring_capacity
    );
    info!("voice_assistant: press Ctrl+C to stop");

    let tts_engine = generator::build_from_env()?;
    let mut asr = rcat_voice::asr::SherpaAsrStream::from_env()?;

    let barge_in_min_speech_ms = std::env::var("BARGE_IN_MIN_SPEECH_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(450)
        .clamp(50, 10_000);

    #[cfg(feature = "turn-smart")]
    let turn_min_silence_ms = std::env::var("SMART_TURN_MIN_SILENCE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(400)
        .clamp(50, 10_000);

    #[cfg(feature = "turn-smart")]
    let turn_commit_ms = std::env::var("SMART_TURN_COMMIT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
        .clamp(0, 10_000);

    #[cfg(feature = "turn-smart")]
    let turn_force_end_ms = std::env::var("SMART_TURN_FORCE_END_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2000)
        .clamp(turn_min_silence_ms.saturating_add(turn_commit_ms), 60_000);

    #[cfg(feature = "turn-smart")]
    let turn_eval_interval_ms = std::env::var("SMART_TURN_EVAL_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(200)
        .clamp(50, 5000);

    let turn_silence_abs = std::env::var("SMART_TURN_SILENCE_ABS")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(200)
        .clamp(0, 20_000);

    let barge_in_silence_abs = std::env::var("BARGE_IN_SILENCE_ABS")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(turn_silence_abs)
        .clamp(0, 20_000);

    #[cfg(feature = "turn-smart")]
    let mut smart_turn: Option<SmartTurnDetector> = match std::env::var("SMART_TURN_MODEL") {
        Ok(value) if !value.trim().is_empty() => {
            let detector = SmartTurnDetector::from_env()?;
            info!(
                "voice_assistant: smart_turn enabled (threshold={:.2}, model={})",
                detector.threshold(),
                value
            );
            info!(
                "voice_assistant: smart_turn gate: min_silence_ms={} commit_ms={} force_end_ms={} eval_interval_ms={} silence_abs={}",
                turn_min_silence_ms,
                turn_commit_ms,
                turn_force_end_ms,
                turn_eval_interval_ms,
                turn_silence_abs,
            );
            Some(detector)
        }
        _ => None,
    };

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()?,
    )];
    trim_history(&mut messages, history_max_messages);

    let mut assistant: Option<RunningAssistant> = None;

    let mut turn_text = String::new();
    let mut speech_streak_ms: u64 = 0;

    #[cfg(feature = "turn-smart")]
    let mut smart_turn_threshold = smart_turn.as_ref().map(|d| d.threshold()).unwrap_or(0.5);
    #[cfg(feature = "turn-smart")]
    let mut turn_dirty = false;
    #[cfg(feature = "turn-smart")]
    let mut trailing_silence_ms: u64 = 0;
    #[cfg(feature = "turn-smart")]
    let mut endpoint_armed = false;
    #[cfg(feature = "turn-smart")]
    let mut last_eval_at: Option<Instant> = None;
    #[cfg(feature = "turn-smart")]
    let mut smart_turn_infer: Option<tokio::task::JoinHandle<Result<f32>>> = None;

    let frames = ((sample_rate as u64 * feed_ms) / 1000).max(1) as usize;
    let chunk_samples = frames
        .saturating_mul(channels as usize)
        .max(channels as usize);
    let mut chunk = Vec::<i16>::with_capacity(chunk_samples);

    let mut poll = tokio::time::interval(Duration::from_millis(5));
    poll.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut drop_tick = tokio::time::interval(Duration::from_secs(1));
    drop_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => break,
            _ = drop_tick.tick() => {
                let n = dropped.swap(0, Ordering::AcqRel);
                if n > 0 {
                    if n < drop_warn_samples {
                        debug!("voice_assistant: dropped {} samples (ring buffer full)", n);
                    } else {
                        warn!("voice_assistant: dropped {} samples (ring buffer full)", n);
                    }
                }
            }
            _ = poll.tick() => {
                if assistant.as_ref().is_some_and(|running| running.handle.is_finished()) {
                    let finished = assistant.take().expect("assistant");
                    match finished.handle.await {
                        Ok(Ok(result)) => {
                            if !result.cancelled && !result.text.trim().is_empty() {
                                messages.push(ChatCompletionRequestMessage::Assistant(
                                    ChatCompletionRequestAssistantMessageArgs::default()
                                        .content(result.text)
                                        .build()?,
                                ));
                                trim_history(&mut messages, history_max_messages);
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("voice_assistant: assistant task failed: {e}");
                        }
                        Err(e) => {
                            warn!("voice_assistant: assistant join failed: {e}");
                        }
                    }
                }

                while chunk.len() < chunk_samples {
                    let Some(sample) = queue.pop() else {
                        break;
                    };
                    chunk.push(sample);
                }

                if chunk.len() >= chunk_samples {
                    let is_silence = is_silence_chunk(&chunk, barge_in_silence_abs);
                    if is_silence {
                        speech_streak_ms = 0;
                    } else {
                        speech_streak_ms = speech_streak_ms.saturating_add(feed_ms);
                    }

                    if let Some(running) = assistant.as_mut() {
                        if !running.cancel_requested && speech_streak_ms >= barge_in_min_speech_ms {
                            warn!(
                                "voice_assistant: barge-in detected (speech_ms={} >= {}), cancelling assistant",
                                speech_streak_ms, barge_in_min_speech_ms
                            );
                            speech_streak_ms = 0;
                            running.cancel_requested = true;
                            let _ = running.cancel_tx.send(true);
                            let cancel_handle = running.cancel_handle.clone();
                            tokio::spawn(async move {
                                let _ = cancel_handle.cancel().await;
                            });
                        }
                    }

                    #[cfg(feature = "turn-smart")]
                    if let Some(detector) = smart_turn.as_mut() {
                        let is_silence = is_silence_chunk(&chunk, turn_silence_abs);
                        if is_silence {
                            trailing_silence_ms = trailing_silence_ms.saturating_add(feed_ms);
                        } else {
                            trailing_silence_ms = 0;
                            endpoint_armed = false;
                            if let Some(handle) = smart_turn_infer.take() {
                                handle.abort();
                            }
                            last_eval_at = None;
                        }
                        detector.push_pcm_i16(&chunk, sample_rate, channels)?;
                    }

                    asr.write_pcm_i16(&chunk, sample_rate, channels).await?;
                    chunk.clear();
                }

                while let Some(seg) = asr.try_read() {
                    println!("[{:.2}-{:.2}] {}", seg.start, seg.end, seg.text);
                    if !turn_text.is_empty() {
                        turn_text.push(' ');
                    }
                    turn_text.push_str(&seg.text);

                    #[cfg(feature = "turn-smart")]
                    {
                        turn_dirty = true;
                    }

                    let should_fallback_turn = {
                        #[cfg(feature = "turn-smart")]
                        {
                            smart_turn.is_none()
                        }
                        #[cfg(not(feature = "turn-smart"))]
                        {
                            true
                        }
                    };

                    if should_fallback_turn {
                        // No smart-turn: treat each VAD segment as a complete user turn.
                        let user_text = turn_text.trim().to_string();
                        turn_text.clear();
                        if !user_text.is_empty() {
                            if let Some(running) = assistant.take() {
                                stop_running(running).await?;
                            }
                            messages.push(ChatCompletionRequestMessage::User(
                                ChatCompletionRequestUserMessageArgs::default()
                                    .content(user_text.clone())
                                    .build()?,
                            ));
                            trim_history(&mut messages, history_max_messages);
                            println!("USER: {user_text}");
                            assistant = Some(
                                start_assistant(
                                    tts_engine.clone(),
                                    base_url.clone(),
                                    api_key.clone(),
                                    model.clone(),
                                    messages.clone(),
                                )
                                .await?,
                            );
                        }
                    }
                }

                #[cfg(feature = "turn-smart")]
                if smart_turn_infer
                    .as_ref()
                    .is_some_and(|handle| handle.is_finished())
                {
                    let handle = smart_turn_infer.take().expect("smart_turn_infer");
                    match handle.await {
                        Ok(Ok(prob)) => {
                            smart_turn_threshold = smart_turn_threshold.clamp(0.0, 1.0);
                            endpoint_armed = prob >= smart_turn_threshold;
                            println!(
                                "smart_turn: p={:.3} endpoint={} silence_ms={}",
                                prob, endpoint_armed, trailing_silence_ms
                            );
                            turn_dirty = false;
                        }
                        Ok(Err(e)) => {
                            warn!("smart_turn: predict failed: {e}");
                            endpoint_armed = false;
                        }
                        Err(e) => {
                            warn!("smart_turn: predict task failed: {e}");
                            endpoint_armed = false;
                        }
                    }
                }

                #[cfg(feature = "turn-smart")]
                if let Some(detector) = smart_turn.as_mut() {
                    if turn_text.trim().is_empty() {
                        endpoint_armed = false;
                        turn_dirty = false;
                        if let Some(handle) = smart_turn_infer.take() {
                            handle.abort();
                        }
                    } else if trailing_silence_ms >= turn_force_end_ms {
                        let user_text = turn_text.trim().to_string();
                        println!("TURN_END: {}", user_text);
                        turn_text.clear();
                        detector.reset();
                        trailing_silence_ms = 0;
                        endpoint_armed = false;
                        turn_dirty = false;
                        last_eval_at = None;
                        if let Some(handle) = smart_turn_infer.take() {
                            handle.abort();
                        }

                        if let Some(running) = assistant.take() {
                            stop_running(running).await?;
                        }
                        messages.push(ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(user_text.clone())
                                .build()?,
                        ));
                        trim_history(&mut messages, history_max_messages);
                        println!("USER: {user_text}");
                        assistant = Some(start_assistant(
                            tts_engine.clone(),
                            base_url.clone(),
                            api_key.clone(),
                            model.clone(),
                            messages.clone(),
                        ).await?);
                    } else if trailing_silence_ms >= turn_min_silence_ms {
                        let now = Instant::now();
                        let should_eval = turn_dirty
                            || last_eval_at
                                .map(|t| now.duration_since(t).as_millis() as u64 >= turn_eval_interval_ms)
                                .unwrap_or(true);
                        if should_eval && smart_turn_infer.is_none() {
                            let audio = detector.snapshot_audio();
                            let model = detector.model();
                            smart_turn_infer = Some(tokio::task::spawn_blocking(move || {
                                model.predict_probability(&audio)
                            }));
                            last_eval_at = Some(now);
                        }

                        if endpoint_armed
                            && trailing_silence_ms
                                >= turn_min_silence_ms.saturating_add(turn_commit_ms)
                        {
                            let user_text = turn_text.trim().to_string();
                            println!("TURN_END: {}", user_text);
                            turn_text.clear();
                            detector.reset();
                            trailing_silence_ms = 0;
                            endpoint_armed = false;
                            turn_dirty = false;
                            last_eval_at = None;
                            if let Some(handle) = smart_turn_infer.take() {
                                handle.abort();
                            }

                            if let Some(running) = assistant.take() {
                                stop_running(running).await?;
                            }
                            messages.push(ChatCompletionRequestMessage::User(
                                ChatCompletionRequestUserMessageArgs::default()
                                    .content(user_text.clone())
                                    .build()?,
                            ));
                            trim_history(&mut messages, history_max_messages);
                            println!("USER: {user_text}");
                            assistant = Some(start_assistant(
                                tts_engine.clone(),
                                base_url.clone(),
                                api_key.clone(),
                                model.clone(),
                                messages.clone(),
                            ).await?);
                        }
                    }
                }
            }
        }
    }

    drop(stream);
    if let Some(running) = assistant.take() {
        stop_running(running).await?;
    }
    asr.finish().await?;
    Ok(())
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
struct LlmResult {
    text: String,
    cancelled: bool,
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
struct RunningAssistant {
    cancel_tx: watch::Sender<bool>,
    cancel_handle: StreamCancelHandle,
    handle: JoinHandle<Result<LlmResult>>,
    cancel_requested: bool,
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
async fn stop_running(mut running: RunningAssistant) -> Result<()> {
    if !running.cancel_requested {
        running.cancel_requested = true;
        let _ = running.cancel_tx.send(true);
        let _ = running.cancel_handle.cancel().await;
    }
    let _ = running.handle.await;
    Ok(())
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
async fn start_assistant(
    tts_engine: Arc<dyn rcat_voice::generator::TtsEngine>,
    base_url: Arc<String>,
    api_key: Arc<String>,
    model: Arc<String>,
    messages: Vec<ChatCompletionRequestMessage>,
) -> Result<RunningAssistant> {
    let session = StreamSession::from_env(tts_engine);
    let cancel_handle = session.cancel_handle();
    let control = session.control();
    control.mark_llm_start();
    let delta_tx = control.sender();
    drop(control);

    let (cancel_tx, cancel_rx) = watch::channel(false);
    print!("ASSISTANT: ");
    io::stdout().flush().ok();

    let llm_cancel = cancel_rx.clone();
    let drain_cancel = cancel_rx.clone();
    let handle = tokio::spawn(async move {
        let result = stream_chat(
            base_url,
            api_key,
            model,
            messages,
            delta_tx,
            llm_cancel,
        )
        .await?;
        session.finish_or_cancel(drain_cancel).await?;
        Ok(result)
    });

    Ok(RunningAssistant {
        cancel_tx,
        cancel_handle,
        handle,
        cancel_requested: false,
    })
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
async fn stream_chat(
    base_url: Arc<String>,
    api_key: Arc<String>,
    model: Arc<String>,
    messages: Vec<ChatCompletionRequestMessage>,
    delta_tx: mpsc::Sender<String>,
    mut cancel: watch::Receiver<bool>,
) -> Result<LlmResult> {
    let config = OpenAIConfig::new()
        .with_api_key((*api_key).clone())
        .with_api_base((*base_url).clone());
    let client = Client::with_config(config);

    let request = CreateChatCompletionRequestArgs::default()
        .model((*model).clone())
        .messages(messages)
        .stream(true)
        .build()?;
    let mut stream = client.chat().create_stream(request).await?;

    let mut text = String::new();
    let mut cancelled = false;

    loop {
        tokio::select! {
            res = cancel.changed() => {
                if res.is_ok() && *cancel.borrow() {
                    cancelled = true;
                    break;
                }
            }
            maybe_chunk = stream.next() => {
                match maybe_chunk {
                    Some(Ok(response)) => {
                        for choice in response.choices {
                            if let Some(content) = choice.delta.content {
                                text.push_str(&content);
                                print!("{content}");
                                io::stdout().flush().ok();
                                if delta_tx.send(content).await.is_err() {
                                    cancelled = true;
                                    break;
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        cancelled = true;
                        eprintln!("\nvoice_assistant: LLM stream error: {e}");
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    println!();
    Ok(LlmResult { text, cancelled })
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
fn trim_history(messages: &mut Vec<ChatCompletionRequestMessage>, max_messages: usize) {
    if max_messages == 0 {
        return;
    }
    if messages.len() <= max_messages {
        return;
    }
    if messages.len() <= 1 {
        return;
    }
    let keep = max_messages.saturating_sub(1).max(1);
    let extra = messages.len().saturating_sub(1).saturating_sub(keep);
    if extra > 0 {
        messages.drain(1..=extra);
    }
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
fn select_input_device(host: &cpal::Host, hint: Option<String>) -> Result<cpal::Device> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let mut devices: Vec<(String, cpal::Device)> = Vec::new();
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
        devices.push((name, device));
    }

    if let Some(hint) = hint
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        let needle = hint.to_lowercase();
        if let Some(index) = devices
            .iter()
            .position(|(name, _)| name.to_lowercase().contains(&needle))
        {
            return Ok(devices.swap_remove(index).1);
        }
        let available = devices
            .iter()
            .map(|(name, _)| format!("- {name}"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!(
            "ASR_MIC_DEVICE={hint} did not match any input device. Available:\n{available}"
        );
    }

    host.default_input_device()
        .context("no default input device (set ASR_MIC_DEVICE to select one)")
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
fn build_cpal_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    queue: Arc<crossbeam_queue::ArrayQueue<i16>>,
    dropped: Arc<std::sync::atomic::AtomicU64>,
) -> Result<cpal::Stream> {
    use cpal::traits::DeviceTrait;

    let err_fn = |err| tracing::error!("cpal input stream error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => {
            let queue = queue.clone();
            let dropped = dropped.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        let scaled = (sample * 32767.0).round();
                        let clamped = scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                        if queue.push(clamped).is_err() {
                            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let queue = queue.clone();
            let dropped = dropped.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        if queue.push(sample).is_err() {
                            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            let queue = queue.clone();
            let dropped = dropped.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    for &sample in data {
                        let sample = sample as i32 - 32768;
                        let sample = sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        if queue.push(sample).is_err() {
                            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => bail!("Unsupported input sample format: {other:?}"),
    }
}

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
fn is_silence_chunk(pcm: &[i16], abs_threshold: u16) -> bool {
    if pcm.is_empty() {
        return true;
    }
    let threshold = abs_threshold as i64;
    let mut sum: i64 = 0;
    for &sample in pcm {
        sum += (sample as i32).abs() as i64;
    }
    let avg = sum / pcm.len() as i64;
    avg <= threshold
}
