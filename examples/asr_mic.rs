#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use anyhow::{Context, Result, bail};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use rcat_voice::metrics::{MetricsSink, TracingMetricsSink};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic", feature = "turn-smart"))]
use rcat_voice::turn::{SmartTurnConfig, SmartTurnDetector};
#[cfg(all(feature = "asr-sherpa", feature = "asr-mic"))]
use std::sync::Arc;

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
    use tracing::{info, warn};
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let feed_ms = std::env::var("ASR_FEED_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(20)
        .clamp(5, 200);

    let ring_seconds = std::env::var("ASR_MIC_RING_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30)
        .clamp(1, 120);

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
        bail!(
            "Invalid input audio format: {}Hz/{}ch",
            sample_rate,
            channels
        );
    }

    let ring_capacity = (sample_rate as usize)
        .saturating_mul(channels as usize)
        .saturating_mul(ring_seconds as usize)
        .max(1024);

    let queue: Arc<ArrayQueue<i16>> = Arc::new(ArrayQueue::new(ring_capacity));
    let dropped = Arc::new(AtomicU64::new(0));

    let stream = build_cpal_stream(
        &device,
        &config,
        sample_format,
        queue.clone(),
        dropped.clone(),
    )
    .context("failed to build input stream")?;
    stream.play().context("failed to start input stream")?;

    info!(
        "asr_mic: device={} format={:?} input={}Hz/{}ch feed_ms={} ring={}s cap_samples={}",
        device_name, sample_format, sample_rate, channels, feed_ms, ring_seconds, ring_capacity
    );
    info!("asr_mic: press Ctrl+C to stop");

    let metrics: Arc<dyn MetricsSink> = Arc::new(TracingMetricsSink::from_env());
    let mut asr = rcat_voice::asr::SherpaAsrStream::from_env_with_metrics(metrics)?;
    let mut turn_text = String::new();

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

    #[cfg(feature = "turn-smart")]
    let turn_silence_abs = std::env::var("SMART_TURN_SILENCE_ABS")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(200)
        .clamp(0, 20_000);

    #[cfg(feature = "turn-smart")]
    let mut smart_turn: Option<SmartTurnDetector> = {
        let env_has = |key: &str| std::env::var(key).ok().is_some_and(|v| !v.trim().is_empty());
        let smart_turn_disabled = std::env::var("SMART_TURN_DISABLE").ok().is_some_and(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        });
        let smart_turn_enabled =
            !smart_turn_disabled && (env_has("SMART_TURN_MODEL") || env_has("RCAT_MODELS_DIR"));
        if !smart_turn_enabled {
            None
        } else {
            match SmartTurnConfig::from_env().and_then(SmartTurnDetector::new) {
                Ok(detector) => {
                    info!(
                        "asr_mic: smart_turn enabled (threshold={:.2})",
                        detector.threshold()
                    );
                    info!(
                        "asr_mic: smart_turn gate: min_silence_ms={} commit_ms={} force_end_ms={} eval_interval_ms={} silence_abs={}",
                        turn_min_silence_ms,
                        turn_commit_ms,
                        turn_force_end_ms,
                        turn_eval_interval_ms,
                        turn_silence_abs,
                    );
                    Some(detector)
                }
                Err(err) => {
                    warn!("asr_mic: smart_turn disabled: {err}");
                    None
                }
            }
        }
    };

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
                    warn!("asr_mic: dropped {} samples (ring buffer full)", n);
                }
            }
            _ = poll.tick() => {
                while chunk.len() < chunk_samples {
                    let Some(sample) = queue.pop() else {
                        break;
                    };
                    chunk.push(sample);
                }

                if chunk.len() >= chunk_samples {
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
                        println!("TURN_END: {}", turn_text);
                        turn_text.clear();
                        detector.reset();
                        trailing_silence_ms = 0;
                        endpoint_armed = false;
                        turn_dirty = false;
                        last_eval_at = None;
                        if let Some(handle) = smart_turn_infer.take() {
                            handle.abort();
                        }
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
                            println!("TURN_END: {}", turn_text);
                            turn_text.clear();
                            detector.reset();
                            trailing_silence_ms = 0;
                            endpoint_armed = false;
                            turn_dirty = false;
                            last_eval_at = None;
                            if let Some(handle) = smart_turn_infer.take() {
                                handle.abort();
                            }
                        }
                    }
                }
            }
        }
    }

    drop(stream);
    asr.finish().await?;
    while let Some(seg) = asr.read().await {
        println!("[{:.2}-{:.2}] {}", seg.start, seg.end, seg.text);
        if !turn_text.is_empty() {
            turn_text.push(' ');
        }
        turn_text.push_str(&seg.text);
    }

    if !turn_text.trim().is_empty() {
        println!("TURN_END: {}", turn_text);
    }

    Ok(())
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

    if let Some(hint) = hint.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) {
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
        bail!("ASR_MIC_DEVICE={hint} did not match any input device. Available:\n{available}");
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

#[cfg(all(feature = "asr-sherpa", feature = "asr-mic", feature = "turn-smart"))]
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
