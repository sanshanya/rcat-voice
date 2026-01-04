#[cfg(feature = "asr-sherpa")]
use anyhow::{Context, Result, bail};

#[cfg(not(feature = "asr-sherpa"))]
fn main() {
    eprintln!("This example requires `--features asr-sherpa`");
}

#[cfg(feature = "asr-sherpa")]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let metrics = std::env::var("ASR_METRICS")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

    let wav = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("ASR_WAV").ok())
        .context("usage: cargo run --example asr_file --features asr-sherpa -- <path.wav> (or set ASR_WAV)")?;

    let (pcm, sample_rate, channels) = read_wav_i16(&wav)?;
    let audio_seconds = pcm.len() as f64 / (sample_rate as f64 * channels as f64);

    let feed_ms = std::env::var("ASR_FEED_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(20)
        .min(2000);

    let init_start = std::time::Instant::now();
    let mut stream = rcat_voice::asr::SherpaAsrStream::from_env()?;
    let init_ms = init_start.elapsed().as_millis() as u64;

    if metrics {
        println!(
            "asr_file: init_ms={} audio_s={:.2} feed_ms={} sample_rate={} channels={}",
            init_ms, audio_seconds, feed_ms, sample_rate, channels
        );
    }

    let chunk_samples = if feed_ms == 0 {
        pcm.len().max(1)
    } else {
        let frames = (sample_rate as u64 * feed_ms / 1000).max(1) as usize;
        frames
            .saturating_mul(channels as usize)
            .max(channels as usize)
    };

    let start_ts = std::time::Instant::now();
    let mut transcript = String::new();
    let mut seg_count = 0usize;
    let mut first_seg_at_ms: Option<u64> = None;
    let mut max_lag_ms: f64 = 0.0;

    for chunk in pcm.chunks(chunk_samples) {
        stream
            .write_pcm_i16(chunk, sample_rate, channels)
            .await?;

        // Poll available results so we can estimate streaming latency.
        loop {
            let Some(seg) = stream.try_read() else {
                break;
            };
            seg_count += 1;
            if !transcript.is_empty() {
                transcript.push(' ');
            }
            transcript.push_str(&seg.text);

            let now = std::time::Instant::now();
            let elapsed_s = now.duration_since(start_ts).as_secs_f64();
            let lag_ms = (elapsed_s - seg.end as f64) * 1000.0;
            if first_seg_at_ms.is_none() {
                first_seg_at_ms = Some(now.duration_since(start_ts).as_millis() as u64);
            }
            if lag_ms > max_lag_ms {
                max_lag_ms = lag_ms;
            }

            if metrics {
                println!(
                    "[{:.2}-{:.2}] {} (lag_ms={:.0})",
                    seg.start, seg.end, seg.text, lag_ms
                );
            } else {
                println!("[{:.2}-{:.2}] {}", seg.start, seg.end, seg.text);
            }
        }

        if feed_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(feed_ms)).await;
        }
    }

    stream.finish().await?;
    while let Some(seg) = stream.read().await {
        seg_count += 1;
        if !transcript.is_empty() {
            transcript.push(' ');
        }
        transcript.push_str(&seg.text);

        let now = std::time::Instant::now();
        let elapsed_s = now.duration_since(start_ts).as_secs_f64();
        let lag_ms = (elapsed_s - seg.end as f64) * 1000.0;
        if first_seg_at_ms.is_none() {
            first_seg_at_ms = Some(now.duration_since(start_ts).as_millis() as u64);
        }
        if lag_ms > max_lag_ms {
            max_lag_ms = lag_ms;
        }

        if metrics {
            println!(
                "[{:.2}-{:.2}] {} (lag_ms={:.0})",
                seg.start, seg.end, seg.text, lag_ms
            );
        } else {
            println!("[{:.2}-{:.2}] {}", seg.start, seg.end, seg.text);
        }
    }

    if metrics {
        let wall_ms = start_ts.elapsed().as_millis() as u64;
        let rtf = if audio_seconds > 0.0 {
            (wall_ms as f64 / 1000.0) / audio_seconds
        } else {
            0.0
        };
        println!(
            "asr_file: segments={} first_ms={:?} wall_ms={} rtf={:.3} max_lag_ms={:.0}",
            seg_count, first_seg_at_ms, wall_ms, rtf, max_lag_ms
        );

        if let Some((ref_text, ref_name)) = load_ref_text()? {
            let cer = cer_percent(&ref_text, &transcript);
            println!("asr_file: cer_percent={:.2} ref={} hyp_len={}", cer, ref_name, transcript.len());
        }
    }

    Ok(())
}

#[cfg(feature = "asr-sherpa")]
fn read_wav_i16(path: &str) -> Result<(Vec<i16>, u32, u16)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav: {path}"))?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        bail!("only 16-bit PCM wav is supported (got {:?}/{})", spec.sample_format, spec.bits_per_sample);
    }
    if spec.channels == 0 {
        bail!("wav channels must be >= 1");
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to read wav samples")?;

    Ok((samples, spec.sample_rate, spec.channels))
}

#[cfg(feature = "asr-sherpa")]
fn load_ref_text() -> Result<Option<(String, String)>> {
    if let Ok(path) = std::env::var("ASR_REF_FILE") {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read ASR_REF_FILE: {path}"))?;
        return Ok(Some((text, path)));
    }
    if let Ok(text) = std::env::var("ASR_REF_TEXT") {
        return Ok(Some((text, "ASR_REF_TEXT".to_string())));
    }
    Ok(None)
}

#[cfg(feature = "asr-sherpa")]
fn cer_percent(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_text(reference);
    let hypothesis = normalize_text(hypothesis);
    if reference.is_empty() {
        return 0.0;
    }
    let dist = edit_distance(&reference, &hypothesis);
    dist as f64 * 100.0 / reference.len() as f64
}

#[cfg(feature = "asr-sherpa")]
fn normalize_text(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| *c != '\u{feff}' && *c != '\u{200b}')
        .filter(|c| !c.is_whitespace())
        .filter(|c| {
            !matches!(
                c,
                '，' | '。' | '！' | '？' | '；' | '：' | '、' | ',' | '.' | '!' | '?' | ';' | ':'
                    | '"' | '\'' | '“' | '”' | '‘' | '’' | '(' | ')' | '（' | '）' | '《'
                    | '》' | '【' | '】' | '[' | ']' | '{' | '}' | '<' | '>' | '…' | '—'
                    | '-' | '·'
            )
        })
        .collect()
}

#[cfg(feature = "asr-sherpa")]
fn edit_distance(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
        }
        prev.copy_from_slice(&curr);
    }

    prev[b.len()]
}
