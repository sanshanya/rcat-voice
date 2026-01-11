//! Offline Speaker Diarization Example
//!
//! Demonstrates offline speaker diarization using sherpa-rs Diarize API.
//! Outputs "who spoke when" for a multi-speaker audio file.
//!
//! **Usage**:
//! ```bash
//! cargo run --example diarize_offline --features asr-sherpa -- \
//!     --seg-model path/to/segmentation_model.onnx \
//!     --emb-model path/to/embedding_model.onnx \
//!     --audio multi_speaker.wav
//! ```
//!
//! **Model Download**:
//! ```bash
//! # Segmentation model (pyannote)
//! wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
//! tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
//! # Use: sherpa-onnx-pyannote-segmentation-3-0/model.onnx
//! # Or for CPU: sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx (smaller, faster on CPU)
//!
//! # Embedding model (3dspeaker)
//! wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
//! ```
//!
//! **Low-Latency Notes**:
//! The `Diarize::compute()` API is designed for offline/batch processing of complete audio.
//! For online/streaming diarization, use the "utterance-level speaker labeling" approach:
//! 1. VAD produces utterances (start/end + samples)
//! 2. For each utterance, compute embedding (using EmbeddingExtractor from speaker_id_gate)
//! 3. Compare with known speaker prototypes (using EmbeddingManager.search())
//! 4. Assign or create new speaker label
//!
//! See `sherpa_rs::embedding_manager::EmbeddingManager` for online speaker tracking.

#[cfg(feature = "asr-sherpa")]
use anyhow::{Context, Result, bail};

#[cfg(not(feature = "asr-sherpa"))]
fn main() {
    eprintln!("This example requires `--features asr-sherpa`");
}

#[cfg(feature = "asr-sherpa")]
fn main() -> Result<()> {
    use sherpa_rs::diarize::{Diarize, DiarizeConfig};

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    // ─────────────────────────────────────────────────────────────────────────
    // CLI Parsing
    // ─────────────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let cli = parse_args(&args)?;

    println!("Offline Speaker Diarization");
    println!("============================");
    println!("Segmentation model: {}", cli.seg_model.display());
    println!("Embedding model: {}", cli.emb_model.display());
    println!("Audio file: {}", cli.audio_path.display());
    println!("Num clusters: {:?}", cli.num_clusters);
    println!("Threshold: {:?}", cli.threshold);
    println!("Min duration on: {:.2}s", cli.min_duration_on);
    println!("Min duration off: {:.2}s", cli.min_duration_off);
    println!("Provider: {}", cli.provider);
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Load Audio
    // ─────────────────────────────────────────────────────────────────────────
    let load_start = std::time::Instant::now();
    let (samples, sample_rate) = read_wav_mono_16k(&cli.audio_path)?;
    let audio_duration_s = samples.len() as f32 / sample_rate as f32;
    let load_ms = load_start.elapsed().as_millis();
    println!("[Audio] Loaded {:.2}s ({} samples) in {}ms", audio_duration_s, samples.len(), load_ms);

    // ─────────────────────────────────────────────────────────────────────────
    // Initialize Diarizer
    // ─────────────────────────────────────────────────────────────────────────
    let config = DiarizeConfig {
        num_clusters: cli.num_clusters,
        threshold: cli.threshold,
        min_duration_on: Some(cli.min_duration_on),
        min_duration_off: Some(cli.min_duration_off),
        provider: Some(cli.provider.clone()),
        debug: cli.debug,
    };

    let init_start = std::time::Instant::now();
    let seg_model_str = cli.seg_model.to_string_lossy().to_string();
    let emb_model_str = cli.emb_model.to_string_lossy().to_string();
    let mut diarizer = Diarize::new(&seg_model_str, &emb_model_str, config)
        .map_err(|e| anyhow::anyhow!("Failed to create Diarize instance: {}", e))?;
    let init_ms = init_start.elapsed().as_millis();
    println!("[Diarizer] Initialized in {}ms", init_ms);

    // ─────────────────────────────────────────────────────────────────────────
    // Run Diarization
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[Processing]");
    let compute_start = std::time::Instant::now();

    // Progress callback: print to stderr so it doesn't interfere with JSON output
    let progress_callback = |processed: i32, total: i32| -> i32 {
        if total > 0 {
            let percent = (processed as f32 / total as f32) * 100.0;
            eprint!("\rProgress: {:>5.1}% ({}/{} chunks)", percent, processed, total);
        }
        0 // Return 0 to continue processing
    };

    let segments = diarizer
        .compute(samples, Some(Box::new(progress_callback)))
        .map_err(|e| anyhow::anyhow!("Diarization failed: {}", e))?;

    let compute_ms = compute_start.elapsed().as_millis();
    eprintln!(); // Newline after progress
    let rtf = compute_ms as f32 / 1000.0 / audio_duration_s;
    println!("[Completed] {} segments in {}ms (RTF={:.3})", segments.len(), compute_ms, rtf);

    // ─────────────────────────────────────────────────────────────────────────
    // Output Results
    // ─────────────────────────────────────────────────────────────────────────
    if cli.json {
        // JSON output for programmatic use
        let json_segments: Vec<serde_json::Value> = segments
            .iter()
            .map(|seg| {
                serde_json::json!({
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "duration": seg.end - seg.start,
                })
            })
            .collect();

        let output = serde_json::json!({
            "audio_file": cli.audio_path.to_string_lossy(),
            "audio_duration_s": audio_duration_s,
            "processing_ms": compute_ms,
            "rtf": rtf,
            "num_speakers": count_unique_speakers(&segments),
            "segments": json_segments,
        });

        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Human-readable output
        println!("\n[Segments]");
        println!("{:>10} {:>10} {:>8} {:>10}", "START", "END", "SPEAKER", "DURATION");
        println!("{:-<42}", "");

        for seg in &segments {
            let duration = seg.end - seg.start;
            println!(
                "{:>10.2}s {:>9.2}s {:>8} {:>9.2}s",
                seg.start, seg.end, format!("S{}", seg.speaker), duration
            );
        }

        println!("{:-<42}", "");
        println!(
            "Total: {} segments, {} unique speakers",
            segments.len(),
            count_unique_speakers(&segments)
        );

        // Per-speaker statistics
        println!("\n[Speaker Statistics]");
        let stats = compute_speaker_stats(&segments);
        for (speaker, (total_duration, segment_count)) in &stats {
            let percent = (total_duration / audio_duration_s) * 100.0;
            println!(
                "  Speaker S{}: {:.2}s ({:.1}%), {} segments",
                speaker, total_duration, percent, segment_count
            );
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Loading: Read WAV, enforce mono + 16kHz
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "asr-sherpa")]
fn read_wav_mono_16k(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    // Enforce 16kHz sample rate (sherpa-onnx requirement)
    if sample_rate != 16000 {
        bail!(
            "WAV file must be 16kHz, got {}Hz. Please resample:\n\
             ffmpeg -i {} -ar 16000 -ac 1 output_16k.wav",
            sample_rate,
            path.display()
        );
    }

    // Read samples based on format
    let samples_i16: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample == 16 {
                reader.samples::<i16>().collect::<std::result::Result<Vec<_>, _>>()?
            } else if spec.bits_per_sample == 32 {
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| (v >> 16) as i16))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                bail!("Unsupported bits_per_sample: {}", spec.bits_per_sample);
            }
        }
        hound::SampleFormat::Float => {
            reader
                .samples::<f32>()
                .map(|s| s.map(|v| (v * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    // Downmix to mono if stereo
    let mono_samples: Vec<i16> = if channels > 1 {
        samples_i16
            .chunks(channels)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as i16
            })
            .collect()
    } else {
        samples_i16
    };

    // Convert i16 to f32 (normalized to [-1.0, 1.0])
    let samples_f32: Vec<f32> = mono_samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect();

    Ok((samples_f32, sample_rate))
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "asr-sherpa")]
fn count_unique_speakers(segments: &[sherpa_rs::diarize::Segment]) -> usize {
    let mut speakers = std::collections::HashSet::new();
    for seg in segments {
        speakers.insert(seg.speaker);
    }
    speakers.len()
}

#[cfg(feature = "asr-sherpa")]
fn compute_speaker_stats(
    segments: &[sherpa_rs::diarize::Segment],
) -> std::collections::BTreeMap<i32, (f32, usize)> {
    let mut stats: std::collections::BTreeMap<i32, (f32, usize)> = std::collections::BTreeMap::new();
    for seg in segments {
        let duration = seg.end - seg.start;
        let entry = stats.entry(seg.speaker).or_insert((0.0, 0));
        entry.0 += duration;
        entry.1 += 1;
    }
    stats
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI Parsing
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "asr-sherpa")]
struct CliArgs {
    seg_model: std::path::PathBuf,
    emb_model: std::path::PathBuf,
    audio_path: std::path::PathBuf,
    num_clusters: Option<i32>,
    threshold: Option<f32>,
    min_duration_on: f32,
    min_duration_off: f32,
    provider: String,
    debug: bool,
    json: bool,
}

#[cfg(feature = "asr-sherpa")]
fn parse_args(args: &[String]) -> Result<CliArgs> {
    use std::path::PathBuf;

    let mut seg_model: Option<PathBuf> = None;
    let mut emb_model: Option<PathBuf> = None;
    let mut audio_path: Option<PathBuf> = None;
    let mut num_clusters: Option<i32> = None;
    let mut threshold: Option<f32> = None;
    let mut min_duration_on = 0.3f32;
    let mut min_duration_off = 0.2f32;
    let mut provider = "cpu".to_string();
    let mut debug = false;
    let mut json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seg-model" | "-s" => {
                i += 1;
                seg_model = Some(PathBuf::from(args.get(i).context("--seg-model requires a value")?));
            }
            "--emb-model" | "-e" => {
                i += 1;
                emb_model = Some(PathBuf::from(args.get(i).context("--emb-model requires a value")?));
            }
            "--audio" | "-a" => {
                i += 1;
                audio_path = Some(PathBuf::from(args.get(i).context("--audio requires a value")?));
            }
            "--num-clusters" | "-n" => {
                i += 1;
                num_clusters = Some(
                    args.get(i)
                        .context("--num-clusters requires a value")?
                        .parse()
                        .context("--num-clusters must be an integer")?,
                );
            }
            "--threshold" => {
                i += 1;
                threshold = Some(
                    args.get(i)
                        .context("--threshold requires a value")?
                        .parse()
                        .context("--threshold must be a float")?,
                );
            }
            "--min-on" => {
                i += 1;
                min_duration_on = args
                    .get(i)
                    .context("--min-on requires a value")?
                    .parse()
                    .context("--min-on must be a float")?;
            }
            "--min-off" => {
                i += 1;
                min_duration_off = args
                    .get(i)
                    .context("--min-off requires a value")?
                    .parse()
                    .context("--min-off must be a float")?;
            }
            "--provider" => {
                i += 1;
                provider = args.get(i).context("--provider requires a value")?.clone();
            }
            "--debug" => {
                debug = true;
            }
            "--json" => {
                json = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            arg if arg.starts_with('-') => {
                bail!("Unknown option: {}", arg);
            }
            _ => {
                // Positional arguments for legacy compatibility
                if seg_model.is_none() {
                    seg_model = Some(PathBuf::from(&args[i]));
                } else if emb_model.is_none() {
                    emb_model = Some(PathBuf::from(&args[i]));
                } else if audio_path.is_none() {
                    audio_path = Some(PathBuf::from(&args[i]));
                } else {
                    bail!("Unexpected positional argument: {}", args[i]);
                }
            }
        }
        i += 1;
    }

    // Validate required arguments
    let seg_model = seg_model.context(
        "Missing segmentation model. Use --seg-model or positional arg 1.",
    )?;
    let emb_model = emb_model.context(
        "Missing embedding model. Use --emb-model or positional arg 2.",
    )?;
    let audio_path = audio_path.context(
        "Missing audio file. Use --audio or positional arg 3.",
    )?;

    // Validate file existence
    if !seg_model.exists() {
        bail!("Segmentation model not found: {}", seg_model.display());
    }
    if !emb_model.exists() {
        bail!("Embedding model not found: {}", emb_model.display());
    }
    if !audio_path.exists() {
        bail!("Audio file not found: {}", audio_path.display());
    }

    Ok(CliArgs {
        seg_model,
        emb_model,
        audio_path,
        num_clusters,
        threshold,
        min_duration_on,
        min_duration_off,
        provider,
        debug,
        json,
    })
}

#[cfg(feature = "asr-sherpa")]
fn print_usage() {
    eprintln!(
        r#"Offline Speaker Diarization - Identify "who spoke when" in an audio file

USAGE:
    cargo run --example diarize_offline --features asr-sherpa -- [OPTIONS]

    # Or with positional arguments:
    cargo run --example diarize_offline --features asr-sherpa -- \
        <seg_model.onnx> <emb_model.onnx> <audio.wav>

OPTIONS:
    -s, --seg-model <PATH>   Segmentation model (pyannote .onnx) [required]
    -e, --emb-model <PATH>   Embedding model (3dspeaker .onnx) [required]
    -a, --audio <PATH>       Audio file to diarize (.wav, 16kHz mono) [required]
    -n, --num-clusters <N>   Expected number of speakers (auto if not set)
        --threshold <FLOAT>  Clustering threshold (default: auto)
        --min-on <FLOAT>     Min speech duration in seconds (default: 0.3)
        --min-off <FLOAT>    Min silence gap to merge in seconds (default: 0.2)
        --provider <NAME>    ONNX provider: cpu, cuda, coreml (default: cpu)
        --debug              Enable debug output
        --json               Output result as JSON
    -h, --help               Print this help

EXAMPLES:
    # Basic usage
    cargo run --example diarize_offline --features asr-sherpa -- \
        --seg-model pyannote/model.onnx \
        --emb-model 3dspeaker.onnx \
        --audio meeting.wav

    # With known number of speakers
    cargo run --example diarize_offline --features asr-sherpa -- \
        --seg-model pyannote/model.int8.onnx \
        --emb-model 3dspeaker.onnx \
        --audio interview.wav \
        --num-clusters 2 \
        --json

NOTES:
    - Audio must be 16kHz sample rate (use ffmpeg to resample if needed)
    - Stereo files will be automatically downmixed to mono
    - Use model.int8.onnx for faster CPU inference
    - RTF < 1.0 means faster than real-time processing
    - For online/streaming use, consider utterance-level speaker labeling
      with EmbeddingExtractor + EmbeddingManager instead
"#
    );
}
