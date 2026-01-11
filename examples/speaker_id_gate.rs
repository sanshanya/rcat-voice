//! Speaker ID Gate Example
//!
//! Demonstrates speaker verification using sherpa-rs EmbeddingExtractor.
//! Use this as a gate to only respond to the "owner" in a voice assistant.
//!
//! **Usage**:
//! ```bash
//! cargo run --example speaker_id_gate --features asr-sherpa -- \
//!     --model path/to/speaker_model.onnx \
//!     --enroll owner_sample1.wav \
//!     --enroll owner_sample2.wav \
//!     --test unknown_speaker.wav
//! ```
//!
//! **Model Download**:
//! ```bash
//! # 3dspeaker (192-dim, recommended for Chinese)
//! wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
//!
//! # NeMo SpeakerNet (512-dim, English)
//! wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
//! ```
//!
//! **Low-Latency Integration**:
//! In a voice assistant, insert this gate between VAD/turn-end and ASR:
//! ```text
//! Mic → VAD/TurnEnd → speaker_id(embedding) → [if match] → ASR → LLM → TTS
//!                            │
//!                            └── [if not match] → discard/log
//! ```
//! Key: Run embedding once per utterance (VAD segment), not per audio chunk.

#[cfg(feature = "asr-sherpa")]
use anyhow::{Context, Result, bail};

#[cfg(not(feature = "asr-sherpa"))]
fn main() {
    eprintln!("This example requires `--features asr-sherpa`");
}

#[cfg(feature = "asr-sherpa")]
fn main() -> Result<()> {
    use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig, DEFAULT_SIMILARITY_THRESHOLD};

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

    println!("Speaker ID Gate");
    println!("================");
    println!("Model: {}", cli.model.display());
    println!("Enrollment files: {} file(s)", cli.enroll_wavs.len());
    println!("Test file: {}", cli.test_wav.display());
    println!("Threshold: {:.3}", cli.threshold);
    println!("Threads: {}", cli.num_threads);
    println!("Provider: {}", cli.provider);
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // Initialize Extractor (reuse for all embeddings)
    // ─────────────────────────────────────────────────────────────────────────
    let config = ExtractorConfig {
        model: cli.model.to_string_lossy().to_string(),
        provider: Some(cli.provider.clone()),
        num_threads: Some(cli.num_threads),
        debug: cli.debug,
    };

    let init_start = std::time::Instant::now();
    let mut extractor = EmbeddingExtractor::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create EmbeddingExtractor: {}", e))?;
    let init_ms = init_start.elapsed().as_millis();
    println!("Extractor initialized in {}ms (embedding_dim={})", init_ms, extractor.embedding_size);

    // ─────────────────────────────────────────────────────────────────────────
    // Enrollment: Extract embeddings from all enrollment files and average
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[Enrollment Phase]");
    let mut enroll_embeddings: Vec<Vec<f32>> = Vec::with_capacity(cli.enroll_wavs.len());

    for (i, enroll_path) in cli.enroll_wavs.iter().enumerate() {
        let start = std::time::Instant::now();
        let (samples, sample_rate) = read_wav_mono_16k(enroll_path)?;
        let duration_s = samples.len() as f32 / sample_rate as f32;

        let emb = extractor
            .compute_speaker_embedding(samples, sample_rate)
            .map_err(|e| anyhow::anyhow!("Failed to compute embedding for {:?}: {}", enroll_path, e))?;

        let elapsed_ms = start.elapsed().as_millis();
        println!(
            "  [{}] {} ({:.2}s) → embedding computed in {}ms",
            i + 1,
            enroll_path.display(),
            duration_s,
            elapsed_ms
        );
        enroll_embeddings.push(emb);
    }

    // Average all enrollment embeddings to create the "voiceprint"
    let voiceprint = average_embeddings(&enroll_embeddings);
    println!("Voiceprint created (averaged {} embeddings)", enroll_embeddings.len());

    // ─────────────────────────────────────────────────────────────────────────
    // Test: Extract embedding from test file and compare
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[Verification Phase]");
    let start = std::time::Instant::now();
    let (test_samples, test_sr) = read_wav_mono_16k(&cli.test_wav)?;
    let test_duration_s = test_samples.len() as f32 / test_sr as f32;

    let test_emb = extractor
        .compute_speaker_embedding(test_samples, test_sr)
        .map_err(|e| anyhow::anyhow!("Failed to compute embedding for test file: {}", e))?;
    let elapsed_ms = start.elapsed().as_millis();
    println!(
        "Test: {} ({:.2}s) → embedding computed in {}ms",
        cli.test_wav.display(),
        test_duration_s,
        elapsed_ms
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Similarity Scoring
    // ─────────────────────────────────────────────────────────────────────────
    let similarity = cosine_similarity(&voiceprint, &test_emb);
    let matched = similarity >= cli.threshold;

    println!("\n[Result]");
    println!("Cosine Similarity: {:.4}", similarity);
    println!("Threshold: {:.4} (default: {:.4})", cli.threshold, DEFAULT_SIMILARITY_THRESHOLD);
    println!("Decision: {}", if matched { "✓ MATCH (allow)" } else { "✗ NO MATCH (reject)" });

    // Structured output for programmatic use
    if cli.json {
        let json = serde_json::json!({
            "similarity": similarity,
            "threshold": cli.threshold,
            "matched": matched,
            "enrollment_count": enroll_embeddings.len(),
            "test_duration_s": test_duration_s,
        });
        println!("\n{}", serde_json::to_string_pretty(&json)?);
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
            "WAV file must be 16kHz, got {}Hz. Please resample: \n\
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
                // Convert i32 to i16
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| (v >> 16) as i16))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                bail!("Unsupported bits_per_sample: {}", spec.bits_per_sample);
            }
        }
        hound::SampleFormat::Float => {
            // Convert f32 to i16
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
// Math Utilities
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "asr-sherpa")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Embedding dimensions must match");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    dot / denom
}

#[cfg(feature = "asr-sherpa")]
fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let count = embeddings.len() as f32;
    let mut avg = vec![0.0f32; dim];

    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            avg[i] += v;
        }
    }

    for v in &mut avg {
        *v /= count;
    }

    // L2-normalize the averaged embedding
    let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for v in &mut avg {
        *v /= norm;
    }

    avg
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI Parsing
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "asr-sherpa")]
struct CliArgs {
    model: std::path::PathBuf,
    enroll_wavs: Vec<std::path::PathBuf>,
    test_wav: std::path::PathBuf,
    threshold: f32,
    num_threads: usize,
    provider: String,
    debug: bool,
    json: bool,
}

#[cfg(feature = "asr-sherpa")]
fn parse_args(args: &[String]) -> Result<CliArgs> {
    use sherpa_rs::speaker_id::DEFAULT_SIMILARITY_THRESHOLD;
    use std::path::PathBuf;

    let mut model: Option<PathBuf> = None;
    let mut enroll_wavs: Vec<PathBuf> = Vec::new();
    let mut test_wav: Option<PathBuf> = None;
    let mut threshold = DEFAULT_SIMILARITY_THRESHOLD;
    let mut num_threads = 1usize;
    let mut provider = "cpu".to_string();
    let mut debug = false;
    let mut json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model = Some(PathBuf::from(args.get(i).context("--model requires a value")?));
            }
            "--enroll" | "-e" => {
                i += 1;
                enroll_wavs.push(PathBuf::from(args.get(i).context("--enroll requires a value")?));
            }
            "--test" | "-t" => {
                i += 1;
                test_wav = Some(PathBuf::from(args.get(i).context("--test requires a value")?));
            }
            "--threshold" => {
                i += 1;
                threshold = args
                    .get(i)
                    .context("--threshold requires a value")?
                    .parse()
                    .context("--threshold must be a float")?;
            }
            "--num-threads" => {
                i += 1;
                num_threads = args
                    .get(i)
                    .context("--num-threads requires a value")?
                    .parse()
                    .context("--num-threads must be an integer")?;
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
                // Positional arguments: model, test_wav (legacy mode)
                if model.is_none() {
                    model = Some(PathBuf::from(&args[i]));
                } else if test_wav.is_none() {
                    test_wav = Some(PathBuf::from(&args[i]));
                } else {
                    bail!("Unexpected positional argument: {}", args[i]);
                }
            }
        }
        i += 1;
    }

    // Validate required arguments
    let model = model.context(
        "Missing --model. Usage: --model speaker_model.onnx --enroll x.wav --test y.wav",
    )?;
    let test_wav = test_wav.context(
        "Missing --test. Usage: --model speaker_model.onnx --enroll x.wav --test y.wav",
    )?;

    if enroll_wavs.is_empty() {
        bail!(
            "At least one --enroll file is required.\n\
             Tip: Use 2-5 enrollment samples of 2-5 seconds each for best accuracy."
        );
    }

    // Validate file existence
    if !model.exists() {
        bail!("Model file not found: {}", model.display());
    }
    for p in &enroll_wavs {
        if !p.exists() {
            bail!("Enrollment file not found: {}", p.display());
        }
    }
    if !test_wav.exists() {
        bail!("Test file not found: {}", test_wav.display());
    }

    Ok(CliArgs {
        model,
        enroll_wavs,
        test_wav,
        threshold,
        num_threads,
        provider,
        debug,
        json,
    })
}

#[cfg(feature = "asr-sherpa")]
fn print_usage() {
    eprintln!(
        r#"Speaker ID Gate - Verify if a speaker matches enrolled voiceprints

USAGE:
    cargo run --example speaker_id_gate --features asr-sherpa -- [OPTIONS]

OPTIONS:
    -m, --model <PATH>       Speaker embedding model (.onnx) [required]
    -e, --enroll <PATH>      Enrollment WAV file (can specify multiple) [required]
    -t, --test <PATH>        Test WAV file to verify [required]
        --threshold <FLOAT>  Similarity threshold (default: 0.5)
        --num-threads <N>    ONNX threads (default: 1)
        --provider <NAME>    ONNX provider: cpu, cuda, coreml (default: cpu)
        --debug              Enable debug output
        --json               Output result as JSON
    -h, --help               Print this help

EXAMPLES:
    # Basic usage with one enrollment file
    cargo run --example speaker_id_gate --features asr-sherpa -- \
        --model 3dspeaker.onnx --enroll owner.wav --test test.wav

    # Multiple enrollment files (recommended for stability)
    cargo run --example speaker_id_gate --features asr-sherpa -- \
        --model 3dspeaker.onnx \
        --enroll owner1.wav --enroll owner2.wav --enroll owner3.wav \
        --test unknown.wav --threshold 0.6

NOTES:
    - All WAV files must be 16kHz sample rate
    - Stereo files will be automatically downmixed to mono
    - Recommended: 2-5 enrollment samples, each 2-5 seconds of clean speech
    - Threshold tuning: Start with 0.5, increase to reduce false positives
"#
    );
}
