#[cfg(not(feature = "tts-worker"))]
compile_error!("`rcat_voice::worker` requires the `tts-worker` feature");
#[cfg(not(target_os = "windows"))]
compile_error!("`rcat_voice::worker` is currently Windows-only (GPU gpt-sovits via libtorch)");

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::{Method, Request, Response, StatusCode};
use hyper::body::Incoming;
use hyper::body::Frame;
use hyper::header::{CONTENT_TYPE, HeaderValue};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::generator::gpt_sovits::GptSovitsWorkerModel;
use crate::internal::env;
use crate::remote_tts_protocol::{ErrorBody, ErrorResponse, SpeechRequest};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

fn metrics_enabled() -> bool {
    env::bool01("VOICE_TTS_METRICS", false) || env::bool01("TTS_WORKER_METRICS", false)
}

pub async fn run_from_env() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let bind = std::env::var("TTS_WORKER_BIND").unwrap_or_else(|_| "127.0.0.1:7878".to_string());
    let addr = parse_bind(&bind)?;

    let state = Arc::new(WorkerState::from_env()?);

    if metrics_enabled() {
        info!("tts-worker: metrics enabled (VOICE_TTS_METRICS=1 or TTS_WORKER_METRICS=1)");
    }
    info!("tts-worker: listening on http://{}", addr);
    serve(addr, state).await
}

fn parse_bind(raw: &str) -> Result<SocketAddr> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("TTS_WORKER_BIND is empty");
    }
    if trimmed.contains(':') {
        return trimmed
            .parse::<SocketAddr>()
            .with_context(|| format!("invalid TTS_WORKER_BIND: {trimmed}"));
    }
    format!("127.0.0.1:{trimmed}")
        .parse::<SocketAddr>()
        .with_context(|| format!("invalid TTS_WORKER_BIND: {trimmed}"))
}

async fn serve(addr: SocketAddr, state: Arc<WorkerState>) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind failed: {addr}"))?;

    loop {
        let (stream, peer) = listener.accept().await.context("accept failed")?;
        let io = TokioIo::new(stream);
        let state = state.clone();
        tokio::spawn(async move {
            let service = service_fn(move |req| handle(req, state.clone()));
            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                warn!("tts-worker: connection error (peer={}): {}", peer, err);
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    state: Arc<WorkerState>,
) -> std::result::Result<Response<BoxBody>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    match (method, path.as_str()) {
        (Method::GET, "/health") => Ok(text_response(StatusCode::OK, "ok\n")),
        (Method::POST, "/v1/audio/speech") => Ok(handle_speech(req, state).await),
        _ => Ok(text_response(StatusCode::NOT_FOUND, "not found\n")),
    }
}

async fn handle_speech(req: Request<Incoming>, state: Arc<WorkerState>) -> Response<BoxBody> {
    let bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => return json_error(StatusCode::BAD_REQUEST, format!("read body failed: {err}")),
    };

    let mut request: SpeechRequest = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(err) => return json_error(StatusCode::BAD_REQUEST, format!("invalid json: {err}")),
    };

    request.input = request.input.trim().to_string();
    if request.input.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "input is empty".to_string());
    }

    let fmt = request.response_format.trim().to_ascii_lowercase();
    if fmt != "pcm16" && fmt != "pcm_s16le" && fmt != "pcm" {
        return json_error(
            StatusCode::BAD_REQUEST,
            format!("unsupported response_format: {fmt} (expected pcm16)"),
        );
    }

    if !request.stream {
        // We only implement chunked streaming for now; clients may still read it as a full body.
        request.stream = true;
    }

    if request.sample_rate.is_some_and(|v| v != 32_000) || request.channels.is_some_and(|v| v != 1) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "only 32000Hz mono PCM is supported".to_string(),
        );
    }

    match request.model.trim().to_ascii_lowercase().as_str() {
        "gpt-sovits" => {}
        other => {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("unsupported model: {other} (expected gpt-sovits)"),
            );
        }
    }

    let (tx, rx) = mpsc::channel::<Bytes>(8);
    let input = request.input;
    let model = state.gpt_sovits.clone();
    let request_id = state.next_request_id();
    let log_metrics = metrics_enabled();

    tokio::task::spawn_blocking(move || {
        let start_ts = Instant::now();
        if log_metrics {
            info!(
                "tts-worker: req#{} start (chars={})",
                request_id,
                input.chars().count()
            );
        }

        let outcome = model.stream_pcm16le(&input, tx, log_metrics);
        if log_metrics {
            let done_ts = Instant::now();
            let ttfb_ms = outcome
                .stats
                .first_audio_ts
                .map(|ts| ts.duration_since(start_ts).as_millis());
            let gen_ms = done_ts.duration_since(start_ts).as_millis();
            let audio_ms = (outcome.stats.samples != 0)
                .then_some(((outcome.stats.samples as f64) * 1000.0 / 32_000.0) as u64);
            let rtf = audio_ms
                .and_then(|ms| (ms != 0).then_some(ms))
                .map(|ms| (gen_ms as f64) / (ms as f64));
            info!(
                "tts-worker: req#{} done ok={} ttfb_ms={:?} gen_ms={} chunks={} samples={} audio_ms={:?} bytes={} rtf={:?}",
                request_id,
                outcome.result.is_ok(),
                ttfb_ms,
                gen_ms,
                outcome.stats.chunks,
                outcome.stats.samples,
                audio_ms,
                outcome.stats.bytes,
                rtf
            );
        }

        if let Err(err) = outcome.result {
            warn!("tts-worker: req#{} synth failed: {err:?}", request_id);
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async {
        match rx.recv().await {
            Some(chunk) => Some((Ok::<Frame<Bytes>, Infallible>(Frame::data(chunk)), rx)),
            None => None,
        }
    });
    let body = StreamBody::new(stream).boxed();

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("audio/L16; rate=32000; channels=1"),
    );
    response
}

fn text_response(status: StatusCode, text: &str) -> Response<BoxBody> {
    let mut response = Response::new(Full::new(Bytes::from(text.to_string())).boxed());
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    response
}

fn json_error(status: StatusCode, message: String) -> Response<BoxBody> {
    let body = ErrorResponse {
        error: ErrorBody {
            message,
            r#type: Some("tts_worker_error".to_string()),
        },
    };
    let json = serde_json::to_vec(&body).unwrap_or_else(|_| br#"{"error":{"message":"internal json error"}}"#.to_vec());
    let mut response = Response::new(Full::new(Bytes::from(json)).boxed());
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

struct WorkerState {
    gpt_sovits: Arc<GptSovitsWorkerModel>,
    request_seq: AtomicU64,
}

impl WorkerState {
    fn from_env() -> Result<Self> {
        Ok(Self {
            gpt_sovits: Arc::new(GptSovitsWorkerModel::from_env_cuda_only()?),
            request_seq: AtomicU64::new(1),
        })
    }

    fn next_request_id(&self) -> u64 {
        self.request_seq.fetch_add(1, Ordering::Relaxed)
    }
}
