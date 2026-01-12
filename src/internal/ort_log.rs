use std::sync::OnceLock;

static ORT_LOG_LEVEL_ONCE: OnceLock<()> = OnceLock::new();

pub(crate) fn apply_from_env() {
    ORT_LOG_LEVEL_ONCE.get_or_init(|| {
        let level = parse_level_from_env();
        match ort::environment::get_environment() {
            Ok(env) => env.set_log_level(level),
            Err(err) => {
                tracing::debug!("Failed to initialize ORT environment for log level: {err}");
            }
        }
    });
}

fn parse_level_from_env() -> ort::logging::LogLevel {
    let raw = std::env::var("ORT_LOG").ok();
    let Some(raw) = raw else {
        // Default to Warning to avoid noisy ORT info logs (GraphTransformer, allocator, etc.)
        return ort::logging::LogLevel::Warning;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "verbose" => ort::logging::LogLevel::Verbose,
        "info" => ort::logging::LogLevel::Info,
        "warning" | "warn" => ort::logging::LogLevel::Warning,
        "error" => ort::logging::LogLevel::Error,
        "fatal" => ort::logging::LogLevel::Fatal,
        _ => ort::logging::LogLevel::Warning,
    }
}

