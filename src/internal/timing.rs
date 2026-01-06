use std::time::Duration;
use tokio::time::Instant;

pub(crate) fn time_if<T, F>(enabled: bool, f: F) -> (T, Option<Duration>)
where
    F: FnOnce() -> T,
{
    if enabled {
        let start = Instant::now();
        let out = f();
        (out, Some(start.elapsed()))
    } else {
        (f(), None)
    }
}

