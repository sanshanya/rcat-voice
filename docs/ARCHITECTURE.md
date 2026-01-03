# rcat-voice Architecture

## Goals

- Provide a low-latency, streaming TTS pipeline that accepts text deltas.
- Keep LLM integration out of the core library (examples only).
- Expose a stable, typed config API; environment variables are optional.

## Non-goals

- Full LLM client or prompt orchestration.
- Remote TTS service implementation (placeholder only).

## High-level Components

- `streaming`: `StreamSession` + `StreamControl` (lifecycle + control).
- `tokenizer`: turns deltas into `Segment` objects with timestamps.
- `pipeline`: schedules TTS synthesis and playback.
- `generator`: TTS backends (`gpt-sovits`, `gpt-sovits-onnx`, `os`, `remote`).
- `audio`: audio backends (`rodio`, placeholders for others).

## Data Flow

```
LLM Deltas
    |
    v
Tokenizer ----> Segment ----> Pipeline ----> TtsEngine ----> AudioBackend
```

1. LLM deltas are pushed into a channel.
2. `Tokenizer` buffers text, splits on boundaries, and emits `Segment`.
3. `Pipeline` consumes segments and calls `TtsEngine` to synthesize/play.
4. `AudioBackend` plays PCM samples and reports playback completion.

## Session Lifecycle

`StreamSession` spawns three async tasks:

- Tokenizer task (delta -> segment).
- Pipeline task (segment -> TTS -> audio).
- Buffer poll task (periodically reads `buffered_ms()` for adaptive chunking).

`StreamControl` provides:

- `sender()` to push deltas.
- `mark_llm_start()` to record t0 for metrics.
- `pause()` to stop playback and clear queued work.
- `cancel()` to stop playback and cancel the current stream.

`StreamSession::new()` uses default configs. `StreamSession::from_env()` uses
environment variables. A builder is provided for explicit configuration.

## Tokenizer Details

- The tokenizer accumulates deltas into a buffer and flushes on boundaries.
- It emits eager short chunks first (low TTFA), then normal chunks.
- Relax mode is triggered by audio buffer waterline (`buffered_ms()`):
  - If buffer is deep enough, it increases chunk size to improve throughput.
- Each `Segment` contains timestamps:
  - `task_start` (t0), `first_token_ts` (t1), `last_token_ts`, `segment_sent_ts` (t2).

## Pipeline Details

- Sequential by default.
- Parallel synth queue is only used when:
  - `PipelineConfig.parallel_synth` is true, and
  - `TtsEngine::supports_synthesis_queue()` is true.
- `gpt-sovits` and `gpt-sovits-onnx` currently return `false` to avoid
  misleading parallelism (they serialize on an internal mutex).

Playback metrics are logged using real timestamps only:

- `first_audio_ts` and `gen_done_ts` are always available.
- `play_done_rx` is optional. If a backend does not provide it, the pipeline
  does not log "play done" metrics for that segment.

## Cancellation Semantics

`stop()` means: terminate all in-flight synthesis and playback.

Implementation:

- `CancelToken` stores a monotonic epoch.
- Each operation captures a `CancelScope` (epoch snapshot).
- `stop()` increments the epoch and calls `AudioBackend::stop()`.
- Loops exit when `CancelScope::is_cancelled()` is true.

This prevents concurrent tasks from resetting a global cancel flag.

## TTS Engine Contract

`TtsEngine` defines:

- `speak(text)` for direct synth+play.
- `synthesize(text)` + `play_samples(audio)` for decoupled paths.
- `stop()` to cancel all in-flight work.
- `buffered_ms()` for adaptive tokenization.

`TtsMetrics` fields:

- `start_ts`
- `first_audio_ts` (optional)
- `gen_done_ts`
- `play_done_ts`
- `play_done_rx` (optional, for real completion)

## Audio Backend Contract

`AudioBackend` defines:

- `begin_segment()` -> `SegmentWriter`
- `stop()` to clear queued audio
- `sample_rate()` / `channels()` / `buffered_ms()`

`SegmentWriter` defines:

- `push(samples, CancelScope)` for streaming PCM.
- `finish(cancelled)` returning `SegmentPlayback` with `play_done_rx` (optional).

`rodio` backend:

- Uses a ring buffer and a playback marker queue.
- Supports `play_done_rx` for real playback completion timestamps.
- Supports buffer waterline reporting via `buffered_ms()`.

## Backend Specifics

### GPT-SoVITS (CUDA)

- Windows + CUDA LibTorch only.
- Requires 32kHz mono output.
- Uses a single internal mutex (serial inference).
- Supports dynamic first chunk tokens to reduce TTFA.

### GPT-SoVITS ONNX (CPU)

- CPU inference via ONNX Runtime.
- Output sample rate/channels must match audio backend.
- Uses a single internal mutex (serial inference).

### OS TTS

- Uses OS commands for speech.
- Synchronous (no playback completion callback).

### Remote TTS

- Placeholder; not implemented.

## Configuration Model

Typed configs for all major components:

- `AudioConfig` / `RodioConfig`
- `TokenizerConfig`
- `PipelineConfig`
- `StreamConfig`
- `GptSovitsConfig` / `GptSovitsChunkPolicy`
- `GptSovitsOnnxConfig` / `GptSovitsOnnxSampling`

Environment variables only apply to `*_from_env` and `StreamSession::from_env`.
Explicit configs never read the environment.

## Extensibility

To add a TTS backend:

- Implement `TtsEngine`.
- Decide whether you support synthesis/playback decoupling.
- If you provide `play_done_rx`, you get real playback completion logs.

To add an audio backend:

- Implement `AudioBackend` + `SegmentWriter`.
- Provide `buffered_ms()` if you want tokenizer relax mode to adapt.
