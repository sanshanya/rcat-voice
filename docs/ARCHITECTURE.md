# rcat-voice Architecture

This repository contains:

- A low-latency **streaming TTS pipeline** that consumes text deltas.
- Optional **ASR (mic/file)** and **turn detection** modules to build an end-to-end voice loop.
- Examples that wire everything together (mic ASR → turn end → LLM stream → TTS stream).

Feature flags are used to keep the core small:

- `audio-rodio` (playback), `gpt-sovits` / `gpt-sovits-onnx` / `os` (TTS)
- `asr-sherpa` (ASR via sherpa-onnx), `asr-mic` (cpal mic input)
- `turn-smart` (Smart Turn ONNX)

## Goals

- Low-latency, streaming **Speech ↔ Text ↔ Speech** loop building blocks.
- Clear lifecycle and cancellation semantics (full-duplex + barge-in friendly).
- Typed configuration APIs; env vars are optional conveniences.

## Non-goals

- Full “agent framework” (planning, tools, memory, etc).
- Full LLM prompt orchestration (examples keep it minimal).
- Remote TTS service (placeholder only).

## High-level Components

- `asr`: offline streaming ASR (currently sherpa-onnx models) + Silero VAD segmentation.
- `turn`: turn-end detection (Smart Turn ONNX, usually evaluated during silence).
- `streaming`: `StreamSession` + `StreamControl` (delta ingestion, lifecycle).
- `tokenizer`: delta → `Segment` (text chunking with timestamps).
- `pipeline`: `Segment` → `TtsEngine` scheduling + playback metrics.
- `generator`: TTS backends (`gpt-sovits`, `gpt-sovits-onnx`, `os`, `remote` placeholder).
- `audio`: audio backends (`rodio`, placeholders for others).

## Data Flows

### TTS streaming pipeline

```
LLM deltas
   |
   v
Tokenizer ----> Segment ----> Pipeline ----> TtsEngine ----> AudioBackend
```

1. LLM deltas are pushed into a channel (`StreamControl::sender()`).
2. `Tokenizer` buffers text and flushes on boundaries, producing `Segment`.
3. `Pipeline` consumes segments and calls `TtsEngine` to synthesize/play.
4. `AudioBackend` plays PCM samples and optionally reports playback completion.

### ASR streaming (offline model + VAD segmentation)

```
PCM (mic/file) -> resample/downmix -> Silero VAD -> speech segment -> ASR transcribe -> AsrSegment
```

- Input accepts `i16` PCM, any sample rate/channels (converted to 16k mono internally).
- “Streaming” here means: you can continuously feed audio and **receive segmented results** as VAD endpoints trigger.

### Full duplex demo (voice assistant)

`examples/voice_assistant.rs` composes:

```
mic -> ring buffer -> ASR segments -> (Smart Turn) -> user turn text
                                               |
                                               v
                                         LLM stream (SSE)
                                               |
                                               v
                                        StreamSession (TTS)
```

It also supports “conservative barge-in”: require a short continuous speech streak before cancelling the assistant.

## Session Lifecycle (TTS)

`StreamSession` spawns async tasks:

- Tokenizer task (delta → segment).
- Pipeline task (segment → TTS → audio).
- Buffer poll task (reads `buffered_ms()` to enable tokenizer relax mode).

Control handles:

- `StreamControl`: holds delta sender + cancel/interrupt (keeps delta channel open).
- `StreamCancelHandle`: cancel/interrupt only (does not keep delta channel open).

Lifecycle APIs:

- `shutdown()`: cancel + stop audio immediately, then join tasks.
- `finish()`: close delta channel and wait for tasks to drain.
- `finish_or_cancel(cancel_rx)`: drain like `finish()`, but abort quickly on cancel (for barge-in).

## Tokenizer Details

- Starts with a few “eager” short segments (lower TTFA), then normal segments.
- Thresholds are configurable via env vars (`TOKENIZER_*`); see `README.md`.
- Has a “relax mode” triggered by audio buffer waterline (`buffered_ms()`):
  - if TTS buffer is deep enough, emits larger segments for throughput/efficiency.
- Each `Segment` carries timestamps:
  - `llm_start_ts` (t0), `first_token_ts` (t1, only first segment), `last_token_ts`, `segment_sent_ts` (t2)

## Pipeline Details

- Sequential by default (`speak(text)`).
- Optional parallel synth path is only used when:
  - `PipelineConfig.parallel_synth` is true, and
  - the engine reports `supports_synthesis_queue() == true`.
- `gpt-sovits` / `gpt-sovits-onnx` currently return `false` to avoid misleading parallelism
  (they serialize on an internal mutex anyway).

Playback metrics:

- `first_audio_ts` / `gen_done_ts` are always available.
- `play_done_rx` is optional; only some audio backends (e.g. `rodio`) can provide real completion.

## Cancellation Semantics

- TTS backends call `stop()` to terminate playback and clear queued work.
- `audio::CancelToken` uses a monotonic epoch; work checks a `CancelScope` snapshot to avoid global mutable flags.
- `interrupt` is treated as “stop and clear queued segments, then allow a new stream to start cleanly”.

## Audio Backend Contract

`AudioBackend`:

- `begin_segment()` → `SegmentWriter`
- `stop()` clears queued audio
- `sample_rate()` / `channels()` / optional `buffered_ms()`

`SegmentWriter`:

- `push(samples, CancelScope)` streams PCM samples
- `finish(cancelled)` returns `SegmentPlayback` (optionally includes `play_done_rx`)

`rodio` backend:

- ring buffer + playback markers
- exposes `buffered_ms()` for tokenizer relax mode

## ASR / Turn Detection Notes

- ASR is “offline model + online feeding”; VAD produces segments, then ASR transcribes those segments.
- Smart Turn is intended to run **during silence** (gated by VAD or a silence heuristic) to decide turn end.

## Docs

- Optimization proposals and next steps live in `docs/OPTIMIZATIONS.md`.
