# Vendored `sherpa-rs` (Upstream Tracking)

This directory contains a vendored copy of `sherpa-rs` + `sherpa-rs-sys`, used by `rcat-voice` for the `asr-sherpa` feature.

## Upstream

- Repo: https://github.com/thewh1teagle/sherpa-rs
- Baseline tag: `v0.6.8` (commit `6b17eb373733dc6f59a51b69c04d735e9bb45537`)
- `sherpa-onnx` prebuilt archives pinned in `sherpa-rs-sys/dist.json` (currently `v1.12.20`)

## Why vendored

- Some newer ASR models require newer `sherpa-onnx` C-API/header than the upstream crate release we started from.
- We want deterministic builds by pinning the exact `sherpa-onnx` prebuilt archives + checksums we use.

## Local changes (keep minimal)

This vendor copy may include changes such as:

- Pin/update `sherpa-onnx` archive versions and checksums (`sherpa-rs-sys/dist.json`, `sherpa-rs-sys/checksum.txt`).
- Small wrapper fixes for offline recognizers/VAD to match the C-API expectations (e.g. explicit `decoding_method`, avoid extra allocations).
- Build-time robustness improvements (e.g. handling proxy env vars, checksum parsing).

## Maintenance checklist

1) **Prefer switching back to upstream**  
   If upstream `sherpa-rs` supports the models and C-API we need, switch `rcat-voice/Cargo.toml` back to the official dependency and delete this vendor copy.

2) **If we keep patches, track them explicitly**  
   If local changes grow, convert this directory into a git submodule or forked repo (with an upstream remote), so we can rebase and contribute changes back upstream.

3) **Consider upstream PRs**  
   If our changes are generally useful (new model support, wrapper fixes, build improvements), open a PR to upstream `sherpa-rs` so we can eventually remove the vendor directory.
