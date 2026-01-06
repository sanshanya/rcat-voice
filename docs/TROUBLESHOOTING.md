# Troubleshooting（Windows / 混用原生依赖）

本页记录近期在 Windows 上落地时遇到的关键问题、排查结论与推荐方案。

## 1) `gpt-sovits`（libtorch/tch）进程内崩溃：`0xc0000374 STATUS_HEAP_CORRUPTION`

**现象**

- 进程直接退出，Windows 报错 `0xc0000374 STATUS_HEAP_CORRUPTION`。
- 常见触发点在 GPT-SoVITS 初始化/推理阶段（如 `SpeakerV2Pro::pre_handle_ref` 附近），没有 Rust backtrace。

**判断**

- 这是典型的 **native heap corruption**（内存被写坏/错 free），通常由 **多个大型原生运行时在同进程混用** 引起（libtorch/CUDA + ONNX Runtime + 其它 C/C++ 依赖）。
- `KMP_DUPLICATE_LIB_OK=TRUE` 只对部分 OpenMP 重复加载问题有效，无法保证不崩。

**最稳妥方案（推荐）**

- **应用内（例如 Tauri）优先使用 `gpt-sovits-onnx`**（不加载 libtorch），将 GPU TTS 留给独立进程：
  - 进程隔离 worker（本机 HTTP，OpenAI 风格接口 `/v1/audio/speech`）。

本仓库内置了一个最小 worker（流式 `pcm16le`）：

```powershell
cd rcat-voice
cargo run --bin tts_worker --features tts-worker --release
```

**可尝试的缓解（不保证）**

- `KMP_DUPLICATE_LIB_OK=TRUE`：可能缓解 OpenMP 运行时冲突，但可能影响性能/线程配置。
- 确保同一进程内不要同时加载多份不同来源的 OpenMP/CRT（很难完全手工保证）。

## 2) `GSV_FP16=0`（fp32 输入）导致 TorchScript 报错

**现象**

- `GSV_FP16=0` 后，可能出现类似错误：
  - `RuntimeError: Input type (CUDAFloatType) and weight type (CUDAHalfType) should be the same`
- 上游库可能只返回固定文案（需要打开 debug 日志才能看到真实 TorchScript 栈）。

**结论**

- 你当前使用的 CUDA TorchScript 权重/算子链很可能是 **half-only**（权重是 Half，要求输入也是 Half）。
- 这意味着：在不更换/重导权重的前提下，**只能用 fp16 跑通**。

**建议**

- 保持 `GSV_FP16=1`（默认）来匹配 half-only 权重。
- 若确实要 fp32：需要一套 fp32 兼容的导出（权重/算子链必须支持 Float）。

## 3) Smart Turn：CPU/GPU ONNX 混用与选择

**建议**

- 若 `SMART_TURN_MODEL` 指向目录且目录里同时有 `*-cpu.onnx`/`*-gpu.onnx`：
  - 用 `SMART_TURN_VARIANT=gpu|cpu` 选择（默认 `gpu`）。
- 在 Windows 上若同时使用 `TTS_BACKEND=gpt-sovits`：
  - 已对 `smart-turn-*-cpu.onnx` 做了 fail-fast（默认拒绝）以避免已观测到的崩溃路径；
  - 如需强行使用：`SMART_TURN_ALLOW_CPU_MODEL=1`（不推荐）。

## 4) 日志与可观测性

- 打开 GPT-SoVITS 上游库 debug（用于看到真实 TorchScript 错误）：
  - `RUST_LOG=gpt_sovits_rs=debug`
- 打开本项目的文本/流式指标：
  - `VOICE_TTS_METRICS=1`（推荐，总开关；也会联动开启 `GSV_TEXT_METRICS`）
  - `TTS_WORKER_METRICS=1`（仅 worker 进程请求指标）
  - `GSV_TEXT_METRICS=1`（仅 GPT-SoVITS 文本/推理阶段）
