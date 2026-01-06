# rcat-voice

流式文本 -> TTS 管线。核心库只消费文本 delta，LLM 集成放在 examples。

## 作为库使用

### 最小示例（OS TTS，无需额外特性）

```rust
use rcat_voice::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tts = TtsEngineBuilder::new(TtsBackend::Os).build()?;
    let session = StreamSession::builder(tts.clone()).build();
    let control = session.control();
    control.mark_llm_start();
    control.sender().send("Hello,".to_string()).await?;
    control.sender().send(" world!".to_string()).await?;
    session.shutdown().await?;
    Ok(())
}
```

### 典型用法（CUDA GPT-SoVITS）

需要启用 `gpt-sovits` 特性，并在 Windows + CUDA LibTorch 环境运行。

```rust
use rcat_voice::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let audio = rcat_voice::audio::build(&AudioConfig::default())?;
    let config = GptSovitsConfig::from_dir("v2pro")?;
    let tts = TtsEngineBuilder::new(TtsBackend::GptSovits(config))
        .audio_backend(audio)
        .build()?;
    let session = StreamSession::builder(tts.clone()).build();
    // ... send deltas via session.control().sender()
    session.shutdown().await?;
    Ok(())
}
```

如果希望继续使用环境变量风格的配置，可用 `*_from_env()` / `StreamSession::from_env()`。

常用配置入口：

- `AudioConfig` / `RodioConfig`
- `TokenizerConfig`
- `PipelineConfig`
- `StreamConfig`
- `GptSovitsConfig` / `GptSovitsOnnxConfig`

### Tokenizer 调参（无需重新编译）

分段阈值支持环境变量覆盖（单位：字符数；会自动归一化到 `min <= soft_max <= hard_max`，并 clamp 到 1–400）：

- Eager（默认 `2/6/12`）：`TOKENIZER_EAGER_MIN_CHARS` / `TOKENIZER_EAGER_SOFT_MAX_CHARS` / `TOKENIZER_EAGER_HARD_MAX_CHARS`
- Normal（默认 `10/20/40`）：`TOKENIZER_NORMAL_MIN_CHARS` / `TOKENIZER_NORMAL_SOFT_MAX_CHARS` / `TOKENIZER_NORMAL_HARD_MAX_CHARS`
- Relax（默认 `20/35/80`）：`TOKENIZER_RELAX_MIN_CHARS` / `TOKENIZER_RELAX_SOFT_MAX_CHARS` / `TOKENIZER_RELAX_HARD_MAX_CHARS`

其它相关参数：

- `TOKENIZER_EAGER_CHUNKS`（兼容旧名：`CHUNKER_EAGER_CHUNKS`）
- `TOKENIZER_RELAX_BUFFER_MS`（缓冲达到该值后进入 Relax）
- `TOKENIZER_RELAX_LOG=1`（打印 Relax on/off）

## 快速开始（Windows + CUDA GPT-SoVITS）

1) 安装 Rust（MSVC 工具链）。
2) 安装 CUDA LibTorch（2.9 版）并在当前终端设置环境变量：

```powershell
$env:LIBTORCH="C:\libtorch"
$env:Path="$env:LIBTORCH\lib;$env:Path"
$env:LIBTORCH_BYPASS_VERSION_CHECK="1"   # 可选：若 torch-sys 报 PyTorch 版本不匹配（如 2.9.1 vs 2.9.0）
```

3) 把 GPT-SoVITS 模型放到 `v2pro/`（与 `Cargo.toml` 同级，可用 `GSV_MODEL_DIR` 指定其他目录）：

- `mini-bart-g2p.pt`
- `g2pw_model.pt`（或 `g2pw.pt`）
- `bert_model.pt`（或 `bert.pt`）
- `ssl_model.pt`（或 `ssl.pt`）
- `t2s.pt` + `vits.pt`（或 `gpt_sovits_v2pro.cuda.pt` 作为合并权重）
- `ref.wav` + `ref.txt`

`ref.wav` 会自动转单声道并重采样到 32k。

4) 运行模拟流式示例：

```bash
cargo run --example stream_sim --features gpt-sovits
```

可选：开启文本/流式推理指标日志（如 `next_chunk first return time`）：

- PowerShell：`$env:GSV_TEXT_METRICS="1"`
- bash：`GSV_TEXT_METRICS=1 cargo run --example stream_sim --features gpt-sovits`

如果你在 Windows 上同时启用 `gpt-sovits`（libtorch）和 `turn-smart`/`asr-sherpa`（ONNX Runtime），遇到 `STATUS_HEAP_CORRUPTION` 或 OpenMP 运行时冲突，可尝试在启动前设置：`$env:KMP_DUPLICATE_LIB_OK="TRUE"`。

## 快速开始（CPU ONNX GPT-SoVITS）

1) 按 `gpt-sovits-onnx-rs` 的 `scripts/README.md` 转换模型，得到 ONNX 模型目录。
2) 把模型放到 `onnx/`（与 `Cargo.toml` 同级，或通过 `GSV_ONNX_MODEL_DIR` 指定）：

- `custom_vits.onnx`
- `ssl.onnx`
- `custom_t2s_encoder.onnx`
- `custom_t2s_fs_decoder.onnx`
- `custom_t2s_s_decoder.onnx`
- `g2pW.onnx`（可选，但建议保留）
- `bert.onnx`（可选，但建议保留）
- `g2p_en/`（可选，包含 `encoder_model.onnx` / `decoder_model.onnx`）
- `sv.onnx`（可选）
- `ref.wav` + `ref.txt`

`ref.wav` 需为单声道 16-bit PCM，采样率不限（会自动重采样到 16k/32k）。
若导出前缀不是 `custom`，请设置 `GSV_ONNX_EXPORT_NAME`。

3) 运行示例：

```bash
TTS_BACKEND=gpt-sovits-onnx \
GSV_ONNX_MODEL_DIR=onnx \
cargo run --example stream_sim --features gpt-sovits-onnx
```

PowerShell：

```powershell
$env:TTS_BACKEND="gpt-sovits-onnx"
$env:GSV_ONNX_MODEL_DIR="onnx"
cargo run --example stream_sim --features gpt-sovits-onnx
```

## 快速开始（ASR：Sherpa-ONNX / Paraformer）

当前 ASR 以 `sherpa-rs` 为基础，实现 **Paraformer(离线) / SenseVoice(离线) + Silero VAD(分段)** 的流式识别（按 VAD 端点输出分段结果）。本地 CPU 场景默认推荐 Paraformer（更轻量）。

1) 下载模型到 `asrmodel/` 或 `models/`（与 `Cargo.toml` 同级，或通过 `ASR_MODELS_ROOT` 指定）。

默认行为：如果检测到 `asrmodel/` 存在且未设置 `ASR_MODELS_ROOT`，会优先使用 `asrmodel/`；否则使用 `models/`。

- `<ASR_MODELS_ROOT>/silero_vad.onnx`（或 `<ASR_MODELS_ROOT>/silero_vad/silero_vad.onnx`）
- `<ASR_MODELS_ROOT>/sherpa-onnx-paraformer-zh-small-2024-03-09/`
  - `model.int8.onnx`
  - `tokens.txt`

（可选）也可以使用：

- `models/sherpa-onnx-paraformer-zh-2024-03-09/`
  - `model.onnx`（或 `model.int8.onnx`）
  - `tokens.txt`
- `models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en/`
  - `model.onnx`（或 `model.int8.onnx`）
  - `tokens.txt`
- `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`
  - `model.onnx`（或 `model.int8.onnx`）
  - `tokens.txt`
- `models/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17/`（FunASR-Nano，推荐）
  - `model.int8.onnx`
  - `tokens.txt`
- `models/sherpa-onnx-sense-voice-funasr-nano-2025-12-17/`（FunASR-Nano FP32）
  - `model.onnx`
  - `tokens.txt`

对应下载链接（`asr-models`）：

- `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17.tar.bz2`
- `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-funasr-nano-2025-12-17.tar.bz2`

2) 运行文件识别示例：

```bash
ASR_MODELS_ROOT=models \
ASR_MODEL=paraformer-zh-small \
cargo run --example asr_file --features asr-sherpa -- path/to/audio.wav
```

使用 FunASR Nano（推荐）：

```bash
ASR_MODELS_ROOT=asrmodel \
ASR_MODEL=funasr-nano-int8 \
ASR_INFER_LOG=1 \
ASR_METRICS=1 \
cargo run --example asr_file --features asr-sherpa -- path/to/audio.wav
```

3) 运行麦克风流式识别示例（Windows/macOS，按 VAD 端点输出分段；可选 Smart Turn 判定轮次结束）：

```powershell
$env:ASR_MODELS_ROOT="asrmodel"
$env:ASR_MODEL="funasr-nano-int8"
$env:ASR_VAD_MIN_SILENCE="0.45"   # 可选：更大=分段更少/更不敏感，但会更晚出结果
cargo run --example asr_mic --features asr-sherpa,asr-mic --release
```

启用 Smart Turn（在静音处更贴近“人类期望”的 turn end 判断）：

```powershell
$env:ASR_MODELS_ROOT="asrmodel"
$env:ASR_MODEL="funasr-nano-int8"
$env:SMART_TURN_MODEL="path\to\smart-turn-v3*.onnx"  # 也可填目录（自动寻找 smart-turn*.onnx）
$env:SMART_TURN_THRESHOLD="0.5"
$env:SMART_TURN_MIN_SILENCE_MS="400"   # 可选：更大=更不敏感
$env:SMART_TURN_COMMIT_MS="300"        # 可选：更大=更不敏感（会更慢确认 turn end）
cargo run --example asr_mic --features asr-sherpa,asr-mic,turn-smart --release
```

## 编译与后端选择

后端选择在运行时通过环境变量控制（`TTS_BACKEND`、`AUDIO_BACKEND`）。
特性只决定哪些后端被编译。

如果希望运行时随时切换后端，可一次性编译全部特性：

```bash
cargo build --all-features
```

运行时选择示例：

```powershell
$env:TTS_BACKEND="gpt-sovits"  # 或 "gpt-sovits-onnx" / "os"
$env:AUDIO_BACKEND="rodio"
```

## 示例

- `stream_sim`：模拟 LLM 流式增量。

```bash
cargo run --example stream_sim --features gpt-sovits
```

- `deepseek_stream`：从 OpenAI/DeepSeek 兼容 SSE 拉流。

```bash
cargo run --example deepseek_stream --features gpt-sovits
```

- `terminal_chat`：终端交互，输入新内容会取消上一轮。

```bash
cargo run --example terminal_chat --features gpt-sovits
```

- `voice_assistant`：麦克风 ASR → Smart Turn → LLM 流式 → TTS 流式（支持保守 barge-in：连续说话达到阈值才打断）。

先用 OS TTS 跑通（不依赖 GPT-SoVITS/CUDA）：

```powershell
$env:OPENAI_API_KEY="..."                 # 必填
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"   # 可选
$env:OPENAI_MODEL="deepseek-chat"         # 可选

$env:TTS_BACKEND="os"                     # 可选：先跑通建议用 os

$env:ASR_MODELS_ROOT="asrmodel"
$env:ASR_MODEL="funasr-nano-int8"
$env:ASR_VAD_MIN_SILENCE="0.45"           # 可选：更不敏感=更少切段

$env:SMART_TURN_MODEL="path\to\smart-turn-v3*.onnx"  # 可选：不填则每个 VAD 分段视为一个 turn
$env:SMART_TURN_THRESHOLD="0.5"

$env:BARGE_IN_MIN_SPEECH_MS="450"         # 可选：连续说话 >= 450ms 才触发打断（更大=更不容易误打断）
cargo run --example voice_assistant --features asr-sherpa,asr-mic,turn-smart --release
```

使用 GPU GPT-SoVITS（Windows + CUDA LibTorch）：

```powershell
$env:LIBTORCH="C:\libtorch"
$env:Path="$env:LIBTORCH\lib;$env:Path"
$env:LIBTORCH_BYPASS_VERSION_CHECK="1"    # 可选：若 torch-sys 报版本不匹配

$env:TTS_BACKEND="gpt-sovits"
$env:GSV_MODEL_DIR="v2pro"                # 默认就是 v2pro，可按需改成绝对路径
$env:AUDIO_BACKEND="rodio"
$env:AUDIO_SAMPLE_RATE="32000"
$env:AUDIO_CHANNELS="1"

cargo run --example voice_assistant --features asr-sherpa,asr-mic,turn-smart,gpt-sovits --release
```

使用 CPU ONNX GPT-SoVITS（无需 CUDA）：

```powershell
$env:TTS_BACKEND="gpt-sovits-onnx"
$env:GSV_ONNX_MODEL_DIR="onnx"
$env:GSV_ONNX_EXPORT_NAME="custom"        # 可选：与导出前缀一致
$env:AUDIO_BACKEND="rodio"
$env:AUDIO_SAMPLE_RATE="32000"
$env:AUDIO_CHANNELS="1"

cargo run --example voice_assistant --features asr-sherpa,asr-mic,turn-smart,gpt-sovits-onnx --release
```

如需运行时切换后端，编译时可改用 `--all-features`。
使用 ONNX 后端时，将特性替换为 `gpt-sovits-onnx` 并设置 `TTS_BACKEND=gpt-sovits-onnx`。

## 结构说明

- `src/streaming.rs`：`StreamSession` / `StreamControl`（会话与控制）
- `src/tokenizer.rs`：分段与 relax
- `src/pipeline.rs`：TTS 调度与播放指标
- `src/generator/`：TTS 后端（`gpt-sovits` / `gpt-sovits-onnx` / `os` / `remote` 占位）
- `src/audio/`：音频后端（`rodio` / `wasapi` 占位 / `system` 占位）
- `src/asr/`：ASR（目前：`asr-sherpa` / SenseVoice + Silero VAD）
- `src/turn/`：Turn detection（可选：Smart Turn ONNX）

更详细的架构说明见 `docs/ARCHITECTURE.md`。
流程优化建议见 `docs/OPTIMIZATIONS.md`。

## 环境变量

以下环境变量仅影响 `*_from_env` / `StreamSession::from_env` / `build_from_env` 相关路径；显式配置不会依赖环境变量。

> Windows PowerShell 设置环境变量请用 `$env:KEY="value"`；`$KEY="value"` 只是 PowerShell 变量，不会传给 `cargo run`。

### 核心

- `TTS_BACKEND`：`gpt-sovits` | `gpt-sovits-onnx` | `os` | `remote`
  - 默认：Windows + 启用 `gpt-sovits` 特性时为 `gpt-sovits`，否则 `os`
- `TTS_PARALLEL_SYNTH`：`1`/`0`（默认 `1`；支持的后端会先合成再排队播放）
- `TTS_SYNTH_INFLIGHT`：并行合成的最大任务数（默认 `1`，可适当调大）
- `TTS_BACKLOG_LIMIT`：并行合成的排队上限（默认 `32`；满则对上游施加背压）
- `TTS_SYNTH_TIMEOUT_MS`：单段合成超时（默认 `30000`；`0`=关闭；超时会跳过该段以避免卡死）
- `AUDIO_BACKEND`：`rodio`（默认）

### Audio（rodio）

- `AUDIO_RING_SECONDS`：环形缓冲时长（默认 60）
- `AUDIO_PREFILL_MS`：预填充时长（默认 50）
- `AUDIO_BUFFER_POLL_MS`：水位轮询间隔（默认 20）
- `AUDIO_SAMPLE_RATE`：输出采样率（默认 32000）
- `AUDIO_CHANNELS`：输出声道数（默认 1）

### 分段器

- `TOKENIZER_EAGER_CHUNKS`：eager 段数量（默认 1；兼容旧名：`CHUNKER_EAGER_CHUNKS`）
- 阈值可通过环境变量覆盖（单位：字符数；会自动归一化到 `min <= soft_max <= hard_max`，并 clamp 到 1–400）：
  - Eager（默认 `2/6/12`）：`TOKENIZER_EAGER_MIN_CHARS` / `TOKENIZER_EAGER_SOFT_MAX_CHARS` / `TOKENIZER_EAGER_HARD_MAX_CHARS`
  - Normal（默认 `10/20/40`）：`TOKENIZER_NORMAL_MIN_CHARS` / `TOKENIZER_NORMAL_SOFT_MAX_CHARS` / `TOKENIZER_NORMAL_HARD_MAX_CHARS`
  - Relax（默认 `20/35/80`）：`TOKENIZER_RELAX_MIN_CHARS` / `TOKENIZER_RELAX_SOFT_MAX_CHARS` / `TOKENIZER_RELAX_HARD_MAX_CHARS`
- `TOKENIZER_RELAX_BUFFER_MS`：缓存水位达到该值后放松分段（默认 200）
- `TOKENIZER_RELAX_LOG=1`：打印放松状态切换日志

放松分段依赖 `buffered_ms()` 水位（目前仅 `rodio` 支持）。

### GPT-SoVITS

- `GSV_MODEL_DIR`：模型目录（默认 `v2pro`）
- `GSV_FP16`：`1`/`0`（默认 `1`；设为 `0` 时使用 fp32 输入张量；若 CUDA TorchScript 权重为 half-only 可能会报错，可配合 `RUST_LOG=gpt_sovits_rs=debug` 查看真实原因）
- `GSV_TOP_K`：解码 top-k（默认 15）
- `GSV_FIRST_TOP_K`：首段 top-k（默认 12）
- `GSV_FIRST_CHUNK_TOKENS`：每段首块音频 token 目标（默认 10，clamp 3-25）
  - 后续块默认使用 25/50/100（逐步增大以提升连贯性与吞吐）
- `GSV_MAX_CUT_TOKEN`：`next_chunk` 最大切分 token（默认 25，clamp 25-1024）
- `GSV_TEXT_METRICS=1`：打印文本前处理 / stream 指标
- `GSV_JIEBA_BENCH=1`：额外 jieba cut 用于 profiling
- `GSV_FIRST_CHUNK_DYNAMIC`：`1`/`0`（默认 `1`，首块 token 动态调节）
- `GSV_FIRST_CHUNK_SHORT_CHARS`：短句阈值（默认 12）
- `GSV_FIRST_CHUNK_MID_CHARS`：中句阈值（默认 24）
- `GSV_FIRST_CHUNK_SHORT_TOKENS`：短句首块 token（默认 6）
- `GSV_FIRST_CHUNK_MID_TOKENS`：中句首块 token（默认 8）

### Windows（libtorch + ONNX Runtime）

- `KMP_DUPLICATE_LIB_OK=TRUE`：当同时使用 `gpt-sovits`（libtorch）与 `asr-sherpa`/`turn-smart`（ONNX Runtime）时，若遇到 OpenMP 运行时冲突或 `STATUS_HEAP_CORRUPTION`，可尝试设置该环境变量（权衡：可能影响性能/线程配置）。
- `SHERPA_STATIC_CRT=0`：仅影响 **编译阶段**。当启用 `asr-sherpa` 时，建议在 Windows 上使用动态 CRT 以降低与其它原生库（如 libtorch）混用时的崩溃概率；修改后需 `cargo clean -p sherpa-rs-sys` 重新构建。
- `SHERPA_FORCE_BUILD=1`：仅影响 **编译阶段**。跳过下载 sherpa-onnx 预编译包，强制走本地 CMake 构建（用于规避某些预编译二进制与 libtorch 混用不稳定的问题；需要本机具备 CMake + MSVC 工具链）。
- `SHERPA_ONNX_SRC`：仅影响 **编译阶段**。当 `SHERPA_FORCE_BUILD=1` 时，指定本机 `sherpa-onnx` 源码根目录（需包含 `CMakeLists.txt`）；本仓库内置的 `vendor/sherpa-rs/sherpa-rs-sys/sherpa-onnx/` 仅包含 C 头文件，无法直接用于 CMake 构建。

### GPT-SoVITS ONNX

- `GSV_ONNX_MODEL_DIR`：模型目录（默认 `onnx`）
- `GSV_ONNX_EXPORT_NAME`：导出前缀（默认 `custom`）
- `GSV_ONNX_REF_WAV`：参考音频路径（默认 `ref.wav`）
- `GSV_ONNX_REF_TEXT`：参考文本（默认读取 `ref.txt`）
- `GSV_ONNX_LANG`：`auto` | `yue`（默认 `auto`）
- `GSV_ONNX_TOP_K`：top-k（默认 4，设为 0 表示关闭）
- `GSV_ONNX_TOP_P`：top-p（默认 0.9，设为 ≥1 表示关闭）
- `GSV_ONNX_TEMPERATURE`：temperature（默认 1.0）
- `GSV_ONNX_REP_PENALTY`：repetition penalty（默认 1.35）
- `GSV_ONNX_CHUNK_SAMPLES`：推送到音频后端的分块采样数（默认 2048）
- `GSV_ONNX_BERT_PATH`：可选，自定义 `bert.onnx` 路径
- `GSV_ONNX_G2PW_PATH`：可选，自定义 `g2pW.onnx` 路径
- `GSV_ONNX_G2P_EN_PATH`：可选，自定义 `g2p_en/` 目录（需含 `encoder_model.onnx` / `decoder_model.onnx`）
- `GSV_ONNX_SV_PATH`：可选，自定义 `sv.onnx` 路径

### ASR（sherpa-rs）

- `ASR_MODELS_ROOT`：模型根目录（默认：若存在 `asrmodel/` 则为 `asrmodel`，否则为 `models`）
- `ASR_MODEL`：`paraformer-zh-small` | `paraformer-zh` | `paraformer-zh-int8` | `paraformer-trilingual` | `paraformer-en` | `sensevoice` | `sensevoice-int8` | `funasr-nano` | `funasr-nano-int8`（默认 `paraformer-zh-small`）
- `ASR_MODEL_DTYPE`：`auto` | `int8` | `fp32`（默认 `auto`；Paraformer 目录同时存在 `model.int8.onnx`/`model.onnx` 时可用来强制选择）
- `ASR_LANG`：`zh` | `en` | `ja` | `ko` | `yue` | `auto`（默认 `zh`）
- `ASR_PROVIDER`：`cpu`（默认 `cpu`；后续可扩展 `cuda/directml`）
- `ASR_THREADS`：推理线程数（默认 `2`）
- `ASR_SEGMENT_QUEUE`：VAD 分段等待推理的队列大小（默认 `8`；满则丢弃新分段）
- `ASR_VAD_PATH`：VAD 模型路径（可选；默认尝试 `<ASR_MODELS_ROOT>/silero_vad.onnx` 或 `<ASR_MODELS_ROOT>/silero_vad/silero_vad.onnx`）
- `ASR_VAD_CHUNK_MS`：内部喂给 VAD 的 chunk 毫秒数（默认 `20`；即使 `ASR_FEED_MS=0` 也会按该值分块，避免 reshape/空帧问题）
- `ASR_INFER_LOG=1`：打印每个分段的推理耗时（ms）
- `ASR_VAD_MIN_SILENCE`：端点最小静音秒数（默认 `0.25`）
- `ASR_VAD_MIN_SPEECH`：最小语音段秒数（默认 `0.1`）
- `ASR_VAD_MAX_SPEECH`：最大语音段秒数（默认 `30`）
- `ASR_VAD_THRESHOLD`：VAD 阈值（默认 `0.5`）
- `ASR_VAD_WINDOW`：VAD window size（默认 `512`）
- `ASR_VAD_BUFFER_SECONDS`：VAD 环形缓冲秒数（默认 `100`）

### Turn detection（Smart Turn）

- 特性：`turn-smart`
- `SMART_TURN_MODEL`：Smart Turn ONNX 模型路径（或包含 smart-turn*.onnx 的目录；可选，不设置则禁用；输入 16kHz mono，窗口固定 8s）
- `SMART_TURN_THRESHOLD`：端点阈值（默认 `0.5`，范围 `0.0-1.0`）
- `SMART_TURN_MIN_SILENCE_MS`：触发 Smart Turn 的最小静音时长（`voice_assistant` 示例；默认 `400`）
- `SMART_TURN_COMMIT_MS`：端点确认窗口（`voice_assistant` 示例；默认 `300`）
- `SMART_TURN_FORCE_END_MS`：静音超过该值则强制结束 turn（`voice_assistant` 示例；默认 `2000`）
- `SMART_TURN_EVAL_INTERVAL_MS`：静音期间推理间隔（`voice_assistant` 示例；默认 `200`）
- `SMART_TURN_SILENCE_ABS`：静音判定阈值（`voice_assistant` 示例；默认 `200`）

### Barge-in（voice_assistant）

- `BARGE_IN_MIN_SPEECH_MS`：连续说话达到该值才触发打断（默认 `450`）
- `BARGE_IN_SILENCE_ABS`：静音判定阈值（默认继承 `SMART_TURN_SILENCE_ABS`，未设置时为 `200`）

### 示例（stream_sim）

- `STREAM_SIM_DRAIN_MS`：发送结束后等待播放完成的最长时长（默认 10000）

### 示例（asr_file）

- `ASR_WAV`：wav 路径（也可用命令行第 1 个参数）
- `ASR_FEED_MS`：每次喂给 ASR 的 chunk 时长（默认 20；设为 0 表示一次性喂入）
- `ASR_METRICS=1`：打印时延/RTF 统计与每段 lag
- `ASR_REF_FILE`：参考文本文件路径（可选，用于计算 CER）
- `ASR_REF_TEXT`：参考文本（可选，用于计算 CER）

### 示例（asr_mic）

- `ASR_MIC_DEVICE`：输入设备名关键字（可选；不设置则用默认输入设备）
- `ASR_MIC_BUFFER_FRAMES`：cpal 输入 buffer frames（可选；部分设备不支持固定值）
- `ASR_MIC_RING_SECONDS`：输入环形缓冲秒数（默认 8）
- `ASR_FEED_MS`：每次喂给 ASR 的 chunk 时长（默认 20）

### 主程序（内置模拟流，仅用于本地验证）

- `LLM_ROUNDS`：轮数（默认 2）
- `LLM_SIM_TEXT`：模拟文本
- `LLM_SIM_CHUNK_CHARS`：每次发送字符数（默认 3）
- `LLM_SIM_DELAY_MS`：发送间隔（默认 80）
- `AUTO_CANCEL_DELAY_MS`：第 3 轮自动取消（默认 1500）

说明：以上仅作用于 `cargo run` 的内置模拟流。真实对话请使用 examples。

### LLM 示例

- `OPENAI_BASE_URL`：API Base（默认 `https://api.deepseek.com/v1`）
- `OPENAI_API_KEY`：API Key（必填）
- `OPENAI_MODEL`：模型名（默认 `deepseek-chat`）

## 备注

- GPT-SoVITS 后端仅支持 Windows + CUDA LibTorch。
- GPT-SoVITS ONNX 后端基于 ONNX Runtime，CPU 推理，参考音频需为单声道 16-bit PCM。
- `StreamControl::interrupt()` 是“打断当前轮次”的行为，不提供恢复；需要重新发起流或新建会话。
- Rodio 后端使用 `crossbeam-queue` 实现无锁缓冲。

## 维护说明（vendored sherpa-rs）

`asr-sherpa` 使用了 vendored 的 `vendor/sherpa-rs/`，目的是在上游 crate 尚未覆盖/验证最新模型与 C-API 版本时，确保我们可以稳定 pin 住 `sherpa-onnx` C-API 与下载产物。

一旦 `sherpa-rs` 上游 bindings 更新并满足我们需求，优先切回官方依赖（减少维护成本）。如果我们后续改动变多，建议把 vendored 目录改为 git submodule / fork 分支，并考虑向上游提 PR。

详见：`vendor/sherpa-rs/UPSTREAM.md`
