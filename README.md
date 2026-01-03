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

## 快速开始（Windows + CUDA GPT-SoVITS）

1) 安装 Rust（MSVC 工具链）。
2) 安装 CUDA LibTorch（2.9 版）并在当前终端设置环境变量：

```powershell
$env:LIBTORCH="C:\libtorch"
$env:Path="$env:LIBTORCH\lib;$env:Path"
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

如需运行时切换后端，编译时可改用 `--all-features`。
使用 ONNX 后端时，将特性替换为 `gpt-sovits-onnx` 并设置 `TTS_BACKEND=gpt-sovits-onnx`。

## 结构说明

- `src/streaming.rs`：`StreamSession` / `StreamControl`（会话与控制）
- `src/tokenizer.rs`：分段与 relax
- `src/pipeline.rs`：TTS 调度与播放指标
- `src/generator/`：TTS 后端（`gpt-sovits` / `gpt-sovits-onnx` / `os` / `remote` 占位）
- `src/audio/`：音频后端（`rodio` / `wasapi` 占位 / `system` 占位）

更详细的架构说明见 `docs/ARCHITECTURE.md`。

## 环境变量

以下环境变量仅影响 `*_from_env` / `StreamSession::from_env` / `build_from_env` 相关路径；显式配置不会依赖环境变量。

### 核心

- `TTS_BACKEND`：`gpt-sovits` | `gpt-sovits-onnx` | `os` | `remote`
  - 默认：Windows + 启用 `gpt-sovits` 特性时为 `gpt-sovits`，否则 `os`
- `TTS_PARALLEL_SYNTH`：`1`/`0`（默认 `1`；支持的后端会先合成再排队播放）
- `TTS_SYNTH_INFLIGHT`：并行合成的最大任务数（默认 `1`，可适当调大）
- `AUDIO_BACKEND`：`rodio`（默认）

### Audio（rodio）

- `AUDIO_RING_SECONDS`：环形缓冲时长（默认 60）
- `AUDIO_PREFILL_MS`：预填充时长（默认 50）
- `AUDIO_BUFFER_POLL_MS`：水位轮询间隔（默认 20）
- `AUDIO_SAMPLE_RATE`：输出采样率（默认 32000）
- `AUDIO_CHANNELS`：输出声道数（默认 1）

### 分段器

- `CHUNKER_EAGER_CHUNKS`：首段短切数量（默认 2）
- `TOKENIZER_MIN_CHARS`：最小字符数（默认 20）
- `TOKENIZER_MAX_CHARS`：常规最大字符数（默认 50）
- `TOKENIZER_BOUNDARY_OVERFLOW`：等待边界的额外字符数（默认 20）
- `TOKENIZER_RELAX_BUFFER_MS`：缓存水位达到该值后放松分段（默认 200）
- `TOKENIZER_RELAX_SCALE`：放松倍率（默认 1.5）
- 说明：`relaxed_max = TOKENIZER_MAX_CHARS * TOKENIZER_RELAX_SCALE`，上限为 120
- `TOKENIZER_RELAX_BOUNDARY_WINDOW`：优先边界窗口（默认 24）
- `TOKENIZER_RELAX_OVERFLOW`：放松模式下的额外超出（默认 30）
- `TOKENIZER_RELAX_LOG=1`：打印放松状态切换日志

放松分段依赖 `buffered_ms()` 水位（目前仅 `rodio` 支持）。

### GPT-SoVITS

- `GSV_MODEL_DIR`：模型目录（默认 `v2pro`）
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

### 示例（stream_sim）

- `STREAM_SIM_DRAIN_MS`：发送结束后等待播放完成的最长时长（默认 10000）

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
- `StreamControl::pause()` 是中断行为，不提供恢复；需要重新发起流或新建会话。
- Rodio 后端使用 `crossbeam-queue` 实现无锁缓冲。
