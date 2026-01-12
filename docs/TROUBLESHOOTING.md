# 故障排查 (Troubleshooting)

本文档记录常见问题和解决方案。

## 平台说明

本文档主要针对 **Windows** 平台。Linux/macOS 用户可跳过 Windows 特定章节。

---

## Windows 问题

### STATUS_HEAP_CORRUPTION 崩溃

**现象:**
- 进程直接退出，报错 `0xc0000374 STATUS_HEAP_CORRUPTION`
- 常见于 GPT-SoVITS 初始化阶段

**原因:**
多个原生运行时混用 (libtorch/CUDA + ONNX Runtime) 导致内存冲突。

**解决方案:**

| 方案 | 说明 | 推荐度 |
|------|------|-------|
| 使用 `gpt-sovits-onnx` | 不加载 libtorch | ⭐⭐⭐ |
| 进程隔离 TTS Worker | HTTP 接口隔离 | ⭐⭐⭐ |
| `KMP_DUPLICATE_LIB_OK=TRUE` | 可能缓解，不保证 | ⭐ |

**进程隔离示例:**
```powershell
# 终端 1: 启动 Worker
$env:RCAT_MODELS_DIR = "F:\\github\\rcat\\models"
cargo run --bin tts_worker --features tts-worker --release

# 终端 2: 主应用
$env:TTS_BACKEND = "remote"
$env:TTS_REMOTE_BASE_URL = "http://127.0.0.1:7878"
```

相关代码: [voice_assistant.rs](../examples/voice_assistant.rs)

---

### GPT-SoVITS FP16 报错

**现象:**
```
RuntimeError: Input type (CUDAFloatType) and weight type (CUDAHalfType) should be the same
```

**原因:**
TorchScript 权重为 half-only (fp16)，不支持 fp32 输入。

**解决方案:**
保持 `GSV_FP16=1` (默认)。

如需调试:
```powershell
$env:RUST_LOG = "gpt_sovits_rs=debug"
```

---

### Smart Turn CPU/GPU 选择

**配置方式:**
```powershell
$env:SMART_TURN_VARIANT = "gpu"  # 或 "cpu"
```

**注意:** `smart-turn-rs` 当前只支持 CPU execution provider。GPU 模型仍会在 CPU 上运行。

**建议:** 使用 CPU 版模型 (~8MB)，推理 <5ms。

---

## 跨平台问题

### 首轮响应慢

**现象:** 首次对话延迟明显高于后续。

**原因:** 模型冷启动。

**解决方案:** 已实现模型预热，检查日志:
```
smart_turn warmup complete
tts warmup complete
```

相关代码: [voice_assistant.rs:237-257](../examples/voice_assistant.rs#L237-L257)

---

### 误打断频繁

**现象:** 助手说话时频繁被打断，即使用户没说话。

**原因:** VAD 对环境噪声敏感（误判 SpeechStart）或确认窗口过短。

**解决方案:**
```powershell
$env:BARGE_IN_CONFIRM_MS = "150"       # 增大确认窗口
$env:BARGE_IN_MIN_SPEECH_MS = "600"    # 增大触发时长
```

相关代码: [voice_assistant.rs:168-173](../examples/voice_assistant.rs#L168-L173)

---

### LLM 延迟不稳定

**现象:** 不同轮次的 LLM 响应时间差异大。

**原因:** 每轮新建连接导致 TTFB 波动。

**解决方案:** 确认 LLM Client 复用生效。查看日志只有一次连接建立。

相关代码: [voice_assistant.rs:70-74](../examples/voice_assistant.rs#L70-L74)

---

### TTS 音频断断续续

**现象:** 语音播放有明显停顿。

**原因:** 环形缓冲区满导致阻塞。

**解决方案:**

1. 启用指标:
```powershell
$env:AUDIO_RING_METRICS = "1"
```

2. 观察日志 `ring_buffer: blocked Xus`

3. 如阻塞频繁，调整缓冲:
```powershell
$env:AUDIO_RING_SECONDS = "90"
$env:AUDIO_PREFILL_MS = "100"
```

相关代码: [rodio.rs:406-407](../src/audio/rodio.rs#L406-L407)

---

### ASR 识别不准

**现象:** 语音识别错误率高。

**可能原因:**
1. VAD 分段不准
2. 模型不匹配

**解决方案:**

调整 VAD:
```powershell
$env:ASR_VAD_MIN_SILENCE = "0.5"   # 增大
$env:ASR_VAD_THRESHOLD = "0.6"    # 增大
```

切换模型:
```powershell
$env:ASR_MODEL = "funasr-nano-int8"  # 推荐
```

---

### 环境变量不生效

**现象:** 设置了环境变量但程序行为没变。

**原因 (Windows):** 使用了 PowerShell 变量语法。

**错误:**
```powershell
$KEY = "value"  # 这是 PowerShell 变量，不传给子进程
```

**正确:**
```powershell
$env:KEY = "value"  # 这是环境变量
```

---

### 模型路径错误

**现象:**
```
SMART_TURN_MODEL not found: path/to/model.onnx
```

**解决方案:**
1. 确认路径存在
2. Windows 使用反斜杠或正斜杠都可以
3. 可以指定目录，自动查找 `smart-turn*.onnx`
4. 推荐仅设置 `RCAT_MODELS_DIR=/path/to/models`，并按 `models/TURN/` 放置 smart-turn 模型（可配合 `SMART_TURN_VARIANT=cpu|gpu` 选择）

---

## 调试开关

| 变量 | 用途 |
|------|------|
| `RUST_LOG=rcat_voice=debug` | 本项目 debug 日志 |
| `RUST_LOG=gpt_sovits_rs=debug` | GPT-SoVITS 上游库 |
| `ORT_LOG=warning` | ONNX Runtime 日志级别（默认 warning；设为 info/verbose 会很吵） |
| `AUDIO_RING_METRICS=1` | Ring Buffer 阻塞指标 |
| `VOICE_STREAM_METRICS=1` | LLM/TTS 推导指标（TracingMetricsSink） |
| `VOICE_TTS_METRICS=1` | TTS/流式推导指标（TracingMetricsSink） |
| `STREAM_METRICS=1` | 兼容旧开关（等价于启用 TracingMetricsSink） |
| `ASR_INFER_LOG=1` | ASR 推理耗时 |
| `VOICE_METRICS_MAX_TURNS=1024` | 限制 TracingMetricsSink 的 turn 状态数量 |
| `TOKENIZER_RELAX_LOG=1` | 分句 Relax 状态 |

---

> 启用 `VOICE_STREAM_METRICS=1`/`VOICE_TTS_METRICS=1` 后，默认会输出 `tts_ttfa_ms`（音频生成延迟：首段送入 TTS→首音）与 `e2e_ttfa_ms`（端到端延迟：用户说完→首音）。

## 相关文档

- [README.md](./README.md) - 文档索引
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系统架构
- [FEATURE_MAP.md](./FEATURE_MAP.md) - 功能-代码映射
