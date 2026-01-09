# rcat-voice 功能-代码映射

本文档提供 AI 友好的功能-代码映射，帮助通过自然语言描述快速定位代码位置。

## 使用方法

当你想修改某个功能时，告诉 AI：
- "我想修改打断检测的灵敏度"
- "TTS的声音质量需要调整"

AI 会根据映射表定位代码并提供修改方案。

---

## 功能模块总览

| 模块 | 职责 | 代码目录 | Feature |
|------|------|---------|---------|
| TTS 引擎 | 文本转语音合成 | `src/generator/` | `gpt-sovits`, `gpt-sovits-onnx` |
| 音频播放 | PCM 写入环形缓冲并播放 | `src/audio/` | `audio-rodio` |
| 音频录制 | 麦克风采集 | `src/audio/mic.rs` | `asr-mic` |
| 语音识别 | 语音转文字 | `src/asr/` | `asr-sherpa` |
| 端点检测 | 判断用户是否说完 | `src/turn/` | `turn-smart` |
| 流式会话 | 管理 LLM → TTS 流程 | `src/streaming.rs` | - |
| TTS 管道 | 调度合成任务 | `src/pipeline.rs` | - |
| 文本分句 | 切分文本为合成单元 | `src/tokenizer.rs` | - |

---

## TTS 引擎

负责将文本转换为语音。支持多种后端，通过 `TTS_BACKEND` 环境变量切换。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| TTS 引擎接口 | 语音合成, speak | `src/generator/mod.rs` | `TTS_BACKEND` |
| GPT-SoVITS CUDA | 克隆语音, AI声音 | `src/generator/gpt_sovits.rs` | `GSV_MODEL_DIR` |
| GPT-SoVITS ONNX | CPU语音合成 | `src/generator/gpt_sovits_onnx.rs` | `GSV_ONNX_MODEL_DIR` |
| OS TTS | 系统语音, SAPI | `src/generator/os.rs` | - |
| Remote TTS | 远程语音, HTTP | `src/generator/remote.rs` | `TTS_REMOTE_BASE_URL` |

### 常用操作

**切换 TTS 后端:**
```powershell
$env:TTS_BACKEND = "gpt-sovits-onnx"  # 或 gpt-sovits / os / remote
```

**修改声音参数:** 编辑对应后端的 `Config` 结构体或环境变量。

---

## 音频系统

### 音频播放

将 PCM 样本写入环形缓冲区，通过 Rodio 播放。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 音频后端接口 | 播放器, backend | `src/audio/mod.rs` | `AUDIO_BACKEND` |
| Rodio 播放 | 扬声器输出 | `src/audio/rodio.rs` | `AUDIO_SAMPLE_RATE` |
| 环形缓冲 | RingBuffer | `src/audio/rodio.rs:404-460` | `AUDIO_RING_SECONDS` |
| RMS 遥测 | 唇形同步 | `src/audio/mod.rs:RmsPayload` | - |

### 音频录制

采集麦克风输入。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 麦克风输入 | mic, 录音 | `src/audio/mic.rs` | `MIC_DEVICE` |
| 输入配置 | 采样率 | `MicConfig` | `ASR_FEED_MS` |

---

## 语音识别

通过 Sherpa-ONNX 将语音转为文字。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| ASR 接口 | 语音识别, STT | `src/asr/mod.rs` | `ASR_MODEL` |
| Sherpa 实现 | ONNX推理 | `src/asr/sherpa.rs` | `ASR_MODELS_ROOT` |
| VAD 配置 | 语音活动 | `SherpaVadConfig` | `ASR_VAD_THRESHOLD` |

### 常用操作

**切换 ASR 模型:**
```powershell
$env:ASR_MODEL = "funasr-nano-int8"  # 推荐
```

---

## 端点检测

判断用户是否说完，触发 LLM 请求。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| Smart Turn | 端点概率, turn end | `src/turn/smart_turn.rs` | `SMART_TURN_MODEL` |
| 阈值配置 | 灵敏度 | `SmartTurnDetector` | `SMART_TURN_THRESHOLD` |

---

## 打断检测 (Barge-in)

检测用户在助手说话时的打断行为。使用双门机制降低误打断。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 双门检测 | 打断, 插话 | `voice_assistant.rs:339-380` | - |
| Gate 1 能量 | 粗门 | `is_silence_chunk()` | `BARGE_IN_SILENCE_ABS` |
| Gate 2 确认 | 确认窗口 | `consecutive_non_silence_ms` | `BARGE_IN_CONFIRM_MS` |
| 触发阈值 | 语音时长 | `speech_streak_ms` | `BARGE_IN_MIN_SPEECH_MS` |

### 常用操作

**降低误打断:**
```powershell
$env:BARGE_IN_CONFIRM_MS = "150"      # 增大确认窗口
$env:BARGE_IN_MIN_SPEECH_MS = "600"   # 增大触发时长
```

---

## 流式会话

管理从 LLM 增量到 TTS 播放的完整流程。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 会话管理 | StreamSession | `src/streaming.rs` | - |
| 控制接口 | StreamControl | `StreamControl` | - |
| 取消/打断 | interrupt, cancel | `SessionCancel` | - |

---

## TTS 管道

调度 TTS 合成任务，支持并行。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 管道调度 | Pipeline | `src/pipeline.rs` | `TTS_PARALLEL` |
| 并行合成 | run_parallel | `Pipeline.run_parallel()` | `TTS_SYNTH_INFLIGHT` |

---

## 文本分句

将 LLM 增量切分为适合 TTS 合成的句子。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 分句器 | Tokenizer | `src/tokenizer.rs` | `TOKENIZER_*` |
| Eager 模式 | 快速首段 | `TokenizerConfig` | `TOKENIZER_EAGER_*` |
| Relax 模式 | 大段合成 | `TOKENIZER_RELAX_*` | `TOKENIZER_RELAX_BUFFER_MS` |

### 常用操作

**调整分句阈值:**
```powershell
$env:TOKENIZER_NORMAL_MIN_CHARS = "15"
$env:TOKENIZER_NORMAL_SOFT_MAX_CHARS = "30"
```

---

## 延迟指标

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| 轮次延迟 | turn_to_finish | `voice_assistant.rs:319` | 日志输出 |
| Ring 指标 | 缓冲阻塞 | `rodio.rs:406-407` | `AUDIO_RING_METRICS=1` |
| TTS 指标 | 时间线 | `TtsMetrics` | `VOICE_TTS_METRICS=1` |

---

## 模型预热

消除首轮推理延迟。

| 功能 | 别名 | 代码位置 | 配置 |
|------|------|---------|------|
| Smart Turn 预热 | warmup | `voice_assistant.rs:239-250` | 自动 |
| TTS 预热 | warmup | `voice_assistant.rs:253-256` | 自动 |

---

## 相关文档

- [README.md](./README.md) - 文档索引
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系统架构
- [OPTIMIZATIONS.md](./OPTIMIZATIONS.md) - 优化建议
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - 故障排查
