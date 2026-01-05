# 流式语音交互集成方案

## 目标

将 `rcat-voice` 的流式 ASR+LLM+TTS 管线接入 Tauri 前端，实现：
1. **语音输入**：用户通过麦克风说话 → ASR 识别 → 发送到 LLM
2. **语音输出**：LLM 流式回复 → 分段器 → TTS 合成 → 播放
3. **打断机制**：用户可随时打断 AI 回复

---

## 当前状态

| 组件 | 状态 | 位置 |
|:---|:---:|:---|
| TTS 引擎 | ✅ 已集成 | `src-tauri/src/services/voice.rs` |
| 流式分段器 | ✅ rcat-voice | `rcat-voice/src/tokenizer.rs` |
| 流式管线 | ✅ rcat-voice | `rcat-voice/src/pipeline.rs` |
| ASR 模块 | ✅ rcat-voice | `rcat-voice/src/asr/sherpa.rs` |
| 麦克风采集 | ❌ 未实现 | 需要新增 |
| 前端语音 UI | ❌ 未实现 | 需要新增 |

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  MicButton   │  │  VoiceStatus │  │  ChatMessages    │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │ start/stop                                         │
└─────────┼───────────────────────────────────────────────────┘
          │ Tauri Command
┌─────────▼───────────────────────────────────────────────────┐
│                    Backend (Rust/Tauri)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  VoiceConversation                    │   │
│  │  ┌─────────┐   ┌─────────┐   ┌───────────────────┐   │   │
│  │  │   ASR   │──▶│   LLM   │──▶│ StreamSession(TTS)│   │   │
│  │  │ (Sherpa)│   │(OpenAI) │   │ (Tokenizer+Pipeline)│   │   │
│  │  └────▲────┘   └─────────┘   └───────────────────┘   │   │
│  │       │                                               │   │
│  │  ┌────┴────┐                                          │   │
│  │  │   Mic   │ (cpal/WebRTC Audio)                      │   │
│  │  │ Capture │                                          │   │
│  │  └─────────┘                                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## User Review Required

> [!IMPORTANT]
> **麦克风采集方案选择**
>
> 有两种实现方式，请确认偏好：
>
> | 方案 | 优点 | 缺点 |
> |:---|:---|:---|
> | **A: 后端 cpal** | 低延迟、与 ASR 同线程 | 需要系统音频权限 |
> | **B: 前端 WebRTC** | 浏览器权限管理、跨平台 | 需要 PCM 传输到后端 |
>
> 推荐 **方案 A**（后端 cpal），与 `rcat-voice` 现有架构一致。

> [!IMPORTANT]
> **Smart Turn 使用**
>
> 是否启用 Smart Turn 模型进行智能轮次检测？
> - 如果启用：需要额外加载 ONNX 模型
> - 如果不启用：使用简单的静音检测

---

## Proposed Changes

### 1. Backend - 新增麦克风采集模块

#### [NEW] [mic.rs](file:///e:/rcat/rcat-voice/src/audio/mic.rs)

麦克风采集，输出 PCM 流到 mpsc channel：
- 使用 `cpal` 库
- 支持选择输入设备
- 输出 16kHz mono i16 PCM

---

### 2. Backend - 新增语音对话模块

#### [NEW] [voice_conversation.rs](file:///e:/rcat/src-tauri/src/services/voice_conversation.rs)

组合 ASR + LLM + TTS 的完整对话流程：
```rust
pub struct VoiceConversation {
    asr: SherpaAsrStream,
    tts_engine: Arc<dyn TtsEngine>,
    // ...
}

impl VoiceConversation {
    pub async fn start(&mut self) -> Result<()>;
    pub async fn stop(&mut self) -> Result<()>;
}
```

#### [MODIFY] [services/mod.rs](file:///e:/rcat/src-tauri/src/services/mod.rs)

添加 `voice_conversation` 模块。

#### [MODIFY] [lib.rs](file:///e:/rcat/src-tauri/src/lib.rs)

注册新的 Tauri commands：
- `voice_conversation_start`
- `voice_conversation_stop`
- `voice_conversation_status`

---

### 3. Backend - Tauri Events

新增事件用于前端状态同步：
- `voice-asr-result`: ASR 识别结果
- `voice-tts-speaking`: TTS 开始播放
- `voice-conversation-state`: 对话状态变化

---

### 4. Frontend - 语音交互 UI

#### [NEW] [MicButton.tsx](file:///e:/rcat/src/components/MicButton.tsx)

麦克风按钮组件：
- 按住说话 / 点击开关模式
- 显示录音状态动画

#### [NEW] [VoiceStatusIndicator.tsx](file:///e:/rcat/src/components/VoiceStatusIndicator.tsx)

语音状态指示器：
- Listening / Processing / Speaking 状态
- ASR 实时文字预览

#### [MODIFY] [services/voice.ts](file:///e:/rcat/src/services/voice.ts)

添加语音对话相关函数：
```typescript
export const voiceConversationStart = async (): Promise<void>;
export const voiceConversationStop = async (): Promise<void>;
```

---

## Verification Plan

### Automated Tests

1. **rcat-voice 单元测试**
   ```bash
   cd rcat-voice && cargo test
   ```

2. **Tauri 后端编译检查**
   ```bash
   cd src-tauri && cargo check
   ```

### Manual Verification

1. **麦克风采集测试**
   - 启动应用，点击麦克风按钮
   - 对着麦克风说话，确认控制台有 ASR 输出

2. **端到端语音对话测试**
   - 说"你好"，等待 AI 回复
   - 确认 TTS 正常播放
   - 尝试在 AI 回复时打断

---

## 实施顺序

1. ✅ 确认架构方案（需用户输入）
2. 新增 `mic.rs` 麦克风采集
3. 新增 `voice_conversation.rs` 组合模块
4. 修改 `lib.rs` 注册 commands
5. 新增前端组件
6. 集成测试
