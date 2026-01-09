# 优化建议 (rcat-voice)

本文档记录已完成的优化、待改进项和已知设计缺陷。

## 优先级说明

| 级别 | 含义 | 时间框架 |
|------|------|---------|
| P0 | 直接影响稳定低延迟 | 立即 |
| P1 | 体验提升明显 | 短期 |
| P2 | 长期改进/架构调整 | 规划中 |

---

## 已完成优化

### P0.2 LLM Client 复用

**问题**: 每轮对话新建 HTTP 客户端，引入不稳定的 TTFB。

**解决**: 在 `main()` 创建共享 `Arc<Client<OpenAIConfig>>`，复用于所有 turn。

| 项目 | 内容 |
|------|------|
| 代码 | `voice_assistant.rs:70-74` |
| 收益 | 复用 TCP/TLS 连接，稳定 TTFB |

---

### P0.1 Barge-in 双门机制

**问题**: 纯能量阈值受麦克风增益/噪声影响，容易误打断。

**解决**: 双门检测：Gate 1 能量阈值 + Gate 2 确认窗口。

| 项目 | 内容 |
|------|------|
| 代码 | `voice_assistant.rs:339-380` |
| 配置 | `BARGE_IN_CONFIRM_MS=100` |

---

### P1.1 模型预热

**问题**: Smart Turn / TTS 首次推理有冷启动延迟。

**解决**: 启动时做 dummy 推理。

| 项目 | 内容 |
|------|------|
| 代码 | `voice_assistant.rs:237-257` |
| 验证 | 日志 `smart_turn warmup complete` |

---

### P1.2 指标可观测性

**解决**: 添加 `turn_to_finish_ms` 和 Ring Buffer 指标。

| 项目 | 内容 |
|------|------|
| 代码 | `voice_assistant.rs:319`, `rodio.rs:406-407` |
| 启用 | `AUDIO_RING_METRICS=1` |

---

## 待改进项

### P0.3 VAD 替代能量阈值

**问题**: 能量阈值受环境影响大。

**现状**: 双门机制使用能量阈值 + 确认窗口。

**建议**: 复用 `SherpaAsrStream` 中的 Silero VAD 状态，替代能量阈值。

---

### P0.4 LLM 请求取消

**问题**: Barge-in 时无法中止 LLM 请求，只能关闭 channel 等待超时。

**现状**: 取消后 `StreamSession.shutdown()` 会发送 `cancel_tx.send(true)`，但 LLM 侧仍在产生 delta，占用网络/连接资源。

**影响**: 资源浪费，**拖累下一轮响应的 TTFA**。

**缓解措施**:
- **应用层**: 收到 cancel 后**立即丢弃**后续 delta，并停止消费 channel (Drain & Drop)。
- **网络层**: 尝试 Drop response body trigger TCP RST (依赖实现)。
- **根本解决**: HTTP/2 CANCEL 或 WebSocket close。

---

### P0.5 ONNX 推理阻塞 Reactor

**问题**: VAD/Smart Turn ONNX 推理未使用 `spawn_blocking`，高负载时阻塞 tokio reactor。

**代码确认**: VAD 推理在 tokio task 中 (`sherpa.rs:481 tokio::spawn`)，但 `vad.accept_waveform()` 是同步调用。

**影响** (升级为**必须修复**): 
- 不仅是 P95/P99 抖动
- **拖慢 delta channel 接收**
- **影响音频 push 等关键路径**

**建议**: 将 ONNX 推理迁移到 `spawn_blocking` 或专用线程池。

---

### P1.1 TTS (ONNX) 推理阻塞 Reactor

**问题**: `gpt-sovits-onnx` 后端同样运行在 tokio task 中，可能导致 reactor starvation。
**建议**: 同上，迁移至 `spawn_blocking`。

---

### P0.6 代际串台 (Generation Leak)

**问题**: `CancelScope` 只在 `push_blocking` 时检查 epoch (`rodio.rs:425`)，但 Pipeline 队列中的 Segment 没有携带 `generation_id`。

**串台场景**:
1. 用户打断，cancel 发送
2. TTS 正在合成的句子完成
3. 新轮次开始
4. 旧句子音频 push 到新轮次的 RingBuffer

**建议**: 为 LLM delta、TTS segment、PCM frame、RMS 都携带 `generation_id`。

> **修复方向**: 引入 `generation_id`，在 RingBuffer.push() 处做代际过滤，拒绝旧代际的写入。

---

### P2 增强: 区分 write-domain 与 play-domain 时间戳

**背景**: 当前 `first_audio_ts` 是"估算首播时间"(play-domain)，适用于计算 E2E_TTFA。但诊断 RingBuffer/prefill/rodio 调度问题时，缺少"首次写入时刻"(write-domain)。

**建议**:
- 新增 `first_write_ts: Option<Instant>` — 首次 push 成功写入任意样本的时刻
- 考虑将 `first_audio_ts` 改名为 `first_play_ts_est` 以减少误用

**诊断收益**:
- "TTS 慢" → `first_write_ts` 也晚
- "播放链路慢/抖" → `first_write_ts` 早但 `first_play_ts_est` 晚或不稳定

---

### P1.3 RMS 采样点对齐

**问题**: RMS 在**写入 RingBuffer 端**采样，而非**播放端**，唇形动画有 prefill 提前量 (~50ms)。

**现状**: 唇形略微超前于声音。

**建议**: 在 Rodio `Source::next()` 回调中采样，或添加 playback latency 补偿。

---

### P2.1 取消语义统一

**问题**: 当前存在 `watch<bool>` 和 `epoch` 两套机制。
**建议**: 废弃 `watch<bool>`，将 `epoch` (Generation ID) 提升为唯一权威。所有组件只响应携带匹配 `generation_id` 的事件。

---

### P2.2 会话状态机

**问题**: 无显式状态机，状态隐含在 `Option<RunningAssistant>`。

**建议**: 定义 `enum SessionState { Idle, Listening, Thinking, Speaking, Interrupted, Error }`。

---

### P2.3 故障降级

**问题**: 音频设备被占用时 panic，无降级策略。

**建议**: TTS 失败时 fallback 到 OS TTS 或文本输出。

---

## 已知设计缺陷

| 缺陷 | 现状 | 影响 | 短期缓解 |
|------|------|------|---------|
| VAD/Turn 阻塞 reactor | 未用 spawn_blocking | **拖慢 delta/音频关键路径** | 无 |
| 代际串台 | Segment 无 generation_id | 旧音频写入新轮次 | 无 |
| LLM 请求无法取消 | 关闭 channel | **拖累下一轮 TTFA** | 无 |
| ASR segment 队列满丢弃 | `try_send` 丢弃新段 | 可能丢识别 | 增大 `ASR_SEGMENT_QUEUE` |
| first_audio_ts 非播放时间 | 写入端时间戳 | 指标不精确 | +50-70ms 估算 |
| RingBuffer busy-wait | **Polling + Backoff Sleep** | CPU 争抢 | 改用 crossbeam/condvar |

> **RingBuffer 澄清** (`rodio.rs:435-438`):
> ```rust
> let delay_us = 100u64 << backoff.min(3);  // 100us, 200us, 400us, 800us
> std::thread::sleep(Duration::from_micros(delay_us));
> ```
> 这是 exponential backoff + sleep，不是纯 busy-wait。但 `std::thread::sleep` 在音频线程中可能不精确，建议改用条件变量。

---

## ASR 丢弃可观测性

**已有部分可观测性** (`sherpa.rs:667-676`):

```rust
*dropped_segments = dropped_segments.saturating_add(1);
if now.duration_since(*last_drop_log) >= Duration::from_secs(1) {
    warn!("asr: dropped {} segments (segment queue full)", *dropped_segments);
}
```

**缺失项**:
- [ ] 丢弃段的时长
- [ ] CPU/RTF 指标
- [ ] 策略开关 (当前硬编码丢弃新段)

---

## 关键指标

| 指标 | 定义 | 测量方法 | 期望值 |
|------|------|---------|-------|
| E2E_TTFA | 用户说完→首音播放 | `turn_end_ts` → `first_audio_ts` + prefill | < 800ms |
| LLM_TTFB | LLM 首字节 | `mark_llm_start()` → 首个 delta | 200-500ms |
| first_audio_ts | 首个样本**写入** | `TtsMetrics` | - |
| ring_blocked_us | RingBuffer 阻塞 | 日志 | < 1000us |

> **注意**: `first_audio_ts` 是写入时间，真正播放时间需 +`AUDIO_PREFILL_MS` (默认 50ms)

---

## 配置参数

### Barge-in

| 变量 | 默认值 | 调优 |
|------|-------|------|
| `BARGE_IN_MIN_SPEECH_MS` | 450 | 增大减少误打断 |
| `BARGE_IN_CONFIRM_MS` | 100 | 增大提高准确性 |
| `BARGE_IN_SILENCE_ABS` | 200 | 降低提高灵敏度 |

### Smart Turn

| 变量 | 默认值 | 调优 |
|------|-------|------|
| `SMART_TURN_THRESHOLD` | 0.5 | 增大减少提前结束 |
| `SMART_TURN_MIN_SILENCE_MS` | 400 | 增大更稳定 |

### 可观测性

| 变量 | 默认值 | 用途 |
|------|-------|------|
| `AUDIO_RING_METRICS` | off | Ring 指标 |
| `VOICE_TTS_METRICS` | off | TTS 时间线 |

---

## 相关文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系统架构 (含已知限制)
- [FEATURE_MAP.md](./FEATURE_MAP.md) - 功能-代码映射
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - 故障排查
