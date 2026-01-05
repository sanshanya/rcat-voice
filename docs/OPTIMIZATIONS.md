# 优化建议（rcat-voice）

本文档只讨论“面向个人麦克风”的低延迟语音交互场景：**Mic → ASR → Turn End → LLM(流式) → TTS(流式)**。
目标是降低端到端时延、减少误打断、让轮次更自然，并保持实现简单可控。

> 对应参考实现：`examples/voice_assistant.rs`（端到端 Demo）。

---

## 现状摘要（已具备）

- **TTS 流式管线**：`StreamSession`（delta → tokenizer → pipeline → TTS → audio）
- **ASR（离线模型 + 在线喂入）**：`SherpaAsrStream`（输入 PCM → 重采样/混音 → Silero VAD 分段 → 转写）
- **Smart Turn**：在静音期间推理，输出 turn end 概率（`turn-smart` 特性）
- **保守 barge-in（打断）**：在助手说话时，要求“连续非静音语音达到阈值”才触发取消

---

## 关键瓶颈 / 风险点（按对体验影响排序）

1) **ASR 段队列溢出策略 / 可观测性**
- 当前实现已改为：VAD 出段 → **有界 segment queue** → **单 blocking 推理循环**（避免每段 `spawn_blocking` 抖动，并把回压边界清晰化）。
- 仍需关注：队列满时的策略（例如丢弃新段/丢弃旧段/只保留最新段）以及 dropped 统计与日志阈值，避免“静默丢段”导致体验变差却难以定位。

2) **barge-in / Smart Turn gate 依赖振幅阈值**
- 纯能量阈值在不同麦克风增益、噪声底、AEC/回声条件下不稳定。
- 外放时可能“听到自己说话”，误触发 barge-in 或把 TTS 送回 ASR 造成自激。

3) **LLM 客户端每个 turn 重建**
- 每轮对话新建 client/连接，会引入不稳定的 TTFB/首字延迟。

4) **音频 ring 写入存在 busy-wait**
- `rodio` ring 满时 `sleep(1ms)` 自旋式等待，长尾情况下会增大 CPU 抖动，并把“取消/暂停”的响应变慢。

---

## 优化建议（路线图）

### P0（优先做，直接影响稳定低延迟）

**P0.1 解耦 ASR ingestion 与识别（已实现）**
- 已拆分为两条链路：PCM → VAD 分段（async）与“语音段”→ 识别（blocking loop），中间使用有界队列。
- 可继续做：根据使用场景选择更合适的溢出策略，并把 dropped 指标纳入常规日志/metrics（方便线上调参）。

**P0.2 用 VAD 决策替代振幅阈值（barge-in + Smart Turn gate）**
- Smart Turn 官方建议：与轻量 VAD 配合，在静音阶段推理。
- 建议统一“是否静音/是否语音”的判断来源：Silero VAD（或同等轻量 VAD）。
- barge-in 也建议基于 VAD 的“连续语音持续时间”而不是能量阈值。

**P0.3 复用 LLM client / 连接**
- 在 demo/main 初始化一个 `async_openai::Client` 并复用，避免每 turn 重建。
- 如果对方支持 keep-alive，复用可以显著稳定首包/首字延迟。

### P1（第二优先级，体验提升明显）

**P1.1 处理回声/自激（AEC 现实约束）**
- 默认假设用户戴耳机或系统层提供 AEC；文档里明确该前提。
- 如需软处理：
  - “助手播报期间”提高 barge-in 阈值（更保守）
  - 或检测“麦克风能量与播放能量高度相关”时不触发 barge-in（启发式）

**P1.2 模型预热**
- Smart Turn / ASR / GPT-SoVITS 在首次推理常见抖动。
- 在应用启动时做一次 dummy 推理（或读入后立即跑一小段），减少首轮体验波动。

**P1.3 指标与可观测性统一**
- 统一输出：
  - ASR：段延迟（segment end → 文本输出）、RTF、模型推理耗时
  - LLM：TTFB/首字、吞吐
  - TTS：首音、生成耗时、真实播放完成
  - 端到端：用户 turn end → 首音（可作为核心 KPI）

### P2（长期）

- 更智能的“意图型打断”：不是“有声就打断”，而是快速判断用户是否真要打断（例如快速 ASR 出 1~2 词后做分类，或检测特定打断短语）。
- 说话人/声纹（你的规划里后期再做）。
- 真正的 streaming ASR（非 VAD 分段输出）/ partial hypothesis UI（如果模型能力允许）。

---

## 可验证指标（建议你们作为验收标准）

建议在 `examples/voice_assistant.rs` 的日志里可直接观测/计算：

- `E2E_TTFA`: 用户 turn end → 助手首音（越低越好，稳定性更重要）
- `ASR_LAG`: 音频时间线 end → 文本出段
- `LLM_TTFB`: 请求发出 → 首个 delta 到达
- `TTS_FIRST_AUDIO`: 片段送入 pipeline → 首音
- `BARGE_IN_REACT`: 用户开始说话 → 助手停止（并区分“误触发率”）

---

## 配置建议（现有开关的使用方式）

- `ASR_VAD_MIN_SILENCE`：越大越不敏感（更少切段，但出结果更晚）。
- `SMART_TURN_MIN_SILENCE_MS` / `SMART_TURN_COMMIT_MS`：越大越“稳”，但 turn end 确认更慢。
- `BARGE_IN_MIN_SPEECH_MS`：越大越不容易误打断；建议从 `400~600ms` 起调。
