# 优化与未完成事项

本文档用于记录 rcat-voice 当前**尚未实现**、**仍有明显风险/限制**或**文档待同步**的事项，便于后续跟踪与规划。

> 说明：本文件只跟踪“需要做的事”，不保证优先级排序；实际取舍以性能目标、平台约束、产品需求为准。

---

## 1) 尚未实现

### 1.1 音频后端

- `AUDIO_BACKEND=wasapi`：未实现（`src/audio/wasapi.rs` / `src/audio/mod.rs`）
- `AUDIO_BACKEND=system`：未实现（`src/audio/system.rs` / `src/audio/mod.rs`）

### 1.2 跨平台支持缺口

- `gpt-sovits`（CUDA/libtorch）：目前仅 Windows + CUDA 环境验证与支持（Cargo feature 与 `cfg(target_os="windows")` 限制）
- `tts-worker`：目前为 Windows-only（`src/worker.rs` 对非 Windows 直接 `compile_error!`）

---

## 2) 已知限制 / 待优化点

### 2.1 RingBuffer 等待策略（CPU 抖动风险）

- 现状：Rodio RingBuffer 写入侧 `push_blocking()` 使用 polling + exponential backoff sleep（`src/audio/rodio.rs`）
- 风险：sleep 精度受 OS 调度影响，可能导致 CPU 争抢/抖动，或在高负载下出现延迟尖峰
- 方向：用 `Condvar` / 事件通知 / 更合适的并发队列（如基于通知的 bounded queue）替换 busy wait

### 2.2 ASR segment 队列背压（端到端延迟抬升）

- 现状：VAD→ASR 的 segment 队列满时使用 `blocking_send` 形成背压（`src/asr/sherpa.rs`）
- 影响：VAD 线程被阻塞，整条 ASR 链路会停滞，端到端延迟可能上升
- 方向：结合需求决定“阻塞/丢弃/降采样/增大队列/分级队列”等策略，并配合指标观测

### 2.3 RMS 采样点（lipsync 对齐精度）

- 现状：RMS/peak 遥测在写入端采样（`src/audio/mod.rs`），而非播放端
- 影响：用于 lipsync/UI 时可能带有 `prefill` 提前量
- 方向：若要更精确的口型对齐，考虑播放端采样/或将遥测对齐到 play-domain 时钟

### 2.4 Smart Turn 推理阻塞位置（运行时线程占用）

- 现状：`SmartTurnBoundaryDetector.tick()` 内同步调用端点推理（注释也标注为“阻塞操作”），目前依赖调用方的任务模型承载（`src/turn/smart_turn.rs`）
- 风险：若 tick 被高频调用且运行在核心异步线程上，可能造成 runtime 抖动
- 方向：将推理迁移到 `spawn_blocking`/专用线程 + 队列；同时保持 `frame.ts` 驱动的确定性语义

### 2.5 Pipeline 串行/并行调度器重复（技术债）

- 现状：`Pipeline.run()`（串行：直接 `engine.speak()`）与 `Pipeline.run_parallel()`（解耦：`synthesize→play_samples` + 保序）存在一套重复的“按 turn 重置状态 + 指标上报（`TtsFirstSegmentSent`/`TtsFirstAudio`）”逻辑（`src/pipeline.rs`）
- 收敛方向：保留现有引擎接口不变，在 Pipeline 内部统一为“单队列 + 单保序消费”的框架；串行/解耦仅在任务产物形态（`Played(TtsMetrics)` vs `NeedPlay(SynthesizedAudio)`）上不同
- 风险：若误把支持真流式的引擎（如 `speak` 可边产边播）切到 `synthesize→play_samples`（整句/整段合成），首音（TTFA）可能明显上升；需要明确模式语义并用指标回归验证
- 建议步骤：先抽公共逻辑（不改行为）→ 再引入统一 Outcome 消费 → 最后合并 loop（分 PR 降低回归面）
- 当前状态：✅ 已完成（统一调度 loop、OutcomeKind、公共指标逻辑抽取）

### 2.6 StreamSession 控制/取消接口偏复杂（API 技术债）

- 现状：对外暴露 `mpsc::Sender<String>` 导致 finish 语义依赖 “drop 所有 sender clone”；为绕开生命周期问题引入 `StreamCancelHandle`；取消路径再叠加 `AbortHandle`（让 send 立即 Err）与 `watch`（finish_or_cancel 可中断 drain），信号体系偏分散（`src/streaming.rs`）
- 补充背景（历史原因）：
  - `sender()` 暴露是有意的：更容易兼容外部 LLM SDK 的“流式回调/循环里直接 `send(delta).await`”的集成方式（例如 `async-openai`），调用侧只需持有一个可 clone 的 sender 即可持续写入增量
  - `AbortHandle` 是实现 O(1) 取消链路的一部分：仅靠“取消 token 广播”只能让任务“尽快退出”；若上游已在 `send().await` 上阻塞（channel 满/背压），需要通过 drop receiver（或 close channel）把等待中的 send 立刻唤醒并返回 Err
- 收敛方向：
  - 不再暴露 raw sender：改为 `push_delta()` 方法式输入，或引入显式 EOF（如 `DeltaMsg::{Delta,Eof}`）使 finish 不依赖 drop
  - 统一取消广播：用统一的 session cancellation token（`select! { _ = cancelled => ... }`）驱动各任务退出，减少每任务 `AbortHandle` 的需求（并行 synth 可保留 `JoinSet::abort_all()` 作为内部细节）
  - 对外提供更简单的 `stop()`/`finish_input()` 组合（打断 vs drain 语义明确），并考虑删除/隐藏 `pause`（当前等价于 interrupt）
- 风险：属于 API 级变更；若用“发送 EOF”实现 finish，需要避免“EOF 被队列淹没/发送阻塞”以及“finish 后仍成功 send 但被丢弃”的语义陷阱（可用状态机/原子门控拒绝后续发送）
- 当前状态：✅ 已完成（StreamHandle + finish_input；CancellationToken 统一；watch 取消；pause 移除；StreamMsg 输入）

### 2.7 Stream 通道拓扑与容量（可选收敛）

- 现状：`delta_tx: mpsc<String> → Tokenizer → chunk_tx: mpsc<Segment> → Pipeline`，另有 buffer poll task 通过 `watch<u64>` 向 Tokenizer 提供 `engine.buffered_ms()`（`src/streaming.rs` / `src/tokenizer.rs`）
- 痛点：
  - hop/调参点偏多：两条 mpsc + 一条 watch，取消/背压/容量需要分别考虑
  - 默认容量偏大（`delta_channel=8192`、`segment_channel=4096`），更像“把拥塞推迟到内存里”，对“可打断的单活跃 turn”场景不一定划算
- 低风险路径：先做命名统一 + 明确容量与背压策略（活跃 turn 不丢字；旧 turn 立刻 stop/丢弃）
- 中风险路径：若明确系统始终“单活跃 turn”（barge-in 触发 turn 切换），可考虑引入 `epoch/turn_id` 过滤“过期文本”，并进一步把 Tokenizer→Pipeline 的 hop 内聚为同一 orchestrator task（内部 `VecDeque` 代替 mpsc），减少拓扑复杂度
- 高收益/高回归面：去掉 buffer poll + watch（orchestrator 直接读取 `buffered_ms()`；或把 relax 节奏控制迁移到 Pipeline）
- 当前状态：✅ 已完成（StreamMsg 单入口 + Orchestrator；移除 buffer poll/watch；默认容量 8192/4096；epoch 过滤过期输入）

---

## 3) 文档与实现不一致（待同步）

### 3.1 Mic 捕获链路

- 文档描述为 “cpal 回调线程 → mpsc → tokio”
- 现实现为 cpal 回调写入 `crossbeam_queue::ArrayQueue`（`src/audio/mic.rs`），消费端轮询取样（examples 里也复刻了该模式）

### 3.2 VAD 输入通道类型

- 文档描述为 “`spawn_blocking` + std::sync::mpsc”
- 现实现：VAD 输入确实是 `std::sync::mpsc::channel`，但 Mic→VAD 并非 mpsc；建议在文档中明确“Mic 与 VAD 的边界通道类型”

### 3.3 Barge-in（打断）输入信号来源

- 架构图标注 “Mic PCM i16 (RMS) → Barge-in Detector”
- 现实现：example 中 barge-in 基于 ASR 侧 `VadEvent::SpeechStart/SpeechEnd` 做确认窗口计时（`examples/voice_assistant.rs`）

### 3.4 `TtsEngine` trait 展示不完整

- 实现中额外提供 `cancel_token()`，用于将 TurnContext/TurnManager 与 TTS 的 epoch 绑定到同一来源（`src/generator/mod.rs`、`src/turn/context.rs`）
- ARCHITECTURE.md 的 trait 片段目前未体现该方法

### 3.5 Pipeline 并行合成的生效条件

- Pipeline 的并行合成/信号量主要用于 `parallel_synth && engine.supports_synthesis_queue()`（`src/pipeline.rs`）
- 现有主要引擎返回 `supports_synthesis_queue=false`，默认路径仍以串行 `speak()` 为主；建议在文档中更明确这一点

### 3.6 Channel/变量命名约定（segment 歧义）

- 现状：音频分段在类型层面已统一为 `VadSegment`，但代码里很多地方仍使用 `segment_tx/segment_rx` 命名；文本分段类型为 `Segment`，但通道命名仍为 `chunk_tx/chunk_rx`
- 影响：在讨论“segment”时容易混淆（音频 segment vs 文本 segment），增加 review/维护成本
- 方向：逐步将命名收敛到 `vad_segment_*`（音频）与 `text_segment_*`（文本）；或在文档中明确“历史命名”并给出对照表
- 当前状态：✅ 文本分段已收敛为 `TextSegment` + `text_seg_*`/`stream_tx`；⏳ 音频 `vad_segment_*` 命名仍保留历史 `segment_tx`

---

## 4) 维护建议

- 建议每次修复/实现以上事项时：
  - 同步更新 `docs/ARCHITECTURE.md` / `docs/FEATURE_MAP.md`（如涉及配置项也更新 `docs/TROUBLESHOOTING.md`）
  - 在本文件对应条目下补充“已完成/替代方案/验证范围”
