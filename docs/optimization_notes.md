# 2.1 / 2.2 / 2.3 优化思路与取舍

本文不是待办清单（见 `docs/optimized.md`），而是把 **为什么要做**、**可选方案**、**权衡点**、**如何验证** 写清楚，方便后续按目标逐步落地。

---

### 2.1 RingBuffer 等待策略（CPU 抖动风险）

### 现状（代码位置）

- Rodio 后端的输出队列是一个有界 ring buffer：`src/audio/rodio.rs::RingBuffer`（内部使用 `crossbeam_queue::ArrayQueue<f32>`）。
- 写入侧：`RingBuffer::push_blocking()` 在队列满时采用 *polling + exponential backoff sleep*（100/200/400/800us）重试，直到写入完成或取消。
- 读出侧：`RingBufferSource::next()` 在 Rodio 播放线程里按 sample 逐个 `pop()`；队列空时输出 0（静音）。
- 现有观测：`RingBuffer` 内已有 `full_count/blocked_us` 统计，并可通过 `AUDIO_RING_METRICS` 打开调试日志。

### 风险与“为什么值得考虑”

1. **忙等 + 短 sleep 的调度抖动**：当系统负载高或线程优先级竞争激烈时，`sleep(Duration::from_micros(..))` 的实际唤醒时间可能远大于目标，导致写入侧“被动拉长等待”，表现为不稳定的延迟尖峰。
2. **CPU 抖动与抢占效应**：忙等循环里频繁 wake-up 会增加上下文切换与 cache 抖动；在同进程同时跑 ASR/ONNX/TTS 时，会放大对 runtime 的扰动。
3. **中断响应尾延迟**：`push_blocking()` 虽然会检查 `CancelScope`，但 sleep 期间无法立即响应取消；极端情况下（持续满队列）会出现取消“看起来不够即时”的尾部延迟（通常是 sub-ms，但可能叠加）。

> 注意：这类问题通常是“系统过载/队列长期满”的症状放大器。要判断是否要做，首先看 `full_count/blocked_us`、`buffered_ms` 水位是否经常进入危险区间。

### 目标（建议先明确）

- **当队列满时**：写入侧应“阻塞等待可用空间”，但不要忙等；CPU 使用应平稳。
- **当队列不满时**：尽可能保持当前低开销路径（push 成功即返回），不要给播放线程增加锁竞争。
- **不中断音频线程**：播放线程的 `next()` 仍需保持轻量，避免引入高频锁/阻塞。

### 可选方案（按改动风险从低到高）

#### 方案 A：在现有 `ArrayQueue` 上增加事件通知（推荐优先评估）

思路：保留 `ArrayQueue` 的 lock-free 数据路径，仅把“队列满时的等待”从 sleep 改为 *park/unpark* 或等价的事件机制。

- **实现要点**
  - 写入侧 `push` 失败（队列满）时进入等待：`park()` 或 `Condvar::wait()`（不再循环 sleep）。
  - 读出侧每次成功 `pop` 后在“从满→非满”的边界触发一次 `unpark()`/`notify_one()`，避免每个 sample 都 notify。
- **关键细节**
  - 要避免“丢唤醒”：`Condvar` 需要受互斥保护的条件变量；`std::thread::park/unpark` 或 `crossbeam_utils::sync::Parker` 自带 token，更适合做无锁唤醒。
  - 当前 RodioBackend 设计“单 active writer”，使得 ring buffer 接近 **SPSC**，事件机制可以更简单（只需要唤醒一个 writer）。
- **优点**
  - 代码侵入小、易验证；对现有行为影响最少。
  - 保持 `buffered_ms()` 语义不变。
- **缺点**
  - 仍是 sample 级 push/pop（原子操作次数不变），只是把满队列时的等待方式改掉。

#### 方案 B：改为真正的 SPSC RingBuffer（原子 head/tail）+ 等待策略

思路：利用“单写单读”的事实，用更贴近音频场景的 SPSC ring buffer（可自研或引入成熟 crate）替换 `ArrayQueue`。

- **实现要点**
  - 固定容量数组 + 原子 head/tail。
  - 写入侧可以一次性写入 slice（减少 per-sample push 成本）。
  - 满时使用 `park/unpark` 或 `Condvar`。
- **优点**
  - 性能上限更高（尤其是批量写入）；实现可以更可控。
- **缺点**
  - 回归面较大，需要重新验证：`buffered_ms`、prefill 启动、下溢填零策略、stop/reset 语义。

#### 方案 C：以 chunk 为单位排队（`Vec<f32>`/`Arc<[f32]>`），Source 内消费 chunk

思路：将 ring buffer 的元素从 `f32 sample` 提升为 `chunk`（例如 20–50ms），写入侧 push chunk、读出侧在 chunk 内迭代 sample。

- **优点**
  - 显著降低原子/队列操作次数；也便于做 RMS、时间戳对齐等“按窗口”逻辑。
- **缺点**
  - chunk 的内存管理与分配策略需要设计（避免频繁分配/复制）。
  - Rodio `Source::next()` 仍按 sample 调用，但内部可以“从 chunk 里顺序读”，实现复杂度增加。

### 如何验证（建议先做可观测性，再做改造）

- **负载指标**：`RingBuffer.full_count`、`RingBuffer.blocked_us`（已有），建议再加 `max_len`/`avg_len`（或用 `buffered_ms` 水位分布）。
- **用户体验指标**：`TTS_TTFA` 是否出现尖峰、打断(stop_fast)的尾延迟是否变差、是否出现音频“卡顿/破音”。
- **实验方法**：
  - 人为制造“写入过快”（例如播放端降速或增大 prefill），观察满队列频率。
  - 同时开启 ASR + SmartTurn + TTS，观察 CPU 抖动与音频输出稳定性。

---

### 2.2 ASR segment 队列背压（端到端延迟抬升）

### 现状（代码位置）

- `src/asr/sherpa.rs`：
  - VAD 线程（`spawn_blocking`）把 `VadSegment` 发送到 ASR 推理线程：`vad_segment_tx.blocking_send(...)`（有界队列）。
  - ASR 推理线程 `blocking_recv()` 消费 `VadSegment` 并做整段推理。
  - Mic→VAD 的输入是 `std::sync::mpsc::channel`（无界），Tokio 任务持续转发音频到 VAD 线程。

### 为什么可能导致“端到端延迟抬升”

当 ASR 推理速度低于实时（或 VAD 产段过密）时：

1. `vad_segment` 队列逐渐被填满 → `blocking_send` 阻塞 VAD 线程。
2. VAD 线程阻塞后无法继续消费 Mic→VAD 的输入（无界 mpsc）→ 输入音频在内存里排队增长。
3. 排队增长意味着 **识别到的文本永远落后于当前语音**，并且延迟会随积压线性增长（越说越慢）。

这对“实时助手 + 可打断（barge-in）”场景尤其致命：用户体感是“系统慢半拍甚至停住”，并且打断/端点事件也可能变迟。

### 先明确需求（决定策略的关键）

ASR 背压策略本质是：**过载时要牺牲什么？**

- **实时助手（单活跃 turn）**：通常更愿意牺牲“部分转写完整性”，换取“低延迟与及时打断”。
- **录音转写/会议纪要**：通常不能丢音频，宁愿延迟上升，也要保证完整性。
- **多人/远场**：可能需要更复杂的仲裁与分级队列（“主人优先”）。

### 可选策略（从保守到激进）

#### 策略 A：继续阻塞，但把背压传回更上游（“不丢数据”优先）

做法：

- 让 Mic→VAD 的输入也变成有界队列（或在回调层丢帧/降采样），使背压能真正回到输入端，而不是在无界 mpsc 堆内存。

优点：不丢 `VadSegment`，语义最“正确”。
缺点：输入端会丢帧或阻塞（取决于实现），对实时交互体验仍可能不佳。

#### 策略 B：过载时丢弃 / 合并 `VadSegment`（“低延迟”优先）

典型做法：

- 当 `vad_segment` 队列长度超过阈值时：
  - **丢弃旧的 segment**（保留最新的语音），或
  - **丢弃新的 segment**（保证已排队的能处理完），或
  - **合并多个小 segment**（减少推理次数，但会增加单次推理时长）。

适用：实时助手、单活跃 turn。
风险：会出现转写缺失/错漏，需要产品层接受。

#### 策略 C：自适应降载（动态调参）

当检测到 ASR backlog 增大时，动态调整：

- 提高 VAD 的最小语音/静音阈值、增大 chunk、降低产段频率；
- 或降低 ASR 推理频率（例如只在静音窗口/端点候选时触发推理）。

优点：比“简单丢弃”更平滑；能把质量下降控制在可接受范围。
缺点：实现复杂，需要大量参数回归。

#### 策略 D：分级队列（“事件优先”）

保证 `VadEvent`（SpeechStart/End）优先实时，而 `VadSegment` 可以被延迟/丢弃：

- 事件通道保持有界但优先处理；
- segment 通道可采用策略 B/C。

你现在已经将 `VadEvent` 与 `VadSegment` 拆开了，这是做分级策略的基础。

### 如何验证（建议）

1. **新增指标**（强烈建议先做）：`vad_segment_queue_len`、`vad_segment_drop_count`、`vad_segment_block_ms`。
2. **压测场景**：人为把 ASR 推理线程 sleep（模拟低端 CPU/模型过大），观察 turn end 与 barge-in 是否还能稳定实时。
3. **体验阈值**：为实时助手定义“可接受最大滞后”（例如 `queue_len` 对应的秒数），超出即降载/中断。

---

### 2.3 RMS 采样点（lipsync 对齐精度）

### 现状（代码位置）

- `src/audio/mod.rs`：RMS/peak 遥测在写入端（SegmentWriter.push）进行采样，并按 ~50ms 窗口分块发出事件。
- 播放端存在 `prefill_ms`（默认 50ms）以及 ring buffer 排队水位，导致“写入时刻”与“实际发声时刻”存在偏移。

### 为什么会影响 lipsync

如果 UI/口型动画直接消费写入侧 RMS：

- RMS 事件会 **提前于实际声音**（至少提前 `prefill_ms`，并且还会叠加当时的 `buffered_ms` 水位）。
- 当系统负载变化导致 `buffered_ms` 波动时，嘴型会出现“忽快忽慢”的错位感。

### 可选方案（从低风险到高收益）

#### 方案 A：保持写入端采样，但由 UI 用 `buffered_ms` 做延迟对齐

做法：事件带上当时的 `buffered_ms`（你已有），UI 将 RMS 延迟 `buffered_ms + prefill` 再展示。

优点：代码改动小。
缺点：UI 需要准确调度（定时器精度、线程调度都可能影响），并且 `buffered_ms` 是观测值不是严格时钟。

#### 方案 B：写入端采样 + 生成“预计播放时间戳”（推荐优先评估）

做法：在音频侧维护 play-domain 时钟（已有 `stream_start`、sample 计数等基础设施），为每个 RMS 窗口计算一个 `play_ts`：

- `play_ts = stream_start + (sample_offset / sample_rate)`

UI 使用 `play_ts` 而不是“事件到达时间”驱动动画。

优点：更接近真实播放对齐；仍不需要在音频线程里做额外计算。
缺点：需要确认 `stream_start` 的语义与漂移（尤其是 stop/reset/prefill）。

#### 方案 C：在播放端采样（消费端 RMS，最准确）

做法：在 `RingBufferSource::next()`（或类似播放域回调）按实际输出采样计算 RMS，并通过非阻塞方式上报。

优点：对齐最准确（真实“发声域”）。
缺点：

- 音频线程非常敏感：必须避免阻塞、避免分配、避免锁竞争；
- 上报链路需 lock-free（例如 ArrayQueue + tokio poll），否则可能引入音频卡顿。

### 如何验证（建议）

- **对齐验证**：录一段输出（或用 loopback），对比音频波形包络与 RMS 事件时间戳的相位差。
- **prefill 影响**：分别在 `AUDIO_PREFILL_MS=0/50/100` 下验证 lipsync 偏移是否与预期一致。
- **负载扰动**：在 CPU 高负载下验证 RMS 对齐是否仍稳定（避免“抖动嘴型”）。
