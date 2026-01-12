# 架构重构实施计划 v3.3 (Final)

按 PR 拆分，降低回归面。

---

## 关键补强 (vs v3.2)

| 项 | v3.2 | v3.3 |
|---|---|---|
| tick 时间源 | `Instant::now()` | `frame.ts` (确定性) |
| TurnManager | 返回新快照 | 持有 current_context 字段 |
| AudioFrameRef | 缺 channels | 补 `channels: u16` |
| 编排器循环 | 未规范 | drain VAD → push → tick |
| VadEvent.ts | 未定义 | 检测时刻 (detection time) |
| MetricsSink | 缺基准点 | 补 LlmStart |

---

## Phase 1: TurnBoundaryDetector + VadEvent

### 1.1 VadEvent/VadState (PR1)

```rust
// src/asr/mod.rs
pub enum VadEvent {
    SpeechStart { ts: Instant },  // ts = 检测时刻
    SpeechEnd { ts: Instant, duration_ms: u32 },
}

pub struct VadState {
    pub speaking: bool,
    pub last_change_ts: Instant,
    pub seq: u64,
}

impl SherpaAsrStream {
    pub fn try_read_vad_event(&mut self) -> Option<VadEvent>;
    pub fn vad_state(&self) -> VadState;
}
```

### 1.2 TurnBoundaryDetector + VadGateTurnDetector (PR2)

```rust
pub struct AudioFrameRef<'a> {
    pub samples: &'a [i16],
    pub sample_rate: u32,
    pub channels: u16,  // 补齐
    pub ts: Instant,
}

pub trait TurnBoundaryDetector: Send {
    fn push_audio(&mut self, frame: AudioFrameRef<'_>, out: &mut SmallVec<[TurnEvent; 4]>);
    fn push_vad(&mut self, event: VadEvent, out: &mut SmallVec<[TurnEvent; 4]>);
    fn tick(&mut self, now: Instant, out: &mut SmallVec<[TurnEvent; 4]>);
    fn reset(&mut self);
}
```

### 1.3 SmartTurnDetector 适配 (PR3)

### 1.5 voice_assistant 重构 (PR3)

**编排器循环模式**：
```rust
loop {
    // 1. 读取麦克风帧
    let frame = ...;
    
    // 2. drain VAD 事件 (边沿先于 tick)
    while let Some(vad) = asr.try_read_vad_event() {
        detector.push_vad(vad, &mut events);
    }
    
    // 3. push 音频
    detector.push_audio(frame, &mut events);
    
    // 4. tick (使用 frame.ts)
    detector.tick(frame.ts, &mut events);
    
    // 5. 处理事件 (绑定 turn_id)
    let turn_id = turn_manager.current_context().turn_id();
    for event in events.drain(..) {
        // → SessionState 映射
        // → 驱动 turn_end/stop_fast
    }
}
```

---

## Phase 2: TurnContext (PR4)

### 2.1 TurnManager (持有当前上下文)

```rust
pub struct TurnManager {
    cancel_token: Arc<CancelToken>,  // 必须与 TTS epoch 同源
    current: RwLock<TurnContext>,
}

impl TurnManager {
    /// 当前快照 (用于绑定事件/指标)
    pub fn current_context(&self) -> TurnContext;
    
    /// 递增 epoch + 更新当前上下文 (原子)
    pub fn advance_turn(&self) -> TurnContext;
}
```

> **关键约束**：`cancel_token` 必须与 `TtsEngine` 内部 epoch 同源，否则 `From<&TurnContext> for CancelScope` 语义落空。

### 2.2 CancelScope 兼容层

### 2.3 组件迁移

---

## Phase 3: MetricsSink (PR5)

### 3.1 原子事件 (含 LlmStart 基准)

```rust
pub enum MetricEventKind {
    TurnEnd,
    LlmStart,  // 补齐基准点
    LlmFirstToken,
    TtsFirstAudio,
    AsrInference { infer_ms: u64 },
}

pub struct MetricEvent {
    pub turn_id: u64,
    pub kind: MetricEventKind,
    pub ts: Instant,
}
```

### 3.2/3.3 埋点替换 + Examples 注入

---

## PR 拆分

| PR | 范围 | 风险 |
|----|------|------|
| PR1 | VadEvent/VadState + try_read + vad_state | 低 |
| PR2 | TurnBoundaryDetector + VadGateTurnDetector | 中 |
| PR3 | SmartTurnDetector 适配 + voice_assistant 重构 | 中 |
| PR4 | TurnContext/TurnManager + epoch 统一 | 高 |
| PR5 | MetricsSink + 埋点迁移 | 低 |

---

## 文档同步

每个 PR 合并后更新 ARCHITECTURE.md：
- PR1: VadEvent/VadState 类型
- PR2: TurnBoundaryDetector trait
- PR4: TurnManager/TurnContext, epoch 语义
- PR5: MetricsSink
