# 架构重构任务清单 v3.3

## PR1: VadEvent/VadState

- [X] 新增 VadEvent/VadState 公共类型
- [X] SherpaAsrStream 内产生与缓存
- [X] 提供 try_read_vad_event() + vad_state()
- [X] 同步 ARCHITECTURE.md

## PR2: TurnBoundaryDetector

- [X] TurnBoundaryDetector trait (含 tick)
- [X] AudioFrameRef (含 channels)
- [X] VadGateTurnDetector 实现
- [X] 同步 ARCHITECTURE.md

## PR3: SmartTurnDetector 适配

- [X] SmartTurnDetector 实现 TurnBoundaryDetector
- [X] voice_assistant 重构 (drain→push→tick)
- [X] voice_conversation 同步 (如适用)

## PR4: TurnContext

- [X] TurnManager (持有 current_context)
- [X] 确认 cancel_token 与 TTS epoch 同源
- [X] CancelScope 兼容层
- [X] 组件迁移 (watch 移除)
- [ ] turn_id 注入/透传 (日志/metrics/事件载荷)
- [X] 同步 ARCHITECTURE.md

## PR5: MetricsSink

- [ ] MetricsSink trait + 原子事件 (含 LlmStart)
- [ ] 埋点替换 (metrics.rs, sherpa.rs)
- [ ] Examples 注入
- [ ] 同步 ARCHITECTURE.md
