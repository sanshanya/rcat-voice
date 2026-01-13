# rcat-voice 架构重构 v4.0 任务清单

## Phase 1: Pipeline 统一调度器
- [x] PR1-1: 抽取公共逻辑 `on_segment_received` / `on_metrics`
- [x] PR1-2: 引入 `OutcomeKind` 枚举统一产物类型
- [x] PR1-3: 合并 [run()](file:///e:/rcat/rcat-voice/src/main.rs#39-167) 与 `run_parallel()` 为单一调度 loop
- [x] PR1-4: 配置语义收敛 (mode: Auto/Serial/Decoupled)

## Phase 2: 取消/控制接口统一
- [x] PR2-1: 新增 `finish_input()` + `input_finished` 标志位
- [x] PR2-2: 引入 `CancellationToken`，Tokenizer/Pipeline select 退出
- [x] PR2-3: 删除 `watch` cancel 信号，统一到 token
- [x] PR2-4: 合并 [StreamControl](file:///e:/rcat/rcat-voice/src/streaming.rs#174-181) + `StreamCancelHandle` → `StreamHandle`
- [x] PR2-5: 删除 [pause()](file:///e:/rcat/rcat-voice/src/streaming.rs#45-50) (已 deprecated)

## Phase 3: Channel 拓扑优化
- [x] PR3-1: 命名收敛 ([Segment](file:///e:/rcat/rcat-voice/src/asr/mod.rs#5-13) → `TextSegment`, `chunk_tx` → `text_seg_tx`)
- [x] PR3-2: 默认容量调整（`STREAM_DELTA_CAPACITY=8192` / `STREAM_SEGMENT_CAPACITY=4096`）
- [x] PR3-3: 删除 buffer poll task + watch（relax 逻辑留在分句侧）
- [x] PR3-4: 单 Orchestrator 合并 Tokenizer + Pipeline
- [x] PR3-5: 单入口 `StreamMsg` channel + epoch 过滤

## Phase 4: 文档同步与回归
- [x] 更新 ARCHITECTURE.md
- [x] 更新 README.md
- [x] 编译回归（`cargo check`：`rcat-voice` + `src-tauri`）
- [ ] 运行回归（建议在 Windows 上跑 `voice_assistant`/实际 TTS 后端，验证 TTFA 与 barge-in）
