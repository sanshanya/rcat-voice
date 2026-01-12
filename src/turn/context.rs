use crate::audio::{CancelScope, CancelToken};
use crate::generator::TtsEngine;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::time::Instant;

/// 单轮对话上下文（Turn-level snapshot）。
///
/// 语义：
/// - `turn_id`：编排者分配的单调递增 ID（用于日志/metrics 绑定）。
/// - `epoch_snapshot`：取消权威快照；`CancelToken` 的 epoch 是取消与打断的唯一权威。
#[derive(Debug, Clone)]
pub struct TurnContext {
    cancel_token: CancelToken,
    epoch_snapshot: u64,
    created_at: Instant,
    turn_id: u64,
}

impl TurnContext {
    pub fn turn_id(&self) -> u64 {
        self.turn_id
    }

    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    pub fn epoch_snapshot(&self) -> u64 {
        self.epoch_snapshot
    }

    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel_token
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.current_epoch() != self.epoch_snapshot
    }
}

impl From<&TurnContext> for CancelScope {
    fn from(value: &TurnContext) -> Self {
        value.cancel_token.scope_at(value.epoch_snapshot)
    }
}

/// Turn 上下文管理器（持有当前快照）。
///
/// - `advance_turn()`：递增 epoch 并更新 current（进入新轮次）
/// - `advance_turn_no_cancel()`：仅递增 turn_id（用于 epoch 已由 stop_fast() 推进的场景）
/// - `current_context()`：返回当前快照
pub struct TurnManager {
    cancel_token: CancelToken,
    current: RwLock<TurnContext>,
    turn_seq: AtomicU64,
}

impl TurnManager {
    pub fn new(cancel_token: CancelToken) -> Self {
        let epoch = cancel_token.current_epoch();
        let created_at = Instant::now();
        let current = TurnContext {
            cancel_token: cancel_token.clone(),
            epoch_snapshot: epoch,
            created_at,
            turn_id: 0,
        };
        Self {
            cancel_token,
            current: RwLock::new(current),
            turn_seq: AtomicU64::new(0),
        }
    }

    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel_token
    }

    pub fn from_tts_engine(engine: &dyn TtsEngine) -> Option<Self> {
        engine.cancel_token().map(Self::new)
    }

    pub fn current_context(&self) -> TurnContext {
        match self.current.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    pub fn advance_turn(&self) -> TurnContext {
        self.cancel_token.cancel();
        self.advance_turn_no_cancel()
    }

    pub fn advance_turn_no_cancel(&self) -> TurnContext {
        let epoch = self.cancel_token.current_epoch();
        let turn_id = self.turn_seq.fetch_add(1, Ordering::AcqRel).wrapping_add(1);

        let ctx = TurnContext {
            cancel_token: self.cancel_token.clone(),
            epoch_snapshot: epoch,
            created_at: Instant::now(),
            turn_id,
        };

        let mut guard = self.current.write().unwrap_or_else(|e| e.into_inner());
        *guard = ctx.clone();
        ctx
    }
}
