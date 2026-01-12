// Core types and traits (always available)
pub mod types;
pub use types::{
    AudioFrameRef, TurnBoundaryDetector, TurnDetectorConfig, TurnEvent, TurnEventKind,
};

// Turn context (turn_id + epoch snapshots)
pub mod context;
pub use context::{TurnContext, TurnManager};

// VAD-based turn detection (always available when asr-sherpa is enabled)
#[cfg(feature = "asr-sherpa")]
pub mod vad_gate;
#[cfg(feature = "asr-sherpa")]
pub use vad_gate::VadGateTurnDetector;

// Smart Turn detection (requires turn-smart feature)
#[cfg(feature = "turn-smart")]
pub mod smart_turn;
#[cfg(feature = "turn-smart")]
pub use smart_turn::{
    SmartTurnBoundaryDetector, SmartTurnConfig, SmartTurnDecision, SmartTurnDetector,
    SmartTurnModel,
};
