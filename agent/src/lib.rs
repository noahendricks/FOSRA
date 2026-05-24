pub mod core;
pub mod memory;
pub mod domain;

// Flat re-exports so crate::Path resolves correctly for internal module imports
// and for consumers who use fosra_agent::TypeName directly.
pub use core::{
    Step, Executor, Checkpoint, CheckpointStore,
    Tracer, StepEvent, StepEventKind, CacheStage, ValidationStatus, MemoryLayer,
    StepError, DomainError, ExecutionTrace,
};
pub use memory::{
    MemoryEntry, ConflictResolution, ConflictType,
    HLLMLesson, DebugResult, DebugStatus,
};
