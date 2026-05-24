use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct StepEvent {
    pub timestamp: DateTime<Utc>,
    pub trace_id: Uuid,
    pub span_id: Uuid,
    pub parent_id: Option<Uuid>,
    pub event: StepEventKind,
    pub step_name: String,
    pub step_depth: u16,
    pub state_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub enum StepEventKind {
    Started { state_hash: String },
    Completed { state_hash: String },
    Retrying { attempt: u8, error: String },
    Checkpointed { checkpoint_id: Uuid },
    Interrupted { label: String },
    Failed { error: String },
    HLLMLessonRetrieved { error_type: String, lesson: String, source_step: String },
    SkillExecuted { skill_id: String, was_cached: bool, execution_time_ms: u64 },
    CacheHit { cache_stage: CacheStage, cache_key: String },
    FadeMemDecay { layer: MemoryLayer, entries_removed: usize, total_storage_bytes: usize },
    DALIAValidation { status: ValidationStatus, missing_capabilities: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStage {
    AIR,
    VS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLayer {
    SML,
    LML,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Partial,
}
impl Default for ValidationStatus {
    fn default() -> Self {
        Self::Valid
    }
}

pub trait Tracer: Send + Sync {
    fn emit(&self, event: StepEvent);
}
