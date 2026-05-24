use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum StepError {
    #[error("step interrupted: {0} (checkpoint: {1})")]
    Interrupted(String, Uuid),
    #[error("max iterations exceeded")]
    MaxIterationsExceeded,
    #[error("max retries exceeded: {0}")]
    MaxRetriesExceeded(Box<StepError>),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("domain error: {0}")]
    Domain(String),
}

#[derive(Error, Debug)]
pub enum DomainError {
    #[error("LLM error: {0}")]
    Llm(String),
    #[error("entropy threshold not configured")]
    EntropyThresholdMissing,
    #[error("capability not found: {0}")]
    CapabilityNotFound(String),
    #[error("skill execution failed: {0}")]
    SkillFailed(String),
}
