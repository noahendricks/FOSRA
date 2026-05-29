use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
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
pub struct StdOut {}

pub trait Tracer {
    fn emit(&self) -> String;
}
impl Tracer for StdOut {
    fn emit(&self) -> String {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub trace_id: Uuid,
    pub thread_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}
