use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HLLMLesson {
    pub error_type: String,
    pub root_cause_location: String,
    pub why_previous_fixes_failed: String,
    pub suggested_fix_strategy: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugResult {
    pub repaired_code: Vec<u8>,
    pub trace: Vec<u8>,
    pub hllm_lessons_used: Vec<HLLMLesson>,
    pub rollback_count: u32,
    pub final_status: DebugStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebugStatus {
    Success,
    RolledBack,
    Unresolved,
}
