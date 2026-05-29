// procmem paper

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub activation: String,
    pub procedure: String,
    pub termination: String,
    pub score: f64,
    pub total_invocations: u64,
    pub last_used: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPool {
    pub skills: Vec<Skill>,
    pub total_tokens: usize,
    pub active_scores: HashMap<String, f64>,
}
