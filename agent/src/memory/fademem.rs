use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::core::tracer::MemoryLayer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub content: Vec<u8>,
    pub importance: f64,
    pub layer: MemoryLayer,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub decay_rate: f64,
    pub access_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    pub memory_a_id: Uuid,
    pub memory_b_id: Uuid,
    pub resolution: ConflictType,
    pub resolved_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    Compatible,
    Contradictory,
    Subsumes,
    Subsumed,
}
