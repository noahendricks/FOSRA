use crate::core::error::StepError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint<S> {
    pub id: Uuid,
    pub thread_id: String,
    pub label: String,
    pub state: S,
    pub sml: Vec<crate::MemoryEntry>,
    pub lml: Vec<crate::MemoryEntry>,
    pub hllm_lessons: Vec<crate::HLLMLesson>,
    pub skill_scores: HashMap<String, f64>,
    pub parent_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub depth: u16,
    pub trace_id: Uuid,
}

#[async_trait::async_trait]
pub trait CheckpointStore<S: Serialize + Send + Sync>: Send + Sync {
    async fn save(&self, checkpoint: Checkpoint<S>) -> Result<Uuid, StepError>;
    async fn load(&self, id: Uuid) -> Result<Checkpoint<S>, StepError>;
    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>, StepError>;
    async fn load_history(&self, thread_id: &str) -> Result<Vec<Checkpoint<S>>, StepError>;
    async fn delete(&self, id: Uuid) -> Result<(), StepError>;
}

pub struct InMemoryCheckpointStore<S> {
    store: Arc<RwLock<HashMap<Uuid, Checkpoint<S>>>>,
}
