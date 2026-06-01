// use fosra_agent::{Checkpoint, CheckpointStore, StepError};
// use serde::Serialize;
// use std::collections::HashMap;
// use std::marker::PhantomData;
// use std::sync::Arc;
// use tokio::sync::RwLock;
// use uuid::Uuid;

// pub struct InMemoryCheckpointStore<S> {
//     store: Arc<RwLock<HashMap<Uuid, Checkpoint<S>>>>,
// }

// impl<S> InMemoryCheckpointStore<S> {
//     pub fn new() -> Self {
//         Self {
//             store: Arc::new(RwLock::new(HashMap::new())),
//         }
//     }
// }

// #[async_trait::async_trait]
// impl<S: Serialize + Send + Sync + Clone> CheckpointStore<S> for InMemoryCheckpointStore<S> {
//     async fn save(&self, checkpoint: Checkpoint<S>) -> Result<Uuid, StepError> {
//         let id = checkpoint.id;
//         self.store.write().await.insert(id, checkpoint);
//         Ok(id)
//     }

//     async fn load(&self, id: Uuid) -> Result<Checkpoint<S>, StepError> {
//         self.store
//             .read()
//             .await
//             .get(&id)
//             .cloned()
//             .ok_or_else(|| StepError::Storage(format!("checkpoint not found: {id}")))
//     }

//     async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>, StepError> {
//         let store = self.store.read().await;
//         Ok(store
//             .values()
//             .filter(|c| c.thread_id == thread_id)
//             .max_by_key(|c| c.created_at)
//             .cloned())
//     }

//     async fn load_history(&self, thread_id: &str) -> Result<Vec<Checkpoint<S>>, StepError> {
//         let store = self.store.read().await;
//         let mut history: Vec<_> = store
//             .values()
//             .filter(|c| c.thread_id == thread_id)
//             .cloned()
//             .collect();
//         history.sort_by_key(|c| c.created_at);
//         Ok(history)
//     }

//     async fn delete(&self, id: Uuid) -> Result<(), StepError> {
//         self.store.write().await.remove(&id);
//         Ok(())
//     }
// }

// pub struct SqliteCheckpointStore<S> {
//     pool: sqlx::SqlitePool,
//     _phantom: PhantomData<S>,
// }

// impl<S> SqliteCheckpointStore<S> {
//     pub fn new(pool: sqlx::SqlitePool) -> Self {
//         Self {
//             pool,
//             _phantom: PhantomData,
//         }
//     }
// }

// #[async_trait::async_trait]
// impl<S: Serialize + Send + Sync> CheckpointStore<S> for SqliteCheckpointStore<S> {
//     async fn save(&self, _checkpoint: Checkpoint<S>) -> Result<Uuid, StepError> {
//         todo!("SqliteCheckpointStore::save")
//     }

//     async fn load(&self, _id: Uuid) -> Result<Checkpoint<S>, StepError> {
//         todo!("SqliteCheckpointStore::load")
//     }

//     async fn load_latest(&self, _thread_id: &str) -> Result<Option<Checkpoint<S>>, StepError> {
//         todo!("SqliteCheckpointStore::load_latest")
//     }

//     async fn load_history(&self, _thread_id: &str) -> Result<Vec<Checkpoint<S>>, StepError> {
//         todo!("SqliteCheckpointStore::load_history")
//     }

//     async fn delete(&self, _id: Uuid) -> Result<(), StepError> {
//         todo!("SqliteCheckpointStore::delete")
//     }
// }
