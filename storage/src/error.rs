use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] sqlx::Error),
    #[error("SurrealDB error: {0}")]
    SurrealDb(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("checkpoint not found: {0}")]
    NotFound(Uuid),
}
