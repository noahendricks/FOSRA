use surrealdb::Surreal;
use surrealdb::engine::local::{Db, SurrealKv};

pub struct SurrealDb {
    db: Surreal<Db>,
}

impl SurrealDb {
    pub async fn connect(path: &str) -> Result<Self, surrealdb::Error> {
        let db = Surreal::new::<SurrealKv>(path).await?;
        db.use_ns("fosra").use_db("fosra").await?;
        Ok(Self { db })
    }
}
