use std::path::PathBuf;

use fosra::DocumentMetadata;
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::Ws;
use surrealdb::opt::Resource;
use surrealdb::opt::auth::{Record, Root};
use surrealdb::types::{RecordId, SurrealValue, Value};

use fosra::Document;

#[tokio::main]
async fn main() -> surrealdb::Result<()> {
    let db = Surreal::new::<Ws>("127.0.0.1:8000").await?;

    db.signin(Root {
        username: "root".to_string(),
        password: "secret".to_string(),
    })
    .await?;

    db.use_ns("main").use_db("main").await?;

    // Create a new person with a random id
    // let created: Option<Record> = db
    //     .create("person")
    //     .content(Person {
    //         title: "Founder & CEO".to_string(),
    //         name: Name {
    //             first: "Tobie".to_string(),
    //             last: "Morgan Hitchcock".to_string(),
    //         },
    //         marketing: true,
    //     })
    //     .await?;
    // dbg!(created);

    let doc = Document::walk_md(PathBuf::from("/home/roccoluxe/FOSRA/ALGORITHMS.md")).unwrap();

    let doc_created: Option<Document> = db.create("document").content(doc).await?.unwrap();

    println!("{}", "created");
    dbg!(doc_created.unwrap());

    let check: Vec<Document> = db.select("document").await?;
    println!("{}", "check");
    dbg!(check);

    // let doc: Option<Document> = db.delete(("document", "ALGORITHMS")).await?;
    // println!("{}", "delete");
    // dbg!(doc);
    // db.update(Resource::from(("person", "jaime")))
    //     .merge(Responsibility { marketing: true })
    //     .await?;

    // Select all people records
    let people: Vec<Document> = db.select("document").await?;
    dbg!(people);

    // Perform a custom advanced query
    // let mut groups = db
    //     .query("SELECT , count() FROM type::table($table) GROUP BY marketing")
    //     .bind(("table", "document"))
    //     .await?;
    // Use .take() to transform the first query result into
    // anything that can be deserialized, in this case
    // a Value
    // dbg!(groups.take::<Value>(0).unwrap());

    Ok(())
}
