use thiserror::Error;

#[derive(Error, Debug)]
pub enum IngestionError {
    #[error("parsing failed or interrupted - doc: {message} {source} ")]
    Parsing {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
