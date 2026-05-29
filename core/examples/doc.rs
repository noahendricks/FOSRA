use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use fosra::ingestion::doc_types::Document;

use fosra::processing::embedding::EmbeddingEngine;

use fosra::processing::corpus::parse_keywords;

fn main() -> Result<()> {
    let doc_path = PathBuf::from("/home/roccoluxe/FOSRA/z-misc/sample-files/sample-md1.md");
    let mut doc = Document::walk_md(doc_path.clone())?;

    let kw = parse_keywords(doc_path.to_str().unwrap_or("invalid doc path {}"), None)?;

    let model_dir = "/home/roccoluxe/models/embedding/onnx";

    let mut embedder = EmbeddingEngine::new(model_dir.into()).map_err(|e| {
        anyhow!(
            "failed because model dir incorrect or invalid: {}",
            model_dir
        )
    })?;

    // embed sections in place
    _ = doc.embed_document_mut(&mut embedder, 896, doc.content.len());

    doc.metadata.extracted_keywords = Some(kw);

    dbg!(&doc);

    Ok(())
}
