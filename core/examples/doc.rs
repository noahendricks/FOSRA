use fosra::code::CodeSource;
use owo_colors::OwoColorize;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use fosra::ingestion::doc::Document;

use fosra::processing::embedding::EmbeddingEngine;

use fosra::processing::corpus::parse_keywords;

 #[tokio::main]
 async fn main() -> Result<()> {
    let model_dir = "/home/roccoluxe/models/embedding/onnx";
    let mut embedder = EmbeddingEngine::new(model_dir.into()).map_err(|e| {
        anyhow!(
            "failed because model dir incorrect or invalid: {}",
            model_dir
        )
    })?;

    let doc_path = PathBuf::from("/home/roccoluxe/FOSRA/z-misc/sample-files/sample-md1.md");
    let mut doc = Document::walk_md(doc_path.to_string_lossy().to_string())?;

    let code_doc_path = "/home/roccoluxe/FOSRA/core/examples/code.rs".to_string();
    let mut code_doc = CodeSource::parse(code_doc_path).await?;

    _ = code_doc.embed_blocks_mut(896, code_doc.blocks.len(), &mut embedder);

    println!("{}", format!("{:?}", "=".repeat(100)).red());

    dbg!(code_doc);
    println!("{}", format!("{:?}", "=".repeat(100)).yellow());

    let doc_kw = parse_keywords(doc_path.to_str().unwrap_or("invalid doc path {}"), None)?;

    // embed sections in place
    _ = doc.embed_document(&mut embedder, 896, doc.content.as_ref().map(|v| v.len()).unwrap_or(0));

    doc.metadata.extracted_keywords = Some(doc_kw);

    // dbg!(&doc);

    Ok(())
}
