use std::path::PathBuf;

use fosra::ingestion::types::Document;
use fosra::processing::corpus::Corpus;
use fosra::processing::embedding::EmbeddingEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths
    let md_path = PathBuf::from("/home/roccoluxe/FOSRA/z-misc/sample-files/sample-md.md");
    let stopwords_path = PathBuf::from("/home/roccoluxe/FOSRA/z-misc/resources/stopwords.txt");
    let model_dir: PathBuf = [
        std::env::home_dir().unwrap().to_str().unwrap(),
        "models",
        "embedding",
        "onnx",
    ]
    .iter()
    .collect();

    // 1. Ingest document
    println!("Ingesting: {}", md_path.display());
    let mut doc = Document::walk_md(md_path)?;
    println!("  {} sections loaded", doc.content.len());

    // 2. Build corpus and add document
    println!("Building TF-IDF corpus...");
    let mut corpus = Corpus::new(&stopwords_path)?;
    corpus.add_document(&doc);
    println!("  {} doc(s) in corpus", corpus.len());

    // 3. Populate keywords on each section
    println!("Computing keywords...");
    corpus.populate_keywords(&mut doc, 15);
    for section in &doc.content {
        println!(
            "  Section '{}': {} keywords",
            section.path,
            section.keywords.len()
        );
        for kw in &section.keywords {
            println!("    {:30} {:.4}", kw.text, kw.score);
        }
    }

    // 4. Load embedding model
    println!("\nLoading embedding model from: {}", model_dir.display());
    let mut engine = EmbeddingEngine::new(model_dir)?;

    // 5. Embed document sections
    println!("Computing embeddings...");
    engine.embed_document(&mut doc)?;

    // 6. Cosine similarity between all section pairs
    let embeddings: Vec<&[f32]> = doc
        .content
        .iter()
        .flat_map(|s| s.embedding.as_deref())
        .collect();

    println!(
        "  Got {} embeddings, dim={}",
        embeddings.len(),
        embeddings.first().map_or(0, |e| e.len()),
    );

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let dot: f32 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_i = embeddings[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_j = embeddings[j].iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos_sim = dot / (norm_i * norm_j).max(f32::EPSILON);
            println!(
                "  Cos sim [{}] vs [{}]: {:.4}",
                i, j, cos_sim
            );
        }
    }

    Ok(())
}
