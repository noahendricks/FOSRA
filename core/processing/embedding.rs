use std::path::PathBuf;

use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};

/// Embedding engine that wraps fastembed for batched inference.
pub struct EmbeddingEngine {
    model: TextEmbedding,
}

impl EmbeddingEngine {
    /// Load from a local ONNX model directory containing model.onnx, tokenizer.json,
    /// config.json, special_tokens_map.json, tokenizer_config.json.
    pub fn new(model_dir: PathBuf) -> Result<Self, String> {
        let onnx_file =
            std::fs::read(model_dir.join("model.onnx")).map_err(|e| format!("Failed to read model.onnx: {e}"))?;
        let tokenizer_file = std::fs::read(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("Failed to read tokenizer.json: {e}"))?;
        let config_file =
            std::fs::read(model_dir.join("config.json")).map_err(|e| format!("Failed to read config.json: {e}"))?;
        let special_tokens_map_file = std::fs::read(model_dir.join("special_tokens_map.json"))
            .map_err(|e| format!("Failed to read special_tokens_map.json: {e}"))?;
        let tokenizer_config_file = std::fs::read(model_dir.join("tokenizer_config.json"))
            .map_err(|e| format!("Failed to read tokenizer_config.json: {e}"))?;

        let model = UserDefinedEmbeddingModel {
            onnx_file,
            external_initializers: Vec::new(),
            tokenizer_files: TokenizerFiles {
                tokenizer_file,
                config_file,
                special_tokens_map_file,
                tokenizer_config_file,
            },
            pooling: Some(Pooling::Cls),
            quantization: QuantizationMode::None,
            output_key: None,
        };

        let engine = TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::default())
            .map_err(|e| format!("Failed to init embedding model: {e}"))?;

        Ok(Self { model: engine })
    }

    /// Embed a batch of texts. Returns one vector per input.
    pub fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        self.model
            .embed(texts, None)
            .map_err(|e| format!("Embedding failed: {e}"))
    }

    /// Embed all sections in a document, assigning vectors in-place.
    /// Does **not** extract keywords — use a `Corpus` for TF-IDF keywords separately.
    pub fn embed_document(&mut self, doc: &mut crate::Document) -> Result<(), String> {
        let texts: Vec<String> = doc.content.iter().map(|s| s.text.clone()).collect();
        let embeddings = self.embed(texts)?;
        for (section, emb) in doc.content.iter_mut().zip(embeddings) {
            section.embedding = Some(emb);
        }
        // Document-level embedding: mean pool section embeddings
        if !doc.content.is_empty() {
            let dim = doc.content[0].embedding.as_ref().map_or(0, |e| e.len());
            if dim > 0 {
                let mut pooled = vec![0.0f32; dim];
                for section in &doc.content {
                    if let Some(ref emb) = section.embedding {
                        for (p, e) in pooled.iter_mut().zip(emb.iter()) {
                            *p += e;
                        }
                    }
                }
                let n = doc.content.len() as f32;
                for p in pooled.iter_mut() {
                    *p /= n;
                }
                doc.embedding = pooled;
            }
        }
        Ok(())
    }

    /// Embed all code blocks, assigning vectors back in-place.
    pub fn embed_code_blocks(&mut self, blocks: &mut [crate::CodeBlock]) -> Result<(), String> {
        let texts: Vec<String> = blocks.iter().map(|b| b.text.clone()).collect();
        let embeddings = self.embed(texts)?;
        for (block, emb) in blocks.iter_mut().zip(embeddings) {
            block.embedding = Some(emb);
        }
        Ok(())
    }
}
