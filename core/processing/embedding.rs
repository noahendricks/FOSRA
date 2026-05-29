use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Error, Result, anyhow};
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use ndarray::AssignElem;
use serde::{Deserialize, Serialize};

use crate::{Section, ingestion::IsSection};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPair {
    pub query: String,
    pub document: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityFn {
    #[serde(rename = "cosine")]
    Cosine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    ///  "nl2code": {prompt} `qa"`: {prompt}
    pub prompts: HashMap<String, PromptPair>,

    pub default_prompt_name: Option<String>,

    pub similarity_fn_name: SimilarityFn,

    // allowed truncation values via -> `embed_with_dims()` | None = full dimension
    pub output_dims: Option<Vec<usize>>, // from models config.json
}

pub struct EmbeddingEngine {
    model: TextEmbedding,
}

impl EmbeddingEngine {
    pub fn new(model_dir: PathBuf) -> Result<Self, String> {
        let onnx_file = std::fs::read(model_dir.join("model.onnx"))
            .map_err(|e| format!("Failed to read model.onnx: {e}"))?;

        let tokenizer_file = std::fs::read(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("Failed to read tokenizer.json: {e}"))?;

        let config_file = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;

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

        let engine =
            TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::default())
                .map_err(|e| format!("Failed to init embedding model: {e}"))?;

        Ok(Self { model: engine })
    }

    pub fn embed(
        &mut self,
        texts: Vec<String>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.model
            .embed(texts, batch_size)
            .map_err(|e| format!("Embedding failed: {e}"))
    }

    /// Embeds sections in-place
    /// ```
    /// Section.embedding //<- mutated for each
    /// ```
    pub fn embed_sections_mut<T>(
        &mut self,
        sections: T,
        dims: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>>
    where
        T: IntoIterator,
        T::Item: IsSection,
    {
        let texts: Vec<String> = sections
            .iter()
            .filter_map(|s| {
                if !s.text.is_empty() {
                    Some(s.text.clone())
                } else {
                    Some(String::new())
                }
            })
            .collect();

        let mut embeddings = self
            .embed(texts, Some(batch_size))
            .map_err(|e| anyhow!("failed because {e}"))?;

        for vec in &mut embeddings {
            if vec.len() > dims {
                vec.truncate(dims);
            }
        }

        // place embeddings
        for (i, section) in sections.iter_mut().enumerate() {
            section.embedding = Some(embeddings[i].clone());
        }

        // document level embedding from N section vectors
        let n = embeddings.len() as f32;
        let mut doc_embedding = vec![0.0; dims];

        // consolidate onto single embedding
        for emb in embeddings.as_slice() {
            for (out, &val) in doc_embedding.iter_mut().zip(emb.iter()) {
                *out += val;
            }
        }
        // normalize by N vectors
        for val in doc_embedding.iter_mut() {
            *val /= n;
        }

        Ok(doc_embedding)
    }

    /// embed a single text the task-specific query prompt from `spec.prompts` (e.g. `"nl2code"`, `"qa"`).
    pub fn embed_query(
        &mut self,
        query: &str,
        task: &str,
        spec: &ModelSpec,
    ) -> Result<Vec<f32>, String> {
        let pair = spec
            .prompts
            .get(task)
            .ok_or_else(|| format!("Unknown prompt task: {task}"))?;

        let prefixed = format!("{}{}", pair.query, query);

        let mut results = self.embed(vec![prefixed], None)?;

        results
            .pop()
            .ok_or_else(|| "No embedding returned".to_string())
    }

    /// tasks from 'spec.prompts'
    pub fn embed_passages(
        &mut self,
        documents: &[String],
        task: &str,
        spec: &ModelSpec,
    ) -> Result<Vec<Vec<f32>>, String> {
        let pair = spec
            .prompts
            .get(task)
            .ok_or_else(|| format!("Unknown prompt task: {task}"))?;

        let prefixed: Vec<String> = documents
            .iter()
            .map(|d| format!("{}{}", pair.document, d))
            .collect();

        self.embed(prefixed, Some(documents.len()))
    }

    // generic embed tasks - passed directly
    pub fn embed_with_prompt(
        &mut self,
        query: &str,
        documents: &[String],
        task: &str,
        spec: &ModelSpec,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        let q = self.embed_query(query, task, spec)?;
        let d = self.embed_passages(documents, task, spec)?;
        Ok((q, d))
    }

    /// embed all code blocks, assigning vectors in-place.
    pub fn embed_code_blocks(
        &mut self,
        blocks: &mut [crate::CodeBlock],
        dims: usize,
    ) -> Result<(), String> {
        let texts: Vec<String> = blocks.iter().map(|b| b.text.clone()).collect();

        let embeddings = self.embed(texts, Some(texts.len()))?;

        for (block, emb) in blocks.iter_mut().zip(embeddings) {
            block.embedding = Some(emb);
        }
        Ok(())
    }
}

/// value in [-1.0, 1.0]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(f32::EPSILON)
}
