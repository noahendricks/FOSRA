use std::path::PathBuf;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::hash::Hasher;
use surrealdb::types::SurrealValue;
use treemd::{self, HeadingNode, parse_markdown};

pub use tree_sitter::{Point as TSPoint, Range as TSRange, Tree as TSTree};

use crate::{IngestionError, ingestion::IsSection, processing::embedding::EmbeddingEngine};

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue)]
pub struct Document {
    pub id: String,
    pub content: Vec<Section>,
    pub metadata: DocumentMetadata,
    pub embedding: Vec<f32>,
    pub content_hash: u64,
}

impl Document {
    pub fn walk_md(path: PathBuf) -> Result<Document> {
        let doc_str = std::fs::read_to_string(&path).map_err(|e| IngestionError::Parsing {
            message: format!("reading doc: {}", path.display()),
            source: e.into(),
        })?;

        let doc_bytes = doc_str.as_bytes();

        let tree = parse_markdown(&doc_str).build_tree();

        fn walk_children(
            heading: HeadingNode,
            parent_path: &String,
            doc: &[u8],
            sibling_end: &usize,
        ) -> Vec<Section> {
            let mut sections: Vec<Section> = Vec::new();

            let updated_path = format!(
                "H{}::{}::{}",
                heading.heading.level.to_string().as_str(),
                &heading.heading.text,
                parent_path.clone(),
            );

            let start = *sibling_end;
            let end = heading.heading.offset;

            let curr = Section {
                path: updated_path,
                level: heading.heading.level,
                text: String::from_utf8(doc[start..end].to_vec()).unwrap(),
                start,
                end,
                p_path: parent_path.clone(),
                embedding: None,
            };

            sections.push(curr.clone());

            if !heading.children.is_empty() {
                for child in heading.children {
                    let walk_out = walk_children(
                        child,
                        &curr.path,
                        doc,
                        &sections.last().unwrap_or(&curr).end,
                    );
                    sections.extend(walk_out);
                }
            }
            sections
        }

        let sections = walk_children(tree[0].clone(), &tree[0].heading.text, doc_bytes, &0);

        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();

        // Compute content hash from whitespace-normalized text
        let cleaned: String = doc_str.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut hasher = std::hash::DefaultHasher::new();
        use std::hash::Hash;
        cleaned.hash(&mut hasher);
        let content_hash = hasher.finish();

        Ok(Document {
            id,
            content: sections,
            metadata: DocumentMetadata {
                path: Some(String::from(path.to_str().unwrap())),
                ..Default::default()
            },
            embedding: Vec::new(),
            content_hash,
        })
    }

    /// embeds sections in-place and places avg of all vectors on document
    /// ```
    /// Section.embedding //<- mutated for each
    ///
    /// Document.embedding = [Section.embedding,Section.embedding...].avg() // <- average for all
    /// ```
    pub fn embed_document_mut(
        &mut self,
        embedder: &mut EmbeddingEngine,
        dims: usize,
        batch_size: usize,
    ) -> Result<()> {
        let doc_embedding = embedder
            .embed_document(&mut self.content, batch_size)
            .map_err(|e| anyhow!("failed because {e}"))?;

        self.embedding = doc_embedding;

        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]

pub struct Section {
    pub path: String,
    pub level: usize,
    pub text: String,
    pub start: usize,
    pub end: usize,

    pub p_path: String,
    pub embedding: Option<Vec<f32>>,
}

impl IsSection for Section {
    fn text(&self) -> &str {
        &self.text
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
pub struct DocumentMetadata {
    pub title: Option<String>,

    pub path: Option<String>,

    pub subject: Option<String>,

    pub authors: Option<Vec<String>>,

    pub created_at: Option<String>,

    pub modified_at: Option<String>,

    pub extracted_keywords: Option<Vec<Keyword>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue, Hash, PartialEq, Eq)]
pub struct Keyword {
    pub text: String,
    pub score: u64,
}

impl Keyword {
    pub fn new(keyword_pair: (String, u64)) -> Keyword {
        Self {
            text: keyword_pair.0,
            score: keyword_pair.1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadingContext {
    pub headings: Vec<HeadingLevel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadingLevel {
    pub level: u8,
    pub text: String,
}
