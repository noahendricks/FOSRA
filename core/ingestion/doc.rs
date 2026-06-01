use owo_colors::OwoColorize;
use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::hash::Hasher;
use strum::{AsRefStr, EnumString, VariantArray};
use surrealdb::types::SurrealValue;
use treemd::{self, HeadingNode, parse_markdown};

pub use tree_sitter::{Point as TSPoint, Range as TSRange, Tree as TSTree};

use crate::{IngestionError, code::CodeSource, processing::embedding::EmbeddingEngine};

#[derive(Clone, Debug, Serialize, Deserialize, SurrealValue, Default)]
pub struct Document {
    pub id: String,
    pub file_path: String,
    pub metadata: DocumentMetadata,
    pub content: Option<Vec<Section>>,
    pub embedding: Option<Vec<f32>>,
    pub content_hash: Option<i64>,
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    EnumString,
    AsRefStr,
    Serialize,
    Deserialize,
    SurrealValue,
    VariantArray,
)]
pub enum SupportedFileTypes {
    #[strum(serialize = "md")]
    Markdown,
    #[strum(serialize = "pdf")]
    PDF,
}

use std::fs::Metadata;

#[derive(Debug, Clone, SurrealValue, Default)]
pub struct Folder {
    pub name: String,
    pub base: String,
    pub docs: Vec<Document>,
    pub code_docs: Vec<CodeSource>,
    pub metadata: DocumentMetadata,
    pub parent: Option<Box<Folder>>,
}

use ignore::DirEntry;
impl Folder {
    pub fn from_entry(entry: &DirEntry) -> Option<Self> {
        if !entry.file_type()?.is_dir() {
            return None;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        let base = entry.path().to_string_lossy().into_owned();
        let metadata = entry.metadata().ok()?;

        Some(Self {
            name,
            base,
            docs: Vec::new(),
            metadata: metadata.into(),
            code_docs: Vec::new(),
            parent: None,
        })
    }
}

impl Document {
    pub fn walk_md(path: String) -> Result<Document> {
        println!("{}", "entered walk md".red().bold());
        let doc_str = std::fs::read_to_string(&path).map_err(|e| anyhow!("error {e} "))?;

        // println!("\n \n {}", doc_str.magenta().bold());

        let doc_bytes = doc_str.as_bytes();

        println!("{}", "init tree".magenta().bold());
        let tree = parse_markdown(&doc_str).build_tree();
        println!("{}", (tree.len()).magenta().bold());

        fn walk_children(
            heading: HeadingNode,
            parent_path: &String,
            doc: &[u8],
            sibling_end: &usize,
        ) -> Vec<Section> {
            println!("{}", "init sections".red().bold());
            let mut sections: Vec<Section> = Vec::new();
            println!(
                "{} | {}",
                "init sections".red().bold(),
                !sections.is_empty()
            );

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

        println!("{}", "sections".yellow().bold());
        let sections = if tree.is_empty() {
            Vec::new()
        } else {
            let root = &tree[0];
            walk_children(root.clone(), &root.heading.text, doc_bytes, &0)
        };

        println!("{} | {}", "path repr".red().bold(), !sections.is_empty());
        let path_repr = Path::new(path.as_str());

        let id = path_repr
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();

        // Compute content hash from whitespace-normalized text
        let cleaned: String = doc_str.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut hasher = std::hash::DefaultHasher::new();
        use std::hash::Hash;
        cleaned.hash(&mut hasher);
        let content_hash = hasher.finish() as i64;

        Ok(Document {
            id,
            file_path: path.clone(),
            content: Some(sections),
            metadata: DocumentMetadata {
                path: Some(path.clone()),
                ..Default::default()
            },
            embedding: Some(Vec::new()),
            content_hash: Some(content_hash as i64),
        })
    }

    /// embeds sections in-place and places avg of all vectors on document
    /// ```
    /// Section.embedding //<- mutated for each
    ///
    /// Document.embedding = [Section.embedding,Section.embedding...].avg() // <- average for all
    /// ```
    pub fn embed_document(
        &mut self,
        embedder: &mut EmbeddingEngine,
        dims: usize,
        batch_size: usize,
    ) -> Result<()> {
        let mut content = self.content.clone().ok_or(anyhow!(""))?;
        if content.is_empty() {
            return Ok(());
        }

        let doc_embedding = embedder
            .embed_doc_mut(&mut content, dims, batch_size)
            .map_err(|e| anyhow!("failed because {e}"))?;

        self.embedding = Some(doc_embedding);

        Ok(())
    }
}

use surrealdb::types::kind;

#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
pub struct DocumentMetadata {
    pub title: Option<String>,

    pub path: Option<String>,

    pub subject: Option<String>,

    pub authors: Option<Vec<String>>,

    pub created_at: Option<DateTime<Utc>>,

    pub modified_at: Option<DateTime<Utc>>,

    pub extracted_keywords: Option<Vec<Keyword>>,
}
impl From<Metadata> for DocumentMetadata {
    fn from(meta: Metadata) -> Self {
        fn transform(time: Option<SystemTime>) -> Option<DateTime<Utc>> {
            use chrono::{DateTime, Utc};
            use std::time::{SystemTime, UNIX_EPOCH};

            if !time.is_some() {
                return None;
            }

            fn to_datetime(time: Option<SystemTime>) -> Option<DateTime<Utc>> {
                let duration = time.unwrap().duration_since(UNIX_EPOCH).ok()?;
                let secs = duration.as_secs() as i64;
                let nanos = duration.subsec_nanos() as i32;
                DateTime::from_timestamp(secs, nanos as u32)
            }

            time.unwrap()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs().to_string());

            to_datetime(time)
        }

        let created = transform(Some(meta.created().unwrap()));
        let modified = transform(Some(meta.created().unwrap()));

        Self {
            path: None, // set from caller
            created_at: created,
            modified_at: modified,
            ..Default::default()
        }
    }
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

impl Section {
    pub fn embed(
        &mut self,
        engine: &mut EmbeddingEngine,
        dims: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        println!("{}", "in embed".green().bold());
        let texts = vec![self.text.to_string()];

        let mut embeddings = engine
            .embed(texts, Some(batch_size))
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // if embeddings[0].len() > dims {
        //     embeddings[0].truncate(dims);
        // }
        self.embedding = Some(embeddings[0].clone());
        Ok(embeddings[0].clone())
    }
}
