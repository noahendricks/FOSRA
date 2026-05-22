use owo_colors::OwoColorize;

use serde::{Deserialize, Serialize};
use std::{fs::read_to_string, path::PathBuf};
use treemd::{self, HeadingNode, parse_markdown};

use tf_idf_vectorizer;

use crate::ingestion::types::Document;

use tf_idf_vectorizer;

pub fn doc_ingest() -> Document {
    let path = PathBuf::from("/home/roccoluxe/FOSRA/z-misc/sample-files/sample-md.md");
    Document::walk_md(path).unwrap()
}
