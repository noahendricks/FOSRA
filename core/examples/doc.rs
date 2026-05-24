use owo_colors::OwoColorize;

use serde::{Deserialize, Serialize};
use std::{fs::read_to_string, path::PathBuf};
use treemd::{self, HeadingNode, parse_markdown};

use fosra::ingestion::types::Document;

use tf_idf_vectorizer;

pub fn main() {}
