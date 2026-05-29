use crate::{Document, IngestionError, Keyword};
use owo_colors::OwoColorize;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tf_idf_vectorizer::{Corpus, TFIDFVectorizer, TermFrequency};

use anyhow::{Context, Result, anyhow};

const STOPWORD_PATH: &str = "/home/roccoluxe/FOSRA/z-misc/resources/stopwords.txt";

pub fn parse_keywords(doc_path: &str, corpus: Option<Arc<Corpus>>) -> Result<Vec<Keyword>> {
    let corpus = corpus.unwrap_or_else(|| Arc::new(Corpus::new()));

    // pull in stopwords
    let sw_string: String = std::fs::read_to_string(STOPWORD_PATH)
        .map_err(|e| format!("Failed to read stopwords file: {e}"))
        .unwrap();
    let stopwords: HashSet<String> = sw_string
        .lines()
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_lowercase())
        .collect();

    // parse doc to string
    let doc: String = std::fs::read_to_string(doc_path).unwrap();
    let doc_name = String::from(Path::new(doc_path).file_name().unwrap().to_str().unwrap());

    // to markdown
    let doc_md = Document::walk_md(doc_path.into()).unwrap();

    // remove inflected variants
    fn is_inflected(word: &str) -> bool {
        if word.len() <= 2 {
            return false;
        }

        word.ends_with("ing")
            || word.ends_with("ies")
            || word.ends_with("ally")
            || word.ends_with('s') && word.len() > 3
    }

    // split doc to unique words minus stopwords + inflected
    let mut doc_split: Vec<String> = doc
        .split_whitespace()
        .filter(|s| !s.is_empty() && s.chars().all(char::is_alphanumeric))
        .map(|s| s.to_lowercase())
        .collect::<Vec<String>>();
    doc_split.retain(|s| !stopwords.contains(s));
    doc_split.retain(|w| !is_inflected(w));

    // extract headings
    let headings = doc_md
        .content
        .iter()
        .map(|s| &s.path)
        .collect::<Vec<&String>>();

    // unique words in headings
    let mut heading_tokens: Vec<&str> = headings
        .iter()
        .flat_map(|h| h.split("::"))
        .flat_map(|p| p.split_whitespace())
        .filter(|&t| !t.contains(".") && !t.chars().any(|c| !c.is_alphabetic()))
        .collect();
    // remove stopwords
    heading_tokens.retain(|s| !stopwords.contains(s.to_lowercase().as_str()));

    // // make term frequencies
    let mut freq1 = TermFrequency::new();
    let mut freq2 = TermFrequency::new();
    freq1.add_terms(&doc_split);
    freq2.add_terms(&heading_tokens);

    // get list for both

    let doc_kw: Vec<(String, u64)> = freq1
        .sorted_frequency_vector()
        .into_iter()
        .filter(|kw| kw.1 >= 7)
        .take(15)
        .collect();

    let heading_kw: Vec<(String, u64)> = freq2
        .sorted_frequency_vector()
        .into_iter()
        .filter(|kw| kw.1 >= 6)
        .take(15)
        .collect();

    // merge for return
    let mut all_kw: Vec<Keyword> = doc_kw.iter().map(|kw| Keyword::new(kw.clone())).collect();
    let mut seen: HashSet<Keyword> = HashSet::from_iter(all_kw.clone());

    for kw in heading_kw {
        let key = Keyword::new(kw);
        if seen.insert(key.clone()) {
            all_kw.push(key)
        }
    }

    // add to corpus together

    let mut vectorizer: TFIDFVectorizer = TFIDFVectorizer::new(corpus);

    vectorizer.add_doc(doc_name.clone(), &freq2);
    vectorizer.add_doc(doc_name, &freq1);

    Ok(all_kw)
}

//TODO: Parse to SURREALDB

// pub fn save(corpus: Arc<Corpus>) -> Result<(), String> {
//     let json = serde_json::to_string_pretty(self)
//         .map_err(|e| format!("Failed to serialize corpus: {e}"))?;

//     std::fs::write(path, json).map_err(|e| format!("Failed to write corpus: {e}"))
// }

// pub fn load(path: &Path) -> Result<Self, String> {
//     let json =
//         std::fs::read_to_string(path).map_err(|e| format!("Failed to read corpus file: {e}"))?;

//     serde_json::from_str(&json).map_err(|e| format!("Failed to deserialize corpus: {e}"))
// }

// pub fn to_data(&self) -> CorpusData {
//     CorpusData {
//         documents: self.documents.clone(),
//         df: self.df.clone(),
//         n_docs: self.n_docs,
//     }
// }

// pub fn from_data(data: CorpusData, stopwords: Vec<String>) -> Self {
//     Self {
//         documents: data.documents,
//         df: data.df,
//         n_docs: data.n_docs,
//         stopwords,
//     }
// }

// // -> JSON
// pub fn save_data(&self, path: &Path) -> Result<(), String> {
//     let data = self.to_data();

//     let json = serde_json::to_string_pretty(&data)
//         .map_err(|e| format!("Failed to serialize corpus data: {e}"))?;

//     std::fs::write(path, json).map_err(|e| format!("Failed to write corpus data: {e}"))
// }

// // <- JSON + Stopwords
// pub fn load_data(path: &Path, stopwords_path: &Path) -> Result<Self, String> {
//     let json = std::fs::read_to_string(path)
//         .map_err(|e| format!("Failed to read corpus data file: {e}"))?;

//     let data: CorpusData = serde_json::from_str(&json)
//         .map_err(|e| format!("Failed to deserialize corpus data: {e}"))?;

//     let stopwords = Corpus::load_stopwords(stopwords_path)?;

//     Ok(Self::from_data(data, stopwords))
// }
