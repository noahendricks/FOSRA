use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    documents: HashMap<String, HashMap<String, usize>>,
    // document frequency per term / how many documents contain this term
    df: HashMap<String, usize>,
    // number of documents in corpus
    n_docs: usize,
    /// Stop words loaded from file.
    stopwords: Vec<String>,
}

/// serializable for data-only (no stopwords)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusData {
    // per-document term frequencies: doc_id -> term -> count.
    pub documents: HashMap<String, HashMap<String, usize>>,
    // document frequency / how many documents contain this term.
    pub df: HashMap<String, usize>,
    // total number of documents in corpus
    pub n_docs: usize,
}

impl Corpus {
    fn load_stopwords(path: &Path) -> Result<Vec<String>, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read stopwords file: {e}"))?;

        Ok(content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_lowercase())
            .collect())
    }

    pub fn new(stopwords_path: &Path) -> Result<Self, String> {
        let stopwords = Self::load_stopwords(stopwords_path)?;
        Ok(Self {
            documents: HashMap::new(),
            df: HashMap::new(),
            n_docs: 0,
            stopwords,
        })
    }

    pub fn stopwords(&self) -> &[String] {
        &self.stopwords
    }

    fn tokenize(text: &str, stopwords: &[String]) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3 && !stopwords.contains(w))
            .collect()
    }

    pub fn add_doc(&mut self, id: String, text: &str) {
        let tokens = Self::tokenize(text, &self.stopwords);

        let mut tf: HashMap<String, usize> = HashMap::new();

        for token in &tokens {
            *tf.entry(token.clone()).or_default() += 1;
        }

        for term in tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }

        self.documents.insert(id, tf);
        self.n_docs += 1;
    }

    pub fn add_document(&mut self, doc: &crate::Document) {
        for section in &doc.content {
            let id = format!("{}::{}", doc.id, section.path);

            self.add_doc(id, &section.text);
        }
    }

    pub fn remove_doc(&mut self, id: &str) -> bool {
        if let Some(tf) = self.documents.remove(id) {
            for term in tf.keys() {
                if let Some(count) = self.df.get_mut(term) {
                    *count -= 1;

                    if *count == 0 {
                        self.df.remove(term);
                    }
                }
            }

            self.n_docs -= 1;
            true
        } else {
            false
        }
    }

    pub fn remove_document(&mut self, doc: &crate::Document) {
        for section in &doc.content {
            let id = format!("{}::{}", doc.id, section.path);

            self.remove_doc(&id);
        }
    }

    pub fn contains(&self, id: &str) -> bool {
        self.documents.contains_key(id)
    }

    pub fn term_freq(&self, id: &str) -> Option<&HashMap<String, usize>> {
        self.documents.get(id)
    }

    pub fn len(&self) -> usize {
        self.n_docs
    }

    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    fn tfidf_scores(&self, id: &str) -> Vec<(String, f32)> {
        let tf = match self.documents.get(id) {
            Some(tf) => tf,
            None => return Vec::new(),
        };

        let n = self.n_docs as f32;
        let mut scores: Vec<(String, f32)> = Vec::with_capacity(tf.len());

        for (term, count) in tf {
            let tf_val = *count as f32;
            let df = *self.df.get(term).unwrap_or(&1) as f32;
            let idf = (n / df).ln() + 1.0;

            scores.push((term.clone(), tf_val * idf));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        return scores;
    }

    pub fn keywords(&self, id: &str, max: usize) -> Vec<crate::Keyword> {
        self.tfidf_scores(id)
            .into_iter()
            .take(max)
            .map(|(text, score)| crate::Keyword { text, score })
            .collect()
    }

    pub fn populate_keywords(&self, doc: &mut crate::Document, max: usize) {
        for section in &mut doc.content {
            let id = format!("{}::{}", doc.id, section.path);

            section.keywords = self.keywords(&id, max);
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize corpus: {e}"))?;

        std::fs::write(path, json).map_err(|e| format!("Failed to write corpus: {e}"))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read corpus file: {e}"))?;

        serde_json::from_str(&json).map_err(|e| format!("Failed to deserialize corpus: {e}"))
    }

    pub fn to_data(&self) -> CorpusData {
        CorpusData {
            documents: self.documents.clone(),
            df: self.df.clone(),
            n_docs: self.n_docs,
        }
    }

    pub fn from_data(data: CorpusData, stopwords: Vec<String>) -> Self {
        Self {
            documents: data.documents,
            df: data.df,
            n_docs: data.n_docs,
            stopwords,
        }
    }

    // -> JSON
    pub fn save_data(&self, path: &Path) -> Result<(), String> {
        let data = self.to_data();

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| format!("Failed to serialize corpus data: {e}"))?;

        std::fs::write(path, json).map_err(|e| format!("Failed to write corpus data: {e}"))
    }

    // <- JSON + Stopwords
    pub fn load_data(path: &Path, stopwords_path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read corpus data file: {e}"))?;

        let data: CorpusData = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize corpus data: {e}"))?;

        let stopwords = Corpus::load_stopwords(stopwords_path)?;

        Ok(Self::from_data(data, stopwords))
    }
}
