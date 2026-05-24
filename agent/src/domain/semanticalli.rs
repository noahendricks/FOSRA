use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRIntent {
    pub intent_type: IntentType,
    pub scope: Scope,
    pub complexity: Complexity,
    pub entity_types: Vec<String>,
    pub language_hint: Option<String>,
    pub intent_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VSImplementation {
    pub strategy: RetrievalStrategy,
    pub namespace: Namespace,
    pub rerank_decision: RerankDecision,
    pub recall_estimate: f64,
    pub implementation_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentType {
    CodeSearch,
    DocSearch,
    SymbolLookup,
    Refactor,
    Debug,
    Explain,
    Compare,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Scope {
    File,
    Module,
    Package,
    Codebase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Complexity {
    Simple,
    Complex,
    MultiPart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    Exact,
    Semantic,
    Hybrid,
    MultiHop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Namespace {
    Code,
    Docs,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RerankDecision {
    Skip,
    Optional,
    Required,
}
