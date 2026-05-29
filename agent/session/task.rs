// dalia paper

use serde::{Deserialize, Serialize};

use crate::core::tracer::ValidationStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub preconditions: Vec<Condition>,
    pub postconditions: Vec<Condition>,
    pub inputs: Vec<Parameter>,
    pub outputs: Vec<Parameter>,
    pub cost_estimate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub entity: String,
    pub relation: Relation,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Relation {
    Equals,
    Contains,
    Exists,
    NotExists,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    pub capability: Capability,
    pub dependencies: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub dag: Vec<PlanNode>,
    pub missing_capabilities: Vec<String>,
    pub total_estimated_cost: f64,
    pub validation_status: ValidationStatus,
}
