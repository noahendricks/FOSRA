use crate::{StepError, StepEvent, StepEventKind};
use chrono::{DateTime, Utc};
use std::time::Duration;
use std::{collections::HashMap, fmt::Display, marker::PhantomData, sync::Arc};
use strum;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub trace_id: Uuid,
    pub thread_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub events: Vec<StepEvent>,
}

pub trait Tracer {
    fn emit(&self, event: StepEvent) -> String;
}

pub struct StdOut {
    place: String,
}

impl Tracer for StdOut {
    fn emit(&self, event: StepEvent) -> String {
        format!(
            "[trace:{:#?} {:#?} step={:?} depth={:?}",
            event.trace_id, event.event, event.step_name, event.step_depth
        )
    }
}

pub struct SessionState {
    pub system_prompt: String,
    pub model: String,               //TODO: Model struct
    pub thinking_level: u8,          //TODO: Thinking level Enum
    pub tools: String,               //TODO: Tool Struct
    pub messages: Vec<String>,       //TODO: AgentMessage Struct
    pub is_streaming: bool,          //TODO: determine if properly placed
    pub current_message: String,     //TODO: AgentMessage Struct
    pub pending_tool_callls: String, //TODO: Tool calls or String Type
    pub error: String,               //TODO: determine if properly placed
}

pub struct StreamContent {
    pub channel: String,
    pub data: Vec<u8>,
}

pub struct StreamChunk {
    pub metadata: StepMetadata,
    pub content: StreamContent,
}

pub struct ToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
    //func: Box<dyn Fn(ToolCall, &mut S) -> Result<String, E>>,
}

enum ToolExecution {
    Sequential,
    Parallel,
}

// pub enum ToolErrorPolicy {
//     FailStep,
//     SkipTool,
//     RetryTool(max_retries),
// }
