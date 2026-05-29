use crate::session::checkpoint::CheckpointStore;
use crate::{StepError, Tracer};
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Executor<S, E> {
    _phantom: PhantomData<E>,
    state: Arc<RwLock<S>>,
    checkpoint_store: Arc<dyn CheckpointStore<S>>,
    tracer: Arc<dyn Tracer>,
    trace_id: Uuid,
}

impl<S, E> Executor<S, E> {
    pub fn new(
        state: S,
        checkpoint_store: Arc<dyn CheckpointStore<S>>,
        tracer: Arc<dyn Tracer>,
    ) -> Self {
        Self {
            state: Arc::new(RwLock::new(state)),
            checkpoint_store,
            tracer,
            trace_id: Uuid::now_v7(),
            _phantom: PhantomData,
        }
    }
}

pub struct StreamContent {
    pub channel: String,
    pub data: Vec<u8>,
}

pub struct StreamChunk {
    pub metadata: String,
    pub content: StreamContent,
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
