use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Action {
    Quit,

    Tick,

    Render,

    Resize(u16, u16),

    FocusGained,

    FocusLost,

    FocusNext,

    ScrollUp,

    ScrollDown,

    InputChar(char),

    InputBackspace,

    InputClear,

    Submit(String),

    LLMToken(String),

    LLMDone,

    LLMError(String),

    Chunks(Vec<Chunk>),

    Error(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub source: String,
    pub excerpt: String,
    pub score: f32,
}
