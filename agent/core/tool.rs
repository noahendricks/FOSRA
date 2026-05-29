pub struct ToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
    //func: Box<dyn Fn(ToolCall, &mut S) -> Result<String, E>>,
}

pub enum ToolExecution {
    Sequential,
    Parallel,
}

pub enum ToolName {}

// pub enum ToolErrorPolicy {
//     FailStep,
//     SkipTool,
//     RetryTool(max_retries),
// }
