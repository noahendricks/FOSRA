use anyhow::Result;
use rig;

use rig::completion::{Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::{Tool, ToolDyn};
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Add;

const DEFAULT_API_BASE_URL: &str = "http://localhost:8050/v1";
const DEFAULT_API_KEY: &str = "none";
const DEFAULT_MODEL: &str = "model";

fn api_base_url() -> String {
    std::env::var("LLAMACPP_API_BASE_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string())
}

fn api_key() -> String {
    std::env::var("LLAMACPP_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string())
}

fn model_name() -> String {
    std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

fn client() -> openai::Client {
    let api_key = api_key();
    let base_url = api_base_url();

    openai::Client::from_url(&DEFAULT_API_KEY, &DEFAULT_API_BASE_URL)
}

fn completions_client() -> openai::CompletionModel {
    client().completion_model(DEFAULT_MODEL)
}

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number to add" },
                    "y": { "type": "number", "description": "The second number to add" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("CALLL");
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The number to subtract from" },
                    "y": { "type": "number", "description": "The number to subtract" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

fn boxed_tools() -> Vec<Box<dyn ToolDyn>> {
    vec![Box::new(Add), Box::new(Subtract)]
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = client();

    let agent = client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .max_tokens(1024)
        .build();

    let response = agent.prompt("Calculate 2 - 5.").await?;
    println!("{response}");

    Ok(())
}
