//! Tree-sitter node type inspector.
//!
//! Usage:
//!   cargo run --package fosra --example play-debug -- <file> [--lang rust|python|typescript|tsx]
//!
//! Outputs the full `dbg_pls` representation of one representative node per unique node kind.
//! This lets you see all available fields for each node type in the grammar.
use std::io::{self, Read};

use dbg_pls::{DebugPls, color};
use std::collections::HashSet;
use tree_sitter::Parser;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut file_path: Option<String> = None;
    let mut lang: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" => {
                i += 1;
                if i < args.len() {
                    lang = Some(args[i].clone());
                }
            }
            _ if args[i].starts_with("--") => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
            _ => file_path = Some(args[i].clone()),
        }
        i += 1;
    }

    // Resolve source
    let source_code = if let Some(ref path) = file_path {
        std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read '{}': {}", path, e);
            std::process::exit(1);
        })
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
            eprintln!("Failed to read stdin: {}", e);
            std::process::exit(1);
        });
        buf
    };

    // Determine language
    let lang_name = lang
        .unwrap_or_else(|| {
            file_path
                .as_ref()
                .and_then(|p| std::path::Path::new(p).extension())
                .and_then(|e| e.to_str().map(|s| s.to_lowercase()))
                .unwrap_or_else(|| "rust".to_string())
        })
        .to_lowercase();

    let (language, lang_label) = match lang_name.as_str() {
        "rust" | "rs" => (tree_sitter_rust::LANGUAGE.into(), "Rust (tree-sitter-rust)"),
        "python" | "py" => (
            tree_sitter_python::LANGUAGE.into(),
            "Python (tree-sitter-python)",
        ),
        "typescript" | "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "TypeScript (tree-sitter-typescript)",
        ),
        "tsx" => (
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            "TSX (tree-sitter-typescript)",
        ),
        _ => {
            eprintln!(
                "Unknown language '{}'. Supported: rust, python, typescript, tsx",
                lang_name
            );
            std::process::exit(1);
        }
    };

    // Parse
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .expect("Error loading grammar");

    let tree = parser.parse(&source_code, None).expect("Parse failed");

    let root = tree.root_node();

    println!("=== Node Type Inspector ===");
    println!("Language: {}", lang_label);
    println!("File: {:?}", file_path);
    println!("Source bytes: {}", source_code.len());
    println!("Root kind: {}", root.kind());
    println!();

    // Collect one node per unique kind (store FullNode instead of Node to avoid lifetime issues)
    let mut seen_kinds: HashSet<String> = HashSet::new();
    let mut representatives: Vec<FullNode> = Vec::new();

    collect_representatives(root, &source_code, &mut seen_kinds, &mut representatives);

    println!("=== Unique Node Types ({}) ===", representatives.len());
    println!();

    for node in representatives {
        println!("--- {} ---", node.kind);
        println!("{}", color(&node));
        println!();
    }
}

fn collect_representatives(
    node: tree_sitter::Node,
    source_code: &str,
    seen: &mut HashSet<String>,
    reps: &mut Vec<FullNode>,
) {
    let kind = node.kind().to_string();
    if !seen.contains(&kind) {
        seen.insert(kind.clone());
        reps.push(FullNode::from_node(node, source_code));
    }

    // Only iterate named children
    for i in 0..node.named_child_count() as u32 {
        if let Some(child) = node.named_child(i) {
            collect_representatives(child, source_code, seen, reps);
        }
    }
}

/// Aggregated node info for dbg_pls output
#[derive(DebugPls)]
struct FullNode {
    kind: String,
    identifier: Option<String>,
    grammar_name: String,
    is_named: bool,
    has_error: bool,
    is_extra: bool,
    is_missing: bool,
    start_byte: usize,
    end_byte: usize,
    start_point: (usize, usize),
    end_point: (usize, usize),
    id: usize,
    child_count: usize,
    named_child_count: usize,
    text_snippet: String,
}

impl FullNode {
    fn from_node(node: tree_sitter::Node, source_code: &str) -> Self {
        let snippet = source_code
            .get(node.byte_range())
            .map(|s| {
                if s.len() > 100 {
                    format!("{}...", &s[..100])
                } else {
                    s.to_string()
                }
            })
            .unwrap_or_default();

        // Find the identifier named child if any
        let identifier = find_identifier(node, source_code);

        Self {
            kind: node.kind().to_string(),
            identifier,
            grammar_name: node.grammar_name().to_string(),
            is_named: node.is_named(),
            has_error: node.has_error(),
            is_extra: node.is_extra(),
            is_missing: node.is_missing(),
            start_byte: node.start_byte() as usize,
            end_byte: node.end_byte() as usize,
            start_point: (node.start_position().row, node.start_position().column),
            end_point: (node.end_position().row, node.end_position().column),
            id: node.id() as usize,
            child_count: node.child_count() as usize,
            named_child_count: node.named_child_count() as usize,
            text_snippet: snippet,
        }
    }
}

/// Recursively find an identifier child node and return its text
fn find_identifier(node: tree_sitter::Node, source: &str) -> Option<String> {
    for i in 0..node.named_child_count() as u32 {
        if let Some(child) = node.named_child(i) {
            let kind = child.kind();
            if kind == "identifier" || kind == "field_identifier" || kind == "type_identifier" {
                let text = source.get(child.byte_range())?.to_string();
                return Some(text);
            }
            // Check nested children
            if let Some(found) = find_identifier(child, source) {
                return Some(found);
            }
        }
    }
    None
}
