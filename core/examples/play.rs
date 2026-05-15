//! Tree-sitter debug / dev playground.
//!
//! Usage:
//!   cargo run --package fosra --example play -- <file> [--lang rust|python|typescript|tsx]
//!
//! If no file is given, reads from stdin.
//! If no --lang is given, auto-detects from file extension.

use std::io::{self, Read, Write};

use tree_sitter::{Parser, TreeCursor};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut file_path: Option<String> =
        Some("/home/roccoluxe/fosra-rust/core/examples/play-debug.rs".to_string());
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

    // ── Section 1: Summary ──────────────────────────────────────────
    println!("=== Tree-sitter Debug Output ===");
    println!("Language: {}", lang_label);
    println!("Source bytes: {}", source_code.len());
    println!("Source lines: {}", source_code.lines().count());
    println!("Root kind: {}", root.kind());
    println!("Root has_error: {}", root.has_error());
    println!("Root is_named: {}", root.is_named());
    println!("Root child_count: {}", root.child_count());
    println!("Root named_child_count: {} \n", root.named_child_count());

    let cursor = root.walk();

    // ── Section 2: S-expression ─────────────────────────────────────
    // println!("=== S-expression ===");
    // println!("{}", root.to_sexp());
    // println!();

    // ── Section 3: Full tree walk (field-aware) ─────────────────────
    println!("=== Full Tree Walk ===");
    writeln!(std::io::stdout(), "# `{}`", root.kind()).ok();

    for i in 0..root.named_child_count() {
        if let Some(child) = root.named_child(i as u32) {
            let child_field = root.field_name_for_child(i as u32).unwrap_or("");
            print_tree_markdown(
                &mut std::io::stdout(),
                child,
                &source_code,
                cursor.clone(),
                1,
                child_field,
            );
        }
    }

    // ── Section 4: Cursor-based traversal ───────────────────────────
    println!("=== Cursor Traversal ===");
    cursor_walk(&mut std::io::stdout(), &tree, &source_code);
    println!();

    // ── Section 6: Named node index ─────────────────────────────────
    println!("=== Named Nodes (by depth) ===");
    print_named_index(&mut std::io::stdout(), root, &source_code, 0);
}

fn print_tree_markdown(
    out: &mut impl Write,
    node: tree_sitter::Node,
    source_code: &str,
    cursor: TreeCursor,
    depth: usize,
    parent_field: &str,
) {
    let node_id = node.id();
    let byte_range = node.byte_range();
    let start = node.start_position();
    let end = node.end_position();
    let named_flag = if node.is_named() { "" } else { " (anonymous)" };
    let error_flag = if node.has_error() { " [!]" } else { "" };
    let field_info = if parent_field.is_empty() {
        String::new()
    } else {
        format!("**{}:** ", parent_field)
    };
    let indent = "  ".repeat(depth);

    let node_name = if node.grammar_name() == "identifier" {
        node.utf8_text(source_code.as_bytes()).unwrap_or("default")
    } else {
        "no name"
    };

    let node_text = node.utf8_text(source_code.as_bytes()).unwrap();

    writeln!(
        out,
        "{}- {}{} {} `{}` (id={}, bytes={}-{}, ({},{})-({},{}){}){} ",
        indent,
        field_info,
        node.kind(),
        node.grammar_name(),
        node_id,
        byte_range.start,
        byte_range.end,
        start.row,
        start.column,
        end.row,
        end.column,
        named_flag,
        error_flag,
        node_text
    )
    .ok();

    for child in node.named_children(&mut cursor.clone()) {
        let child_field = node.kind();
        print_tree_markdown(
            out,
            child,
            source_code,
            cursor.clone(),
            depth + 1,
            child_field,
        );
    }
}

/// Walk the tree using TreeCursor, showing field names at each step.
fn cursor_walk(out: &mut impl Write, tree: &tree_sitter::Tree, source: &str) {
    let mut cursor = tree.walk();
    cursor_walk_node(out, &mut cursor, source, 0);
}

fn cursor_walk_node(
    out: &mut impl Write,
    cursor: &mut tree_sitter::TreeCursor,
    source: &str,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    let node = cursor.node();
    let field = cursor.field_name().unwrap_or("");

    let text = &source[node.start_byte()..node.end_byte()];

    let display_text = if text.len() > 60 {
        format!("{}", &text)
    } else {
        text.to_string()
    };
    let display_text = display_text.replace('\n', "\\n");

    if depth == 0 {
        writeln!(
            out,
            "{}[depth={}] {} '{}'",
            indent,
            depth,
            node.kind(),
            display_text
        )
        .ok();
    } else {
        writeln!(
            out,
            "{}[depth={}] {} {} -> '{}'",
            indent,
            depth,
            field,
            node.kind(),
            display_text
        )
        .ok();
    }

    if cursor.goto_first_child() {
        loop {
            cursor_walk_node(out, cursor, source, depth + 1);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

/// Print a compact index of all named nodes, grouped by depth.
fn print_named_index(out: &mut impl Write, node: tree_sitter::Node, source: &str, depth: usize) {
    if node.is_named() {
        let text = &source[node.start_byte()..node.end_byte()];
        let display = if text.len() > 50 {
            format!("{}...", &text)
        } else {
            text.to_string()
        };
        let display = display.replace('\n', "\\n");
        writeln!(out, "  {:>3} | {:20} | {:?}", depth, node.kind(), display).ok();
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            print_named_index(out, child, source, depth + 1);
        }
    }
}
