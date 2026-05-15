// Example: parsed `CodeSource` from `core/examples/code.rs`
// When tree-sitter parsing is complete, this structure represents a fully ingested code file.

use fosra::types::{
    BlockType, CodeBlock, CodeSource, ImportGroup, LspSymbolInfo, Param, ScopePath, ScopeSegment,
    Signature, SupportedLanguage, Symbol, Visibility,
};
use std::path::PathBuf;
use tree_sitter::Range;

fn parsed_code_source_example() -> CodeSource {
    CodeSource {
        path: PathBuf::from("/Users/dev/fosra-rust/core/examples/code.rs"),
        language: SupportedLanguage::Rust,
        module_coment: Some(
            "Tree-sitter node type inspector.\n\
             \n\
             Usage:\n\
             \n  cargo run --package fosra --example play-debug -- <file> [--lang rust|python|typescript|tsx]\n\
             \n\
             Outputs the full `dbg_pls` representation of one representative node per unique node kind.\n\
             This lets you see all available fields for each node type in the grammar."
                .to_string(),
        ),
        imports: vec![
            ImportGroup {
                text: "use std::io::{self, Read};".to_string(),
                comment: None,
                byte_range: Range {
                    start_byte: 185,
                    end_byte: 206,
                    start_point: tree_sitter::Point { row: 10, column: 0 },
                    end_point: tree_sitter::Point { row: 10, column: 21 },
                },
            },
            ImportGroup {
                text: "use std::path::Path;".to_string(),
                comment: Some("Path manipulation for file resolution".to_string()),
                byte_range: Range {
                    start_byte: 208,
                    end_byte: 231,
                    start_point: tree_sitter::Point { row: 11, column: 0 },
                    end_point: tree_sitter::Point { row: 11, column: 23 },
                },
            },
        ],
        blocks: vec![
            // ── CodeBlock: impl block for FullNode (module-level container) ────────
            // NOTE: impl_item maps to BlockType::Container in RUST block_types
            CodeBlock {
                block_id: "play_debug".to_string(),
                file: "core/examples/code.rs".to_string(),
                range: Some(Range {
                    start_byte: 0,
                    end_byte: 4000,
                    start_point: tree_sitter::Point { row: 0, column: 0 },
                    end_point: tree_sitter::Point { row: 216, column: 0 },
                }),
                kind: BlockType::Container, // mod_item → (Container, Module)
                scope_path: ScopePath(vec![
                    ScopeSegment {
                        name: "play_debug".to_string(),
                        kind: BlockType::Container,
                        self_type: None,
                    },
                ]),
                parent_id: None,
                visibility: Visibility::Crate,
                signature: None,
                attributes: vec![],
                comment: Some("Tree-sitter node type inspector.".to_string()),
                text: "// ![ ... full module source ... ]".to_string(),
                children: vec![
                    // ── CodeBlock: function `main` ─────────────────────────────────
                    // function_item → (Atomic, Function)
                    CodeBlock {
                        block_id: "play_debug::main".to_string(),
                        file: "core/examples/code.rs".to_string(),
                        range: Some(Range {
                            start_byte: 312,
                            end_byte: 2847,
                            start_point: tree_sitter::Point { row: 123, column: 0 },
                            end_point: tree_sitter::Point { row: 142, column: 1 },
                        }),
                        kind: BlockType::Atomic,
                        scope_path: ScopePath(vec![
                            ScopeSegment {
                                name: "play_debug".to_string(),
                                kind: BlockType::Container,
                                self_type: None,
                            },
                            ScopeSegment {
                                name: "main".to_string(),
                                kind: BlockType::Atomic,
                                self_type: None,
                            },
                        ]),
                        parent_id: Some("play_debug".to_string()),
                        visibility: Visibility::Public,
                        signature: Some(Signature {
                            raw: "fn main() -> Result<()>".to_string(),
                            params: vec![Param {
                                name: None,
                                ty: None,
                                default: None,
                            }],
                            return_type: Some("Result<()>".to_string()),
                            type_params: vec![],
                            bases: vec![],
                        }),
                        attributes: vec![],
                        comment: None,
                        text: "fn main() -> Result<()> {\n    let code_file = env::current_dir()?...".to_string(),
                        children: vec![
                            // ── CodeBlock: nested function `collect_representatives` ──
                            CodeBlock {
                                block_id: "play_debug::main::collect_representatives".to_string(),
                                file: "core/examples/code.rs".to_string(),
                                range: Some(Range {
                                    start_byte: 892,
                                    end_byte: 2156,
                                    start_point: tree_sitter::Point { row: 124, column: 0 },
                                    end_point: tree_sitter::Point { row: 142, column: 22 },
                                }),
                                kind: BlockType::Atomic,
                                scope_path: ScopePath(vec![
                                    ScopeSegment {
                                        name: "play_debug".to_string(),
                                        kind: BlockType::Container,
                                        self_type: None,
                                    },
                                    ScopeSegment {
                                        name: "main".to_string(),
                                        kind: BlockType::Atomic,
                                        self_type: None,
                                    },
                                    ScopeSegment {
                                        name: "collect_representatives".to_string(),
                                        kind: BlockType::Atomic,
                                        self_type: None,
                                    },
                                ]),
                                parent_id: Some("play_debug::main".to_string()),
                                visibility: Visibility::Private,
                                signature: Some(Signature {
                                    raw: "fn collect_representatives(\n    node: tree_sitter::Node,\n    source_code: &str,\n    seen: &mut HashSet<String>,\n    reps: &mut Vec<FullNode>,\n)"
                                        .to_string(),
                                    params: vec![
                                        Param {
                                            name: Some("node".to_string()),
                                            ty: Some("tree_sitter::Node".to_string()),
                                            default: None,
                                        },
                                        Param {
                                            name: Some("source_code".to_string()),
                                            ty: Some("&str".to_string()),
                                            default: None,
                                        },
                                        Param {
                                            name: Some("seen".to_string()),
                                            ty: Some("&mut HashSet<String>".to_string()),
                                            default: None,
                                        },
                                        Param {
                                            name: Some("reps".to_string()),
                                            ty: Some("&mut Vec<FullNode>".to_string()),
                                            default: None,
                                        },
                                    ],
                                    return_type: None,
                                    type_params: vec![],
                                    bases: vec![],
                                }),
                                attributes: vec![],
                                comment: Some(
                                    "Recursively collect representative nodes for each unique node kind."
                                        .to_string(),
                                ),
                                text: "fn collect_representatives(...) { ... }".to_string(),
                                children: vec![],
                                symbols: vec![],
                                lsp: None,
                            },
                        ],
                        symbols: vec![],
                        lsp: Some(LspSymbolInfo {
                            block_id: "play_debug::main".to_string(),
                            calls: vec![],
                            called_by: vec![],
                            references: vec![],
                            type_refs: vec![],
                            overrides: vec![],
                        }),
                    },
                    // ── CodeBlock: struct `FullNode` ────────────────────────────────
                    // struct_item → (Atomic, Struct)
                    CodeBlock {
                        block_id: "play_debug::FullNode".to_string(),
                        file: "core/examples/code.rs".to_string(),
                        range: Some(Range {
                            start_byte: 2341,
                            end_byte: 2567,
                            start_point: tree_sitter::Point { row: 145, column: 0 },
                            end_point: tree_sitter::Point { row: 162, column: 8 },
                        }),
                        kind: BlockType::Atomic,
                        scope_path: ScopePath(vec![
                            ScopeSegment {
                                name: "play_debug".to_string(),
                                kind: BlockType::Container,
                                self_type: None,
                            },
                            ScopeSegment {
                                name: "FullNode".to_string(),
                                kind: BlockType::Atomic,
                                self_type: None,
                            },
                        ]),
                        parent_id: Some("play_debug".to_string()),
                        visibility: Visibility::Crate,
                        signature: None,
                        attributes: vec!["#[derive(DebugPls)]".to_string()],
                        comment: Some("Aggregated node info for dbg_pls output".to_string()),
                        text: "#[derive(DebugPls)]\nstruct FullNode { ... }".to_string(),
                        children: vec![
                            // ── CodeBlock: field `kind` (statement-level) ───────────
                            CodeBlock {
                                block_id: "play_debug::FullNode::kind".to_string(),
                                file: "core/examples/code.rs".to_string(),
                                range: Some(Range {
                                    start_byte: 2430,
                                    end_byte: 2458,
                                    start_point: tree_sitter::Point { row: 148, column: 4 },
                                    end_point: tree_sitter::Point { row: 148, column: 32 },
                                }),
                                kind: BlockType::Statement,
                                scope_path: ScopePath(vec![
                                    ScopeSegment {
                                        name: "play_debug".to_string(),
                                        kind: BlockType::Container,
                                        self_type: None,
                                    },
                                    ScopeSegment {
                                        name: "FullNode".to_string(),
                                        kind: BlockType::Atomic,
                                        self_type: None,
                                    },
                                    ScopeSegment {
                                        name: "kind".to_string(),
                                        kind: BlockType::Statement,
                                        self_type: Some("String".to_string()),
                                    },
                                ]),
                                parent_id: Some("play_debug::FullNode".to_string()),
                                visibility: Visibility::Public,
                                signature: None,
                                attributes: vec![],
                                comment: None,
                                text: "kind: String,".to_string(),
                                children: vec![],
                                symbols: vec![],
                                lsp: None,
                            },
                        ],
                        symbols: vec![
                            Symbol {
                                name: "kind".to_string(),
                                kind: BlockType::Statement,
                                byte_offset: 2430,
                            },
                            Symbol {
                                name: "source".to_string(),
                                kind: BlockType::Statement,
                                byte_offset: 2464,
                            },
                        ],
                        lsp: None,
                    },
                    // ── CodeBlock: impl block for FullNode ──────────────────────────
                    // impl_item → (Container, Impl)
                    CodeBlock {
                        block_id: "play_debug::FullNode::impl".to_string(),
                        file: "core/examples/code.rs".to_string(),
                        range: Some(Range {
                            start_byte: 2580,
                            end_byte: 3012,
                            start_point: tree_sitter::Point { row: 164, column: 0 },
                            end_point: tree_sitter::Point { row: 198, column: 1 },
                        }),
                        kind: BlockType::Container,
                        scope_path: ScopePath(vec![
                            ScopeSegment {
                                name: "play_debug".to_string(),
                                kind: BlockType::Container,
                                self_type: None,
                            },
                            ScopeSegment {
                                name: "FullNode".to_string(),
                                kind: BlockType::Container,
                                self_type: Some("FullNode".to_string()),
                            },
                        ]),
                        parent_id: Some("play_debug".to_string()),
                        visibility: Visibility::Crate,
                        signature: Some(Signature {
                            raw: "impl FullNode".to_string(),
                            params: vec![],
                            return_type: None,
                            type_params: vec![],
                            bases: vec!["FullNode".to_string()],
                        }),
                        attributes: vec![],
                        comment: None,
                        text: "impl FullNode { ... }".to_string(),
                        children: vec![],
                        symbols: vec![],
                        lsp: None,
                    },
                    // ── CodeBlock: function `find_identifier` ──────────────────────
                    // function_item → (Atomic, Function)
                    CodeBlock {
                        block_id: "play_debug::find_identifier".to_string(),
                        file: "core/examples/code.rs".to_string(),
                        range: Some(Range {
                            start_byte: 3015,
                            end_byte: 3321,
                            start_point: tree_sitter::Point { row: 200, column: 0 },
                            end_point: tree_sitter::Point { row: 216, column: 31 },
                        }),
                        kind: BlockType::Atomic,
                        scope_path: ScopePath(vec![
                            ScopeSegment {
                                name: "play_debug".to_string(),
                                kind: BlockType::Container,
                                self_type: None,
                            },
                            ScopeSegment {
                                name: "find_identifier".to_string(),
                                kind: BlockType::Atomic,
                                self_type: None,
                            },
                        ]),
                        parent_id: Some("play_debug".to_string()),
                        visibility: Visibility::Private,
                        signature: Some(Signature {
                            raw: "fn find_identifier(node: tree_sitter::Node, source: &str) -> Option<String>"
                                .to_string(),
                            params: vec![
                                Param {
                                    name: Some("node".to_string()),
                                    ty: Some("tree_sitter::Node".to_string()),
                                    default: None,
                                },
                                Param {
                                    name: Some("source".to_string()),
                                    ty: Some("&str".to_string()),
                                    default: None,
                                },
                            ],
                            return_type: Some("Option<String>".to_string()),
                            type_params: vec![],
                            bases: vec![],
                        }),
                        attributes: vec![],
                        comment: Some(
                            "Recursively find an identifier child node and return its text"
                                .to_string(),
                        ),
                        text: "/// Recursively find an identifier child node and return its text\n\
                               fn find_identifier(...) -> Option<String> { ... }"
                            .to_string(),
                        children: vec![],
                        symbols: vec![],
                        lsp: None,
                    },
                ],
                symbols: vec![],
                lsp: None,
            },
        ],
    }
}

fn main() {
    let _code_source = parsed_code_source_example();
    // In a real parsed scenario, there would also be:
    // - Comment blocks (BlockType::Comment) for doc comments like "//! Tree-sitter node type inspector."
    // - Attribute blocks for #[derive(...)] decorators
    // - Statement blocks for use declarations, imports
}
