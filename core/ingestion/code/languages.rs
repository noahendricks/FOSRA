use owo_colors::OwoColorize;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::{collections::HashMap, hash::Hash};

use crate::print_node_tree;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use strum::{AsRefStr, Display, EnumString, VariantArray};
use surrealdb::types::SurrealValue;
use tree_sitter::{Language, Node, TreeCursor};
pub use tree_sitter::{Point as TSPoint, Range as TSRange, Tree as TSTree};

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct CodeBlock {
    pub block_id: String,
    pub block_info: BlockInfo,
    pub range: Option<Range>,
    pub scope_path: ScopePath,
    pub root: String,
    pub text: String,
    pub lsp: Option<LspSymbolInfo>,
    pub symbol: Vec<Symbol>,
    pub used_symbols: Vec<Symbol>,
    pub parent_id: Option<String>,
    pub visibility: Visibility,
    pub signature: Option<Signature>,
    pub attributes: Vec<String>,
    pub comments: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub line_start: usize,
    pub line_end: usize,
}

impl Default for CodeBlock {
    fn default() -> Self {
        Self {
            block_id: String::new(),
            block_info: (BlockType::Atomic, NodeKind::Statement),
            parent_id: None,
            range: None,
            root: String::new(),
            scope_path: ScopePath {
                segments: Vec::new(),
            },
            visibility: Visibility::Private,
            text: String::new(),
            signature: None,
            attributes: Vec::new(),
            comments: None,
            symbol: Vec::new(),
            used_symbols: Vec::new(),
            lsp: None,
            embedding: None,
            line_start: 0,
            line_end: 0,
        }
    }
}

impl CodeBlock {
    pub fn new(scope_path: ScopePath, range: TSRange, block_info: BlockInfo) -> Self {
        let root = &scope_path.clone().segments[0].name;
        let line_start = range.start_point.row + 1;
        let line_end = range.end_point.row + 1;
        Self {
            block_id: scope_path.qualified(),
            range: Some(range.into()),
            scope_path,
            root: root.to_string(),
            parent_id: None,
            block_info,
            line_start,
            line_end,
            ..Default::default()
        }
    }

    pub fn from_comment(
        comment_node: Node,
        scope: &ScopePath,
        block_info: &BlockInfo,
        src: &[u8],
    ) -> CodeBlock {
        let mut comment = Self::new(scope.clone(), comment_node.range(), *block_info);
        comment.text = String::from(comment_node.utf8_text(src).unwrap());
        comment
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    EnumString,
    AsRefStr,
    Serialize,
    Deserialize,
    SurrealValue,
    VariantArray,
)]
pub enum SupportedLanguage {
    #[strum(serialize = "py")]
    Python,

    #[strum(serialize = "rs")]
    Rust,

    #[strum(serialize = "ts")]
    Typescript,

    #[strum(serialize = "tsx")]
    TSX,
}

#[derive(
    Clone,
    Copy,
    Hash,
    Debug,
    PartialEq,
    Eq,
    Display,
    AsRefStr,
    EnumString,
    Serialize,
    Deserialize,
    SurrealValue,
)]
pub enum NodeKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    Impl,
    Module,
    Const,
    Static,
    TypeAlias,
    Statement,
    Attribute,
    Comment,
    Import,
}

pub type BlockInfo = (BlockType, NodeKind);

pub static LANGUAGE_MAPPING: LazyLock<HashMap<SupportedLanguage, LangSyntax>> =
    LazyLock::new(|| {
        HashMap::from([
            (SupportedLanguage::Python, PYTHON.clone()),
            (SupportedLanguage::Rust, RUST.clone()),
            (SupportedLanguage::Typescript, TYPESCRIPT.clone()),
        ])
    });

pub static PARSER_MAPPING: LazyLock<HashMap<SupportedLanguage, Language>> = LazyLock::new(|| {
    HashMap::from([
        (
            SupportedLanguage::Python,
            tree_sitter_python::LANGUAGE.into(),
        ),
        (SupportedLanguage::Rust, tree_sitter_rust::LANGUAGE.into()),
        (
            SupportedLanguage::Typescript,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        ),
    ])
});

pub fn flatten_import_container(
    node: Node,
    src: &[u8],
    prefix: Vec<String>,
    out: &mut ImportBlock,
    syntax: &LangSyntax,
    _cursor: &mut TreeCursor,
) -> Option<ImportBlock> {
    let Some(block_info) = syntax.block_determine(node.kind()) else {
        return None;
    };
    let lang = &syntax.language;
    let (_, node_kind) = block_info;
    if node_kind != NodeKind::Import {
        return None;
    }

    match lang {
        SupportedLanguage::Rust => match node.kind() {
            "use_list" => {
                for child in node.named_children(&mut node.walk()) {
                    let _ = flatten_import_container(
                        child,
                        src,
                        prefix.clone(),
                        out,
                        syntax,
                        &mut node.walk(),
                    );
                }
                Some(out.clone())
            }
            "use_declaration" => {
                for child in node.children(&mut node.walk()) {
                    let kind = child.kind();
                    if let Some((ct, nk)) = syntax.block_determine(kind) {
                        if matches!(
                            (ct, nk),
                            (BlockType::Container, NodeKind::Import)
                                | (BlockType::Import, NodeKind::Import)
                        ) {
                            let _ = flatten_import_container(
                                child,
                                src,
                                prefix.clone(),
                                out,
                                syntax,
                                &mut node.walk(),
                            );
                        }
                    }
                }
                Some(out.clone())
            }
            "scoped_use_list" => {
                let name_node = node.child_by_field_name("path")?;

                let name = name_node.utf8_text(src).ok()?;
                let mut next_prefix = prefix;
                next_prefix.extend(name.split("::").map(String::from));

                for child in node.named_children(&mut node.walk()) {
                    // Skip the `path` child — its components were already folded into next_prefix
                    if child == name_node {
                        continue;
                    }
                    if let Some((ct, nk)) = syntax.block_determine(child.kind()) {
                        if matches!(
                            (ct, nk),
                            (BlockType::Container, NodeKind::Import)
                                | (BlockType::Import, NodeKind::Import)
                        ) {
                            let _ = flatten_import_container(
                                child,
                                src,
                                next_prefix.clone(),
                                out,
                                syntax,
                                &mut node.walk(),
                            );
                        }
                    }
                }
                Some(out.clone())
            }
            "scoped_identifier" => {
                let path_node = node.child_by_field_name("path")?;
                let path_text = path_node.utf8_text(src).ok()?;

                let name_node = node.child_by_field_name("name")?;
                let name_text = name_node.utf8_text(src).ok()?;

                let mut module = prefix;

                module.extend(path_text.split("::").map(String::from));

                out.imports
                    .entry(module.join("::"))
                    .or_default()
                    .push(name_text.to_string());

                None
            }
            "identifier" => {
                if let Ok(name) = node.utf8_text(src) {
                    out.imports
                        .entry(prefix.join("::"))
                        .or_default()
                        .push(name.to_string());
                }
                None
            }
            _ => None,
        },

        SupportedLanguage::Python => match node.kind() {
            "import_from_statement" => {
                let module_node = node.child_by_field_name("module_name")?;
                let module_text = module_node.utf8_text(src).ok()?;

                let module_parts: Vec<String> = module_text
                    .split(|c| c == '.' || c == ',')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();

                println!("module: {}", module_text.bold().yellow());

                for child in node.named_children(&mut node.walk()) {
                    match child.kind() {
                        "dotted_name" => {
                            if let Ok(symbol) = child.utf8_text(src) {
                                out.imports
                                    .entry(module_parts.join("::"))
                                    .or_default()
                                    .push(symbol.to_string());
                            }
                        }
                        "aliased_import" => {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                if let Ok(symbol) = name_node.utf8_text(src) {
                                    out.imports
                                        .entry(module_parts.join("::"))
                                        .or_default()
                                        .push(symbol.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                dbg!(&out.clone());
                Some(out.clone())
            }
            "import_statement" => {
                for child in node.named_children(&mut node.walk()) {
                    match child.kind() {
                        "dotted_name" => {
                            if let Ok(dotted) = child.utf8_text(src) {
                                let parts: Vec<&str> = dotted.split('.').collect();
                                let len = parts.len();
                                let module: Vec<String> =
                                    parts[..len - 1].iter().map(|s| s.to_string()).collect();

                                for p in parts.as_slice() {
                                    println!("part: {}", p.bold().yellow());
                                }
                                for m in module.as_slice() {
                                    println!("module: {}", m.bold().yellow());
                                }

                                out.imports
                                    .entry(module.join("::"))
                                    .or_default()
                                    .push(parts[len - 1].to_string());
                            }
                        }
                        "aliased_import" => {
                            if let Some(alias_node) = child.child_by_field_name("alias") {
                                if let Ok(alias) = alias_node.utf8_text(src) {
                                    out.imports
                                        .entry(prefix.join("::"))
                                        .or_default()
                                        .push(alias.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                dbg!(&out.clone());
                Some(out.clone())
            }
            _ => None,
        },

        SupportedLanguage::Typescript => match node.kind() {
            "import_statement" | "export_statement" => {
                let mut module_parts: Vec<String> = prefix;
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "string_fragment" {
                        module_parts = child
                            .utf8_text(src)
                            .ok()?
                            .split('/')
                            .map(String::from)
                            .collect();
                    } else if child.kind() != "string" {
                        let _ = flatten_import_container(
                            child,
                            src,
                            module_parts.clone(),
                            out,
                            syntax,
                            &mut node.walk(),
                        );
                    }
                }
                Some(out.clone())
            }
            "import_clause" | "named_imports" | "export_clause" => {
                for child in node.children(&mut node.walk()) {
                    let _ = flatten_import_container(
                        child,
                        src,
                        prefix.clone(),
                        out,
                        syntax,
                        &mut node.walk(),
                    );
                }
                Some(out.clone())
            }
            "import_specifier" | "export_specifier" => {
                let default_name = |n: Node| n.utf8_text(src).ok().map(String::from);
                let symbol = node
                    .child_by_field_name("alias")
                    .and_then(&default_name)
                    .or_else(|| node.child_by_field_name("name").and_then(&default_name))?;
                out.imports
                    .entry(prefix.join("::"))
                    .or_default()
                    .push(symbol);
                None
            }
            "namespace_import" => {
                for child in node.named_children(&mut node.walk()) {
                    if child.kind() == "identifier" {
                        if let Ok(name) = child.utf8_text(src) {
                            out.imports
                                .entry(prefix.join("::"))
                                .or_default()
                                .push(name.to_string());
                        }
                    }
                }
                None
            }
            "identifier" => {
                if let Ok(name) = node.utf8_text(src) {
                    out.imports
                        .entry(prefix.join("::"))
                        .or_default()
                        .push(name.to_string());
                }
                None
            }
            _ => None,
        },
        SupportedLanguage::TSX => None,
    }
}

static RUST_BLOCK_TYPES: LazyLock<HashMap<BlockInfo, Vec<String>>> = LazyLock::new(|| {
    HashMap::from([
        (
            (BlockType::Atomic, NodeKind::Function),
            vec!["function_item".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Struct),
            vec!["struct_item".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Enum),
            vec!["enum_item".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Const),
            vec!["const_item".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Static),
            vec!["static_item".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::TypeAlias),
            vec!["type_item".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Impl),
            vec!["impl_item".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Trait),
            vec!["trait_item".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Module),
            vec!["mod_item".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Import),
            vec![
                "scoped_use_list".to_string(),
                "use_list".to_string(),
                "use_declaration".to_string(),
            ],
        ),
        (
            (BlockType::Import, NodeKind::Import),
            vec!["identifier".to_string(), "scoped_identifier".to_string()],
        ),
        (
            (BlockType::Statement, NodeKind::Module),
            vec!["extern_crate_declaration".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Statement),
            vec![
                // declarations
                "let_declaration".to_string(),
                "expression_statement".to_string(),
                // control flow
                "while_expression".to_string(),
                "if_expression".to_string(),
                "match_expression".to_string(),
                "for_expression".to_string(),
                "return_expression".to_string(),
                "break_expression".to_string(),
                "continue_expression".to_string(),
                // expressions
                "macro_invocation".to_string(),
                "call_expression".to_string(),
                "field_expression".to_string(),
                "binary_expression".to_string(),
                "unary_expression".to_string(),
                "assignment_expression".to_string(),
                "compound_assignment_expr".to_string(),
                "closure_expression".to_string(),
                "index_expression".to_string(),
                "reference_expression".to_string(),
                "type_cast_expression".to_string(),
                "range_expression".to_string(),
                "try_expression".to_string(),
                // Literals
                "string_literal".to_string(),
                "string_content".to_string(),
                "integer_literal".to_string(),
                "char_literal".to_string(),
                "boolean_literal".to_string(),
                // Types
                "generic_type".to_string(),
                "type_identifier".to_string(),
                "type_arguments".to_string(),
                "scoped_type_identifier".to_string(),
                "tuple_type".to_string(),
                "reference_type".to_string(),
                "primitive_type".to_string(),
                // Patterns
                "tuple_pattern".to_string(),
                "match_pattern".to_string(),
                "or_pattern".to_string(),
                "tuple_struct_pattern".to_string(),
                "ref_pattern".to_string(),
                "match_block".to_string(),
                // Parameters/Arguments
                "parameters".to_string(),
                "parameter".to_string(),
                "closure_parameters".to_string(),
                "arguments".to_string(),
                // Blocks
                "block".to_string(),
                "declaration_list".to_string(),
                "field_declaration_list".to_string(),
                "field_initializer_list".to_string(),
                // Fields
                "field_declaration".to_string(),
                "field_initializer".to_string(),
                "shorthand_field_initializer".to_string(),
                // Other
                "token_tree".to_string(),
                "mutable_specifier".to_string(),
                "self".to_string(),
                "visibility_modifier".to_string(),
            ],
        ),
        (
            (BlockType::Attribute, NodeKind::Attribute),
            vec!["attribute_item".to_string(), "attribute".to_string()],
        ),
        (
            (BlockType::Comment, NodeKind::Comment),
            vec![
                "line_comment".to_string(),
                "block_comment".to_string(),
                "inner_doc_comment_marker".to_string(),
                "outer_doc_comment_marker".to_string(),
                "doc_comment".to_string(),
            ],
        ),
    ])
});

pub static RUST: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Rust,
    block_types: RUST_BLOCK_TYPES.clone(),
    name_field: "name".to_string(),
    body_field: String::from("body"),
    docstring_position: DocstringPosition::PrecedingSibling,
    attribute_types: String::from("attribute_item"),
    attribute_position: AttributePosition::PrecedingSibling,
    sig_fields: SigFields {
        parameters: Some("parameters".to_string()),
        return_type: Some("return_type".to_string()),
        type_parameters: Some("type_parameters".to_string()),
        bases: Some("trait".to_string()),
    },
    visibility_type: Some("visibility_modifier".to_string()),
    anon_resolvers: None,
    comment_types: vec!["line_comment".to_string(), "block_comment".to_string()],
});

static PYTHON_BLOCK_TYPES: LazyLock<HashMap<BlockInfo, Vec<String>>> = LazyLock::new(|| {
    HashMap::from([
        (
            (BlockType::Atomic, NodeKind::Function),
            vec![
                "function_definition".to_string(),
                "async_function_definition".to_string(),
            ],
        ),
        (
            (BlockType::Atomic, NodeKind::Statement),
            vec![
                // Literals
                "string".to_string(),
                "string_content".to_string(),
                "string_start".to_string(),
                "string_end".to_string(),
                "number".to_string(),
                "true".to_string(),
                "false".to_string(),
                "none".to_string(),
                // Expressions
                "identifier".to_string(),
                "dotted_name".to_string(),
                "call".to_string(),
                "call_expression".to_string(),
                "member_expression".to_string(),
                "binary_expression".to_string(),
                "unary_expression".to_string(),
                "assignment".to_string(),
                "augmented_assignment".to_string(),
                "parenthesized_expression".to_string(),
                "as_expression".to_string(),
                "ternary_expression".to_string(),
                "await_expression".to_string(),
                "lambda".to_string(),
                // Collections
                "list".to_string(),
                "tuple".to_string(),
                "set".to_string(),
                "dictionary".to_string(),
                "list_comprehension".to_string(),
                "dictionary_comprehension".to_string(),
                "subscript".to_string(),
                "slice".to_string(),
                "star_expression".to_string(),
                // Parameters
                "parameters".to_string(),
                "typed_parameter".to_string(),
                "typed_default_parameter".to_string(),
                "arguments".to_string(),
                "argument_list".to_string(),
                // Types
                "type".to_string(),
                "generic_type".to_string(),
                "type_parameter".to_string(),
                "binary_operator".to_string(),
                "boolean_operator".to_string(),
                "comparison_operator".to_string(),
                "not_operator".to_string(),
                "comparison".to_string(),
                // Other
                "attribute".to_string(),
                "formatted_string".to_string(),
                "escape_sequence".to_string(),
                "yield".to_string(),
                "pattern_list".to_string(),
                "as_pattern".to_string(),
                "class_pattern".to_string(),
                "conditional_expression".to_string(),
                // Control flow
                "if_statement".to_string(),
                "for_statement".to_string(),
                "while_statement".to_string(),
                "try_statement".to_string(),
                "with_statement".to_string(),
                "return_statement".to_string(),
                "break_statement".to_string(),
                "continue_statement".to_string(),
                "pass_statement".to_string(),
                "raise_statement".to_string(),
                "assert_statement".to_string(),
                "delete_statement".to_string(),
                "global_statement".to_string(),
                "nonlocal_statement".to_string(),
                "decorated_definition".to_string(),
                // Clauses
                "else_clause".to_string(),
                "except_clause".to_string(),
                "except_group_clause".to_string(),
                "finally_clause".to_string(),
                "for_in_clause".to_string(),
                "if_clause".to_string(),
                "with_item".to_string(),
            ],
        ),
        (
            (BlockType::Atomic, NodeKind::Module),
            vec!["expression_statement".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Import),
            vec!["import_statement".to_string(), "aliased_import".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Class),
            vec!["class_definition".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Import),
            vec!["import_from_statement".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Statement),
            vec!["block".to_string()],
        ),
        (
            (BlockType::Attribute, NodeKind::Attribute),
            vec!["decorator".to_string()],
        ),
        (
            (BlockType::Comment, NodeKind::Comment),
            vec!["comment".to_string()],
        ),
    ])
});

pub static PYTHON: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Python,
    block_types: PYTHON_BLOCK_TYPES.clone(),
    name_field: String::from("name"),
    body_field: String::from("body"),
    docstring_position: DocstringPosition::FirstBodyChild,
    anon_resolvers: None,
    attribute_types: String::from("decorator"),
    attribute_position: AttributePosition::ChildNode,
    sig_fields: SigFields {
        parameters: Some("parameters".to_string()),
        return_type: Some("return_type".to_string()),
        type_parameters: None,
        bases: Some("argument_list".to_string()),
    },
    visibility_type: None,
    comment_types: vec!["comment".to_string()],
});

static TYPESCRIPT_BLOCK_TYPES: LazyLock<HashMap<BlockInfo, Vec<String>>> = LazyLock::new(|| {
    HashMap::from([
        (
            (BlockType::Atomic, NodeKind::Function),
            vec![
                "function_declaration".to_string(),
                "arrow_function".to_string(),
            ],
        ),
        (
            (BlockType::Atomic, NodeKind::Method),
            vec!["method_definition".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Statement),
            vec![
                // Literals
                "string".to_string(),
                "string_fragment".to_string(),
                "string_content".to_string(),
                "string_start".to_string(),
                "string_end".to_string(),
                "number".to_string(),
                "true".to_string(),
                "false".to_string(),
                "null".to_string(),
                "undefined".to_string(),
                "identifier".to_string(),
                "property_identifier".to_string(),
                "call_expression".to_string(),
                "member_expression".to_string(),
                "new_expression".to_string(),
                "binary_expression".to_string(),
                "unary_expression".to_string(),
                "assignment_expression".to_string(),
                "augmented_assignment_expression".to_string(),
                "update_expression".to_string(),
                "ternary_expression".to_string(),
                "parenthesized_expression".to_string(),
                "as_expression".to_string(),
                "template_string".to_string(),
                "template_substitution".to_string(),
                "optional_chain".to_string(),
                "spread_element".to_string(),
                // Objects
                "object".to_string(),
                "pair".to_string(),
                "object_pattern".to_string(),
                "shorthand_property_identifier".to_string(),
                "shorthand_property_identifier_pattern".to_string(),
                // Collections
                "array".to_string(),
                "subscript_expression".to_string(),
                // Parameters
                "formal_parameters".to_string(),
                "required_parameter".to_string(),
                "property_signature".to_string(),
                "type_predicate".to_string(),
                "type_predicate_annotation".to_string(),
                "variable_declarator".to_string(),
                "lexical_declaration".to_string(),
            ],
        ),
        (
            (BlockType::Atomic, NodeKind::Interface),
            vec!["interface_declaration".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::TypeAlias),
            vec![
                "type_annotation".to_string(),
                "type_arguments".to_string(),
                "type_identifier".to_string(),
                "union_type".to_string(),
                "predefined_type".to_string(),
                "generic_type".to_string(),
                "array_type".to_string(),
                "literal_type".to_string(),
            ],
        ),
        (
            (BlockType::Container, NodeKind::Class),
            vec!["class_definition".to_string()],
        ),
        (
            (BlockType::Container, NodeKind::Interface),
            vec!["interface_body".to_string()],
        ),
        (
            (BlockType::Atomic, NodeKind::Import),
            vec![
                "import_statement".to_string(),
                "import_clause".to_string(),
                "import_specifier".to_string(),
                "named_imports".to_string(),
                "export_clause".to_string(),
                "export_specifier".to_string(),
                "export_statement".to_string(),
            ],
        ),
        (
            (BlockType::Container, NodeKind::Statement),
            vec!["statement_block".to_string()],
        ),
        (
            (BlockType::Attribute, NodeKind::Attribute),
            vec!["decorator".to_string()],
        ),
        (
            (BlockType::Comment, NodeKind::Comment),
            vec!["comment".to_string()],
        ),
    ])
});

pub static TYPESCRIPT: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Typescript,
    block_types: TYPESCRIPT_BLOCK_TYPES.clone(),
    name_field: String::from("name"),
    body_field: String::from("body"),
    anon_resolvers: Some(AnonResolver {
        node_type: "arrow_function".to_string(),
        parent_type: "variable_declarator".to_string(),
        name_field: "name".to_string(),
    }),
    attribute_types: String::from("decorator"),
    attribute_position: AttributePosition::ChildNode,
    sig_fields: SigFields {
        parameters: Some("parameters".to_string()),
        return_type: Some("return_type".to_string()),
        type_parameters: Some("type_parameters".to_string()),
        bases: Some("extends_clause".to_string()),
    },
    visibility_type: Some("accesibility_modifier".to_string()),
    docstring_position: DocstringPosition::FirstBodyChild,
    comment_types: vec!["comment".to_string()],
});

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LangSyntax {
    pub language: SupportedLanguage,
    pub block_types: HashMap<BlockInfo, Vec<String>>,
    pub name_field: String,
    pub body_field: String,
    pub comment_types: Vec<String>,
    pub docstring_position: DocstringPosition,
    pub anon_resolvers: Option<AnonResolver>,
    pub attribute_types: String,
    pub attribute_position: AttributePosition,
    pub sig_fields: SigFields,
    pub visibility_type: Option<String>,
}

impl LangSyntax {
    pub fn block_determine(&self, block_type: &str) -> Option<BlockInfo> {
        for (block_info, names) in &self.block_types {
            if names.contains(&block_type.to_string()) {
                return Some(*block_info);
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonResolver {
    pub node_type: String,
    pub parent_type: String,
    pub name_field: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributePosition {
    PrecedingSibling,
    ChildNode,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigFields {
    pub parameters: Option<String>,
    pub return_type: Option<String>,
    pub type_parameters: Option<String>,
    pub bases: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, SurrealValue)]
pub enum BlockType {
    Import,
    Atomic,
    Container,
    Comment,
    Statement,
    Attribute,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocstringPosition {
    FirstBodyChild,
    PrecedingSibling,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, SurrealValue)]
pub struct ScopeSegment {
    pub name: String,
    pub kind: BlockType,
    pub node_kind: NodeKind,
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct ScopePath {
    pub segments: Vec<ScopeSegment>,
}

impl ScopePath {
    pub fn from_module(name: &str) -> ScopePath {
        let segments = vec![ScopeSegment {
            name: String::from(name),
            kind: BlockType::Container,
            node_kind: NodeKind::Module,
        }];
        Self { segments }
    }

    pub fn qualified(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("::")
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct ImportGroup {
    pub comments: Option<String>,
    pub byte_range: Option<Range>,
    pub imports: Vec<ImportBlock>,
}

impl ImportGroup {
    pub fn new() -> ImportGroup {
        Self {
            comments: None,
            byte_range: None,
            imports: Vec::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct ImportBlock {
    pub base_module: String,
    pub imports: HashMap<String, Vec<String>>,
}

impl ImportBlock {
    pub fn new(base: &str) -> ImportBlock {
        Self {
            base_module: base.to_string(),
            imports: HashMap::new(),
        }
    }
}

impl Default for ImportBlock {
    fn default() -> Self {
        Self {
            base_module: String::new(),
            imports: HashMap::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct Signature {
    pub raw: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub type_params: Vec<String>,
    pub bases: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct Param {
    pub name: Option<String>,
    pub ty: Option<String>,
    pub default: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub enum Visibility {
    Public,
    Crate,
    Module,
    Private,
    Other(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, SurrealValue)]
pub struct Point {
    pub row: usize,
    pub column: usize,
}

impl From<TSPoint> for Point {
    fn from(p: TSPoint) -> Self {
        Self {
            row: p.row,
            column: p.column,
        }
    }
}

impl From<Point> for TSPoint {
    fn from(p: Point) -> Self {
        Self {
            row: p.row,
            column: p.column,
        }
    }
}

/// Serde-compatible wrapper around `tree_sitter::Range`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, SurrealValue)]
pub struct Range {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_point: Point,
    pub end_point: Point,
}

impl From<TSRange> for Range {
    fn from(r: TSRange) -> Self {
        Self {
            start_byte: r.start_byte,
            end_byte: r.end_byte,
            start_point: r.start_point.into(),
            end_point: r.end_point.into(),
        }
    }
}

impl From<Range> for TSRange {
    fn from(r: Range) -> Self {
        Self {
            start_byte: r.start_byte,
            end_byte: r.end_byte,
            start_point: r.start_point.into(),
            end_point: r.end_point.into(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct Symbol {
    pub name: String,
    pub kind: BlockType,
    pub byte_offset: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, SurrealValue)]
pub struct LspSymbolInfo {
    pub block_id: String,
    pub calls: Vec<Symbol>,
    pub called_by: Vec<Symbol>,
    pub references: Vec<Symbol>,
}

pub struct LspSymbolRequest {
    pub file: PathBuf,
    pub symbol_name: String,
    pub byte_offset: usize,
    pub block_id: String,
    pub scope_path: String,
}
