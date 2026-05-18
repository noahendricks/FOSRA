use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::LazyLock;
use std::{collections::HashMap, ffi::OsStr, path::PathBuf, str::FromStr};

use serde::{Deserialize, Serialize};
use strum::{AsRefStr, Display, EnumString};
use tree_sitter::{Language, Node, Parser, TreeCursor};
pub use tree_sitter::{Point as TSPoint, Range as TSRange, Tree as TSTree};

/// Serde-compatible wrapper around `tree_sitter::Point`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, EnumString, AsRefStr, Serialize, Deserialize)]
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
    Clone, Copy, Hash, Debug, PartialEq, Eq, Display, AsRefStr, EnumString, Serialize, Deserialize,
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
                    .entry(module)
                    .or_default()
                    .push(name_text.to_string());
                None
            }
            "identifier" => {
                if let Ok(name) = node.utf8_text(src) {
                    out.imports
                        .entry(prefix)
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
                let module_parts: Vec<String> = module_text.split('.').map(String::from).collect();

                for child in node.named_children(&mut node.walk()) {
                    match child.kind() {
                        "dotted_name" => {
                            if let Ok(symbol) = child.utf8_text(src) {
                                out.imports
                                    .entry(module_parts.clone())
                                    .or_default()
                                    .push(symbol.to_string());
                            }
                        }
                        "aliased_import" => {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                if let Ok(symbol) = name_node.utf8_text(src) {
                                    out.imports
                                        .entry(module_parts.clone())
                                        .or_default()
                                        .push(symbol.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
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
                                out.imports
                                    .entry(module)
                                    .or_default()
                                    .push(parts[len - 1].to_string());
                            }
                        }
                        "aliased_import" => {
                            if let Some(alias_node) = child.child_by_field_name("alias") {
                                if let Ok(alias) = alias_node.utf8_text(src) {
                                    out.imports
                                        .entry(prefix.clone())
                                        .or_default()
                                        .push(alias.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
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
                out.imports.entry(prefix).or_default().push(symbol);
                None
            }
            "namespace_import" => {
                for child in node.named_children(&mut node.walk()) {
                    if child.kind() == "identifier" {
                        if let Ok(name) = child.utf8_text(src) {
                            out.imports
                                .entry(prefix.clone())
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
                        .entry(prefix)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Focus {
    Conversation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: DocumentMetadata,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<Chunk>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted_keywords: Option<Vec<Keyword>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_by: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extraction_duration_ms: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keyword {
    pub text: String,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub byte_start: usize,
    pub byte_end: usize,
    pub token_count: Option<usize>,
    pub chunk_index: usize,
    pub total_chunks: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_page: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_page: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading_context: Option<HeadingContext>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadingContext {
    pub headings: Vec<HeadingLevel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadingLevel {
    pub level: u8,
    pub text: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ScopeSegment {
    pub name: String,
    pub kind: BlockType,
    pub node_kind: NodeKind,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Signature {
    pub raw: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub type_params: Vec<String>,
    pub bases: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Param {
    pub name: Option<String>,
    pub ty: Option<String>,
    pub default: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Visibility {
    Public,
    Crate,
    Module,
    Private,
    Other(String),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Symbol {
    pub name: String,
    pub kind: BlockType,
    pub byte_offset: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
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
    pub embedding: Option<f32>,
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

pub struct CodeSource {
    pub path: PathBuf,
    pub language: SupportedLanguage,
    pub content_hash: u64,
    pub module_coment: Option<String>,
    pub imports: Vec<ImportGroup>,
    pub blocks: Vec<CodeBlock>,
    pub inline_imports: HashMap<Vec<String>, Vec<String>>,
}
use tree_sitter::Node as TSNode;

struct CodeParser {
    path: PathBuf,
    language: SupportedLanguage,
    syntax: LangSyntax,
    scope_stack: ScopePath,
    blocks: Vec<CodeBlock>,
    imports: Vec<ImportGroup>,
    comment_buffer: Vec<CodeBlock>,
    statement_buffer: Vec<String>,
    attribute_buffer: Vec<String>,
}

impl CodeParser {
    fn new(
        path: PathBuf,
        language: SupportedLanguage,
        syntax: LangSyntax,
        _source_text: Vec<u8>,
        module_name: String,
    ) -> Self {
        Self {
            path,
            language,
            syntax,
            scope_stack: ScopePath::from_module(&module_name),
            blocks: Vec::new(),
            imports: Vec::new(),
            comment_buffer: Vec::new(),
            statement_buffer: Vec::new(),
            attribute_buffer: Vec::new(),
        }
    }

    fn drain_comments(&mut self) -> Option<String> {
        if self.comment_buffer.is_empty() {
            return None;
        }
        let text = self
            .comment_buffer
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.comment_buffer.clear();
        Some(text)
    }

    fn drain_attributes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.attribute_buffer)
    }
}

impl CodeParser {
    fn extract_name(&self, node: &TSNode, src: &[u8]) -> String {
        node.child_by_field_name(&self.syntax.name_field)
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or("")
            .to_string()
    }

    fn extract_declared_symbols(&self, node: &TSNode, src: &[u8]) -> Vec<Symbol> {
        // 1. Try name_field (covers function_item, class_definition, etc.)
        let name = self.extract_name(node, src);
        if !name.is_empty() {
            return vec![Symbol {
                name,
                kind: BlockType::Atomic,
                byte_offset: node.start_byte(),
            }];
        }
        // 2. Try pattern: (let_declaration, variable_declarator, parameter)
        if let Some(pattern) = node.child_by_field_name("pattern") {
            if let Some(ident) = first_identifier(pattern, src) {
                return vec![Symbol {
                    name: ident,
                    kind: BlockType::Atomic,
                    byte_offset: node.start_byte(),
                }];
            }
        }
        // 3. Try left: (Python assignment)
        if let Some(left) = node.child_by_field_name("left") {
            if let Some(ident) = first_identifier(left, src) {
                return vec![Symbol {
                    name: ident,
                    kind: BlockType::Atomic,
                    byte_offset: node.start_byte(),
                }];
            }
        }
        // 4. Try name: in named_imports / import_specifier (TS)
        if let Some(name_child) = node.child_by_field_name("name") {
            if let Ok(text) = name_child.utf8_text(src) {
                if !text.is_empty() {
                    return vec![Symbol {
                        name: text.to_string(),
                        kind: BlockType::Atomic,
                        byte_offset: node.start_byte(),
                    }];
                }
            }
        }
        // 5. self parameter
        if node.kind() == "self_parameter" || node.kind() == "self" {
            return vec![Symbol {
                name: "self".into(),
                kind: BlockType::Atomic,
                byte_offset: node.start_byte(),
            }];
        }
        // 6. Walk named children for name/pattern/left (catches lexical_declaration→variable_declarator,
        //    expression_statement→assignment, etc.)
        for i in 0..node.named_child_count() as u32 {
            if let Some(child) = node.named_child(i) {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(text) = name_node.utf8_text(src) {
                        if !text.is_empty() {
                            return vec![Symbol {
                                name: text.to_string(),
                                kind: BlockType::Atomic,
                                byte_offset: node.start_byte(),
                            }];
                        }
                    }
                }
                // Try pattern: on child (let_declaration in some contexts)
                if let Some(pattern) = child.child_by_field_name("pattern") {
                    if let Some(ident) = first_identifier(pattern, src) {
                        return vec![Symbol {
                            name: ident,
                            kind: BlockType::Atomic,
                            byte_offset: node.start_byte(),
                        }];
                    }
                }
                // Try left: on child (Python assignment inside expression_statement)
                if let Some(left) = child.child_by_field_name("left") {
                    if let Some(ident) = first_identifier(left, src) {
                        return vec![Symbol {
                            name: ident,
                            kind: BlockType::Atomic,
                            byte_offset: node.start_byte(),
                        }];
                    }
                }
            }
        }
        vec![]
    }

    fn extract_signature(&self, node: &TSNode, src: &[u8]) -> Option<Signature> {
        let sig = &self.syntax.sig_fields;
        let params: Vec<_> = sig
            .parameters
            .as_ref()
            .and_then(|f| node.child_by_field_name(f))
            .map(|p| {
                p.named_children(&mut p.walk())
                    .filter_map(|c| {
                        let name = c
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(src).ok().map(String::from));
                        let ty = c
                            .child_by_field_name("type")
                            .and_then(|n| n.utf8_text(src).ok().map(String::from));
                        let default = c
                            .child_by_field_name("default")
                            .and_then(|n| n.utf8_text(src).ok().map(String::from));
                        Some(Param { name, ty, default })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let return_type = sig
            .return_type
            .as_ref()
            .and_then(|f| node.child_by_field_name(f))
            .and_then(|n| n.utf8_text(src).ok().map(String::from));
        let type_params: Vec<String> = sig
            .type_parameters
            .as_ref()
            .and_then(|f| node.child_by_field_name(f))
            .map(|p| {
                p.named_children(&mut p.walk())
                    .filter_map(|c| c.utf8_text(src).ok().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let bases: Vec<String> = sig
            .bases
            .as_ref()
            .and_then(|f| node.child_by_field_name(f))
            .map(|b| {
                b.named_children(&mut b.walk())
                    .filter_map(|c| c.utf8_text(src).ok().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if params.is_empty() && return_type.is_none() && type_params.is_empty() && bases.is_empty()
        {
            return None;
        }
        let raw = {
            let mut s = String::new();
            if let Ok(text) = node.utf8_text(src) {
                if let Some(end) = text.find(|c| c == '{' || c == ';' || c == '\n') {
                    s.push_str(&text[..end].trim());
                } else {
                    s.push_str(text.trim());
                }
            }
            s
        };
        Some(Signature {
            raw,
            params,
            return_type,
            type_params,
            bases,
        })
    }

    fn extract_visibility(&self, node: &TSNode, src: &[u8]) -> Visibility {
        let Some(vtype) = &self.syntax.visibility_type else {
            return Visibility::Private;
        };
        if let Some(vnode) = node.child_by_field_name(vtype) {
            if let Ok(text) = vnode.utf8_text(src) {
                return match text.trim() {
                    "pub" | "public" => Visibility::Public,
                    "pub(crate)" => Visibility::Crate,
                    "pub(super)" => Visibility::Module,
                    other => Visibility::Other(other.to_string()),
                };
            }
        }
        Visibility::Private
    }

    fn build_atomic_text(&mut self, node: &TSNode, src: &[u8]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for c in &self.comment_buffer {
            if !c.text.is_empty() {
                parts.push(c.text.clone());
            }
        }
        for a in &self.attribute_buffer {
            if !a.is_empty() {
                parts.push(a.clone());
            }
        }
        for s in &self.statement_buffer {
            if !s.is_empty() {
                parts.push(s.clone());
            }
        }
        if let Ok(text) = node.utf8_text(src) {
            parts.push(text.to_string());
        }
        parts.join("\n")
    }

    fn _walk(&mut self, node: TSNode, src: &[u8]) {
        let children: Vec<TSNode> = (0..node.named_child_count())
            .filter_map(|i| node.named_child(i as u32))
            .collect();
        for child in children {
            self.handle_child(child, src);
        }
    }

    fn handle_child(&mut self, child: TSNode, src: &[u8]) {
        let child_kind = child.kind();
        let Some(block_info) = self.syntax.block_determine(child_kind) else {
            return;
        };
        let (block_type, node_kind) = block_info;
        match (block_type, node_kind) {
            (_, NodeKind::Import) => {
                let mut import_block = ImportBlock::default();
                if let Some(block) = flatten_import_container(
                    child,
                    src,
                    vec![],
                    &mut import_block,
                    &self.syntax,
                    &mut child.walk(),
                ) {
                    let comments = self.drain_comments();
                    let byte_range = Some(child.range().into());
                    self.imports.push(ImportGroup {
                        comments,
                        byte_range,
                        imports: vec![block],
                    });
                }
            }
            (BlockType::Comment, _) => {
                self.comment_buffer.push(CodeBlock::from_comment(
                    child,
                    &self.scope_stack,
                    &block_info,
                    src,
                ));
            }
            (BlockType::Attribute, _) => {
                if let Ok(attr_text) = child.utf8_text(src) {
                    self.attribute_buffer.push(attr_text.to_string());
                }
            }
            (BlockType::Statement, _) => {
                if let Ok(stmt_text) = child.utf8_text(src) {
                    self.statement_buffer.push(stmt_text.to_string());
                }
            }
            (BlockType::Atomic, _) => {
                let mut block = CodeBlock::new(self.scope_stack.clone(), child.range(), block_info);
                block.text = self.build_atomic_text(&child, src);
                block.comments = self.drain_comments();
                block.attributes = self.drain_attributes();
                block.symbol = self.extract_declared_symbols(&child, src);
                self.blocks.push(block);
                if let Some(body_node) = child.child_by_field_name(&self.syntax.body_field) {
                    for i in 0..body_node.named_child_count() as u32 {
                        if let Some(inner) = body_node.named_child(i) {
                            self.handle_child(inner, src);
                        }
                    }
                }
            }
            (BlockType::Container, _) => {
                let saved_scope = self.scope_stack.clone();
                let name = self.extract_name(&child, src);
                self.scope_stack.segments.push(ScopeSegment {
                    name: name.clone(),
                    kind: block_type,
                    node_kind,
                });
                if let Some(body_node) = child.child_by_field_name(&self.syntax.body_field) {
                    self._walk(body_node, src);
                }
                let mut container = CodeBlock::new(saved_scope.clone(), child.range(), block_info);
                container.symbol = vec![Symbol {
                    name,
                    kind: block_type,
                    byte_offset: child.start_byte(),
                }];
                container.comments = self.drain_comments();
                container.attributes = self.drain_attributes();
                container.signature = self.extract_signature(&child, src);
                container.visibility = self.extract_visibility(&child, src);
                self.scope_stack = saved_scope;
                self.blocks.push(container);
            }
            _ => {}
        }
    }
}

/// extract the first identifier child from a node subtree.
fn first_identifier(node: TSNode, src: &[u8]) -> Option<String> {
    if node.kind() == "identifier"
        || node.kind() == "field_identifier"
        || node.kind() == "property_identifier"
    {
        return node.utf8_text(src).ok().map(String::from);
    }
    for i in 0..node.named_child_count() as u32 {
        if let Some(child) = node.named_child(i) {
            if let Some(result) = first_identifier(child, src) {
                return Some(result);
            }
        }
    }
    None
}

fn collect_used_symbols(
    blocks: &[CodeBlock],
    root: TSNode,
    src: &[u8],
) -> Vec<(String, Vec<Symbol>)> {
    let mut all_identifiers: Vec<Symbol> = Vec::new();
    collect_identifiers_used(root, src, &mut all_identifiers);

    // deduplicate by name + byte_offset
    all_identifiers.sort_by(|a, b| a.name.cmp(&b.name).then(a.byte_offset.cmp(&b.byte_offset)));
    all_identifiers.dedup_by(|a, b| a.name == b.name && a.byte_offset == b.byte_offset);

    // Group by containing block (by byte range)
    let mut result: Vec<(String, Vec<Symbol>)> = Vec::new();
    for block in blocks {
        let block_used: Vec<Symbol> = all_identifiers
            .iter()
            .filter(|s| {
                block.range.as_ref().map_or(false, |r| {
                    s.byte_offset >= r.start_byte && s.byte_offset < r.end_byte
                })
            })
            .cloned()
            .collect();
        if !block_used.is_empty() {
            result.push((block.block_id.clone(), block_used));
        }
    }
    result
}

fn collect_identifiers_used(node: TSNode, src: &[u8], out: &mut Vec<Symbol>) {
    match node.kind() {
        // declaration nodes: walk body/reference fields, skip name/pattern/parameters
        "function_item" | "function_definition" | "method_definition" | "function_declaration" => {
            // Walk body (where references live)
            if let Some(body) = node.child_by_field_name("body") {
                walk_all_children(body, src, out);
            }
            // Walk return_type (may contain type identifiers)
            if let Some(rt) = node.child_by_field_name("return_type") {
                walk_all_children(rt, src, out);
            }
            return;
        }
        // Struct/class/etc: declaration-only, skip entirely
        "struct_item"
        | "enum_item"
        | "trait_item"
        | "type_item"
        | "const_item"
        | "static_item"
        | "class_declaration"
        | "interface_declaration"
        | "type_alias_declaration" => {
            return;
        }
        // let/const: skip pattern (declaration), walk value (reference)
        "let_declaration" | "lexical_declaration" | "variable_declarator" => {
            if let Some(value) = node.child_by_field_name("value") {
                walk_all_children(value, src, out);
            }
            return;
        }
        // assignment: skip left (declaration), walk right (reference)
        "assignment" => {
            if let Some(right) = node.child_by_field_name("right") {
                walk_all_children(right, src, out);
            }
            return;
        }
        // parameter: skip name, walk type annotation
        "parameter" | "required_parameter" => {
            if let Some(ty) = node.child_by_field_name("type") {
                walk_all_children(ty, src, out);
            }
            return;
        }
        "identifier" | "field_identifier" | "property_identifier" | "type_identifier" => {
            if let Ok(name) = node.utf8_text(src) {
                if !name.is_empty() {
                    out.push(Symbol {
                        name: name.to_string(),
                        kind: BlockType::Atomic,
                        byte_offset: node.start_byte(),
                    });
                }
            }
        }
        _ => {}
    }
    // recurse into all named children
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_identifiers_used(cursor.node(), src, out);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

/// walk all named children of the given node (no recursion into the node itself).
fn walk_all_children(node: TSNode, src: &[u8], out: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_identifiers_used(cursor.node(), src, out);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}
/// Collect inline path-qualified imports (crate/module references via `::` syntax).
/// Returns a map from module path to list of symbols used from that module,
/// e.g. `tree_sitter_rust::LANGUAGE` → `{["tree_sitter_rust"]: ["LANGUAGE"]}`.
/// Skips scoped paths where the head identifier is already a locally-imported symbol,
/// is a self-imported submodule (e.g. `io` from `use std::io::{self}`),
/// or for Rust/TypeScript skips uppercase names (type method calls like `String::from`).
fn collect_inline_imports(
    root: TSNode,
    src: &[u8],
    imported_symbols: &[String],
    imported_modules: &[String],
    language: SupportedLanguage,
) -> HashMap<Vec<String>, Vec<String>> {
    let mut result: Vec<(Vec<String>, String)> = Vec::new();

    // Reconstruct full scoped path from a node chain.
    // `A::B::C::D` → module = ["A", "B", "C"], symbol = "D"
    fn scoped_path_parts(node: TSNode, src: &[u8]) -> (Vec<String>, String) {
        let path = node.child_by_field_name("path");
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or("")
            .to_string();
        match path {
            Some(p) if p.kind() == "scoped_identifier" || p.kind() == "scoped_type_identifier" => {
                let (mut module, inner_symbol) = scoped_path_parts(p, src);
                module.push(inner_symbol);
                (module, name)
            }
            Some(p) => {
                let text = p.utf8_text(src).unwrap_or("");
                let parts: Vec<String> = text.split("::").map(String::from).collect();
                (parts, name)
            }
            None => (vec![], name),
        }
    }

    fn walk(node: TSNode, src: &[u8], out: &mut Vec<(Vec<String>, String)>) {
        let kind = node.kind();
        // Skip import-related subtrees
        if matches!(
            kind,
            "use_declaration"
                | "use_list"
                | "scoped_use_list"
                | "extern_crate_declaration"
                | "import_statement"
                | "export_statement"
                | "import_clause"
                | "named_imports"
                | "export_clause"
                | "import_from_statement"
        ) {
            return;
        }
        if kind == "scoped_identifier" || kind == "scoped_type_identifier" {
            // Only process the outermost scoped_identifier in each chain
            // (parent is NOT also scoped_identifier/scoped_type_identifier)
            let parent_is_scoped = node.parent().map_or(false, |p| {
                let pk = p.kind();
                pk == "scoped_identifier" || pk == "scoped_type_identifier"
            });
            if !parent_is_scoped {
                let (module, symbol) = scoped_path_parts(node, src);
                out.push((module, symbol));
            }
            return; // don't recurse into children
        }
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                walk(cursor.node(), src, out);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }

    walk(root, src, &mut result);

    // Apply filters
    let skip_module_head = |head: &str| -> bool {
        if imported_symbols.contains(&head.to_string()) {
            return true;
        }
        if imported_modules.contains(&head.to_string()) {
            return true;
        }
        // For Rust/TS, skip uppercase names (type method calls)
        match language {
            SupportedLanguage::Rust | SupportedLanguage::Typescript | SupportedLanguage::TSX => {
                if head.starts_with(|c: char| c.is_uppercase()) {
                    return true;
                }
            }
            SupportedLanguage::Python => {}
        }
        false
    };
    result.retain(|(module, _)| {
        module.first().map_or(false, |head| !skip_module_head(head))
    });

    // Group by module path: accumulate all symbols per module
    let mut map: HashMap<Vec<String>, Vec<String>> = HashMap::new();
    for (module, symbol) in result {
        map.entry(module).or_default().push(symbol);
    }
    // Deduplicate symbols per module
    for symbols in map.values_mut() {
        symbols.sort();
        symbols.dedup();
    }
    map
}

// ── CodeSource public API ─────────────────────────────────────────
impl CodeSource {
    pub fn parse(path: PathBuf) -> Result<CodeSource, String> {
        let source_text =
            std::fs::read_to_string(&path).map_err(|e| format!("Error reading file: {}", e))?;

        let ext = path
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "No file extension".to_string())?;

        let lang = SupportedLanguage::from_str(ext)
            .map_err(|_| format!("Unsupported language '{}'", ext))?;
        let lang_for_ii = lang.clone();
        let module_name = path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();
        let syntax = LANGUAGE_MAPPING.get(&lang).unwrap().clone();
        let mut parser = Parser::new();
        let _ = parser.set_language(PARSER_MAPPING.get(&lang).unwrap());
        let tree = parser
            .parse(&source_text, None)
            .ok_or("Parse returned None".to_string())?;
        let root = tree.root_node();
        let src = source_text.as_bytes();
        let mut code_parser = CodeParser::new(path.clone(), lang, syntax, src.to_vec(), module_name);
        code_parser._walk(root, src);

        // post-walk: collect used symbols (matched by byte range, not block_id)
        let used = collect_used_symbols(&code_parser.blocks, root, src);

        for block in &mut code_parser.blocks {
            if let Some(ref block_range) = block.range {
                // find the matching entry from collect_used_symbols by range
                block.used_symbols = used
                    .iter()
                    .find(|(_, syms)| {
                        syms.first().map_or(false, |s| {
                            s.byte_offset >= block_range.start_byte
                                && s.byte_offset <= block_range.end_byte
                        })
                    })
                    .map(|(_, syms)| syms.clone())
                    .unwrap_or_default();
            }
            // Deduplicate by name (different byte offsets may refer to the same symbol)
            block.used_symbols.sort_by(|a, b| a.name.cmp(&b.name));
            block.used_symbols.dedup_by(|a, b| a.name == b.name);
        }
        let imported_symbols = code_parser.imports.iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| {
                let base = &b.base_module;
                b.imports.values()
                    .flat_map(|v| v.iter().cloned())
                    .map(move |name| if base.is_empty() { name } else { format!("{base}::{name}") })
            })
            .collect::<Vec<String>>();
        let imported_modules: Vec<String> = code_parser.imports.iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| b.imports.keys().filter_map(|k| k.last().cloned()))
            .collect();
        let inline_imports = collect_inline_imports(root, src, &imported_symbols, &imported_modules, lang_for_ii);
        // Filter: used_symbols should only reference symbols known to this file
        // (declared, imported, or referenced via inline :: paths).
        let known: Vec<String> = {
            let mut k: Vec<String> = code_parser.blocks.iter()
                .flat_map(|b| b.symbol.iter().map(|s| s.name.clone()))
                .collect();
            for syms in inline_imports.values() {
                k.extend(syms.iter().cloned());
            }
            k.extend(code_parser.imports.iter()
                .flat_map(|g| g.imports.iter())
                .flat_map(|b| {
                    let base = &b.base_module;
                    b.imports.values()
                        .flat_map(|v| v.iter().cloned())
                        .map(move |name| if base.is_empty() { name } else { format!("{base}::{name}") })
                        .collect::<Vec<_>>()
                }));
            k.sort();
            k.dedup();
            k
        };
        for block in &mut code_parser.blocks {
            block.used_symbols.retain(|s| known.contains(&s.name));
        }
        let mut hasher = DefaultHasher::new();
        source_text.hash(&mut hasher);
        let content_hash = hasher.finish();
        Ok(CodeSource {
            path: code_parser.path,
            language: code_parser.language,
            content_hash,
            inline_imports,
            module_coment: None,
            imports: code_parser.imports,
            blocks: code_parser.blocks,
        })
    }

    pub fn declared_symbols(&self) -> Vec<Symbol> {
        let mut symbols: Vec<Symbol> = self.blocks.iter().flat_map(|b| b.symbol.clone()).collect();
        symbols.sort_by(|a, b| a.name.cmp(&b.name).then(a.byte_offset.cmp(&b.byte_offset)));
        symbols.dedup_by(|a, b| a.name == b.name);
        symbols
    }

    pub fn used_symbols(&self) -> Vec<Symbol> {
        let mut symbols: Vec<Symbol> = self
            .blocks
            .iter()
            .flat_map(|b| b.used_symbols.clone())
            .collect();
        symbols.sort_by(|a, b| a.name.cmp(&b.name).then(a.byte_offset.cmp(&b.byte_offset)));
        symbols.dedup_by(|a, b| a.name == b.name);
        symbols
    }

    pub fn imported_symbols(&self) -> Vec<String> {
        self.imports
            .iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| {
                let base = &b.base_module;
                b.imports
                    .values()
                    .flat_map(|v| v.iter().cloned())
                    .map(move |name| {
                        if base.is_empty() {
                            name
                        } else {
                            format!("{base}::{name}")
                        }
                    })
            })
            .collect()
    }
    /// Combined view: all crates/modules referenced by this source,
    /// both from explicit `use` statements and inline `::` path references.
    pub fn all_dependencies(&self) -> Vec<String> {
        let mut deps: Vec<String> = self.imported_symbols();
        for (module, symbols) in &self.inline_imports {
            let module_str = module.join("::");
            for sym in symbols {
                if module_str.is_empty() {
                    deps.push(sym.clone());
                } else {
                    deps.push(format!("{}::{}", module_str, sym));
                }
            }
        }
        deps.sort();
        deps.dedup();
        deps
    }
    /// Parse source text directly (for testing, or when source is already in memory).
    /// `path` is used for extension detection and module naming.
    pub fn parse_text(path: PathBuf, source_text: &str) -> Result<CodeSource, String> {
        let ext = path
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "No file extension".to_string())?;
        let lang = SupportedLanguage::from_str(ext)
            .map_err(|_| format!("Unsupported language '{}'", ext))?;
        let lang_for_ii = lang.clone();
        let module_name = path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();
        let syntax = LANGUAGE_MAPPING.get(&lang).unwrap().clone();
        let mut parser = Parser::new();
        let _ = parser.set_language(PARSER_MAPPING.get(&lang).unwrap());
        let tree = parser
            .parse(source_text, None)
            .ok_or("Parse returned None".to_string())?;
        let root = tree.root_node();
        let src = source_text.as_bytes();
        let mut code_parser = CodeParser::new(path.clone(), lang, syntax, src.to_vec(), module_name);
        code_parser._walk(root, src);
        let used = collect_used_symbols(&code_parser.blocks, root, src);
        for block in &mut code_parser.blocks {
            if let Some(ref block_range) = block.range {
                block.used_symbols = used
                    .iter()
                    .find(|(_, syms)| {
                        syms.first().map_or(false, |s| {
                            s.byte_offset >= block_range.start_byte
                                && s.byte_offset <= block_range.end_byte
                        })
                    })
                    .map(|(_, syms)| syms.clone())
                    .unwrap_or_default();
            }
            block.used_symbols.sort_by(|a, b| a.name.cmp(&b.name));
            block.used_symbols.dedup_by(|a, b| a.name == b.name);
        }
        let imported_symbols = code_parser.imports.iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| {
                let base = &b.base_module;
                b.imports.values()
                    .flat_map(|v| v.iter().cloned())
                    .map(move |name| if base.is_empty() { name } else { format!("{base}::{name}") })
            })
            .collect::<Vec<String>>();
        let imported_modules: Vec<String> = code_parser.imports.iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| b.imports.keys().filter_map(|k| k.last().cloned()))
            .collect();
        let inline_imports = collect_inline_imports(root, src, &imported_symbols, &imported_modules, lang_for_ii);
        let mut hasher = DefaultHasher::new();
        source_text.hash(&mut hasher);
        let content_hash = hasher.finish();
        Ok(CodeSource {
            path: code_parser.path,
            language: code_parser.language,
            content_hash,
            inline_imports,
            module_coment: None,
            imports: code_parser.imports,
            blocks: code_parser.blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

#[test]
    fn test_rust_declared_symbols() {
        let source = r#"
fn foo() {}
fn bar() {
    let x = foo();
}
"#;
        let cs = CodeSource::parse_text(PathBuf::from("test.rs"), source).unwrap();

        let declared = cs.declared_symbols();
        let declared: Vec<&str> = declared.iter().map(|s| s.name.as_str()).collect();

        assert!(declared.contains(&"foo"), "foo should be declared");
        assert!(declared.contains(&"bar"), "bar should be declared");
        assert!(declared.contains(&"x"), "x should be declared");
    }

    #[test]
    fn test_rust_used_symbols() {
        let source = r#"
fn bar() {
    let x = foo();
}
"#;
        let cs = CodeSource::parse_text(PathBuf::from("test.rs"), source).unwrap();
        let used = cs.used_symbols();
        let used: Vec<&str> = used.iter().map(|s| s.name.as_str()).collect();

        assert!(used.contains(&"foo"), "foo should be used");
    }
    #[test]
    fn test_rust_let_declaration() {
        let source = r#"
fn test() {
    let x = 42;
}
"#;
        let cs = CodeSource::parse_text(PathBuf::from("test.rs"), source).unwrap();
        // The let_declaration block should have "x" as declared symbol
        let let_blocks: Vec<&CodeBlock> = cs
            .blocks
            .iter()
            .filter(|b| b.symbol.iter().any(|s| s.name == "x"))
            .collect();

        assert_eq!(let_blocks.len(), 1, "exactly one block should declare x");
        assert_eq!(let_blocks[0].line_start, 3, "let x = 42 starts on line 3");
    }

    #[test]
    fn test_rust_used_in_function_body() {
        let source = r#"
fn caller() {
    callee();
    other_func();
}
"#;
        let cs = CodeSource::parse_text(PathBuf::from("test.rs"), source).unwrap();

        let caller_block = cs
            .blocks
            .iter()
            .find(|b| b.symbol.iter().any(|s| s.name == "caller"))
            .expect("caller block should exist");

        let used_names: Vec<&str> = caller_block
            .used_symbols
            .iter()
            .map(|s| s.name.as_str())
            .collect();

        assert!(
            used_names.contains(&"callee"),
            "callee should be used in caller"
        );
        assert!(
            used_names.contains(&"other_func"),
            "other_func should be used in caller"
        );
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ImportBlock {
    pub base_module: String,
    pub imports: HashMap<Vec<String>, Vec<String>>,
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

#[derive(Clone, Serialize, Deserialize, Debug)]
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
