use std::str::FromStr;
use std::sync::LazyLock;
use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use strum::{AsRefStr, EnumString};
use tree_sitter::{Language, Node, Range, StreamingIterator, TreeCursor};

#[derive(Clone, Debug, PartialEq, Eq, Hash, EnumString, AsRefStr)]
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

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
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

fn flatten_import_container(
    node: Node,
    src: &[u8],
    prefix: Vec<String>,
    out: &mut ImportBlock,
    syntax: &LangSyntax,
    cursor: &mut TreeCursor,
) {
    let lang = &syntax.language;
    let block_info = syntax.block_types.get(node.kind()).unwrap();

    match lang {
        SupportedLanguage::Rust => {
            //recurse over scoped... types & use_list types -> push identifiers to import block
            match block_info {
                (BlockType::Container, NodeKind::Import) => {
                    if &node.kind() == &"use_list" {
                        for child in node.children(&mut node.walk()) {
                            flatten_import_container(
                                child,
                                src,
                                prefix.clone(),
                                out,
                                syntax,
                                cursor,
                            );
                        }
                    }
                    let name_node = node
                        .children(&mut node.walk())
                        .find(|c| c.kind() == "identifier")
                        .unwrap();

                    let name = name_node.utf8_text(src).unwrap();
                    let mut next_prefix = prefix.clone();
                    next_prefix.push(name.to_string());
                    for child in node.children(&mut cursor.clone()) {
                        flatten_import_container(
                            child,
                            src,
                            next_prefix.clone(),
                            out,
                            syntax,
                            cursor,
                        );
                    }
                }
                (BlockType::Import, NodeKind::Import) => {
                    let module = prefix.clone();
                    let symbol = node.utf8_text(src);
                    out.imports
                        .push(ModuleImport::new(module, String::from(symbol.unwrap())))
                }
                _ => {}
            }
        }
        SupportedLanguage::Python => {
            // split first from statement dotted name as path -> all other dotted names are symbols
            match block_info {
                (BlockType::Container, NodeKind::Import) => {
                    
                }
                (BlockType::Import, NodeKind::Import) => {}
                _ => {}
            }
        }
        SupportedLanguage::Typescript => {
            // reach string fragment -> set as path -> recurse named imports into import block
            match block_info {
                (BlockType::Container, NodeKind::Import) => {}
                (BlockType::Import, NodeKind::Import) => {}
                _ => {}
            }
        }
        _ => {}
    }
}

pub type BlockInfo = (BlockType, NodeKind);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LangSyntax {
    pub language: SupportedLanguage,
    pub block_types: HashMap<String, BlockInfo>,
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
    pub fn determine(&self, block_type: &str) -> Option<BlockInfo> {
        self.block_types
            .iter()
            .find(|(t, _)| *t == block_type)
            .map(|(_, class)| class.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnonResolver {
    pub node_type: &'static str,
    pub parent_type: &'static str,
    pub name_field: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttributePosition {
    PrecedingSibling,
    ChildNode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SigFields {
    pub parameters: Option<&'static str>,
    pub return_type: Option<&'static str>,
    pub type_parameters: Option<&'static str>,
    pub bases: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    Import,
    Atomic,
    Container,
    Comment,
    Statement,
    Attribute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocstringPosition {
    FirstBodyChild,
    PrecedingSibling,
}

pub static RUST: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Rust,
    block_types: {
        HashMap::from([
            (
                "function_item".to_string(),
                (BlockType::Atomic, NodeKind::Function),
            ),
            (
                "struct_item".to_string(),
                (BlockType::Atomic, NodeKind::Struct),
            ),
            ("enum_item".to_string(), (BlockType::Atomic, NodeKind::Enum)),
            (
                "type_item".to_string(),
                (BlockType::Atomic, NodeKind::TypeAlias),
            ),
            (
                "const_item".to_string(),
                (BlockType::Atomic, NodeKind::Const),
            ),
            (
                "static_item".to_string(),
                (BlockType::Atomic, NodeKind::Static),
            ),
            (
                "impl_item".to_string(),
                (BlockType::Container, NodeKind::Impl),
            ),
            (
                "trait_item".to_string(),
                (BlockType::Container, NodeKind::Trait),
            ),
            (
                "mod_item".to_string(),
                (BlockType::Container, NodeKind::Module),
            ),
            (
                "line_comment".to_string(),
                (BlockType::Comment, NodeKind::Comment),
            ),
            (
                "block_comment".to_string(),
                (BlockType::Comment, NodeKind::Comment),
            ),
            //call expression
            // expression_statement
            (
                "scoped_use_list".to_string(),
                (BlockType::Container, NodeKind::Import),
            ),
            (
                "scoped_identifier".to_string(),
                (BlockType::Container, NodeKind::Import),
            ),
            (
                "identifier".to_string(),
                (BlockType::Import, NodeKind::Import),
            ),
            (
                "use_list".to_string(),
                (BlockType::Container, NodeKind::Import),
            ),
            (
                "use_declaration".to_string(),
                (BlockType::Import, NodeKind::Import),
            ),
            (
                "extern_crate_declaration".to_string(),
                (BlockType::Statement, NodeKind::Module),
            ),
            (
                "attribute_item".to_string(),
                (BlockType::Attribute, NodeKind::Attribute),
            ),
        ])
    },
    name_field: "name".to_string(),
    body_field: String::from("body"),
    docstring_position: DocstringPosition::PrecedingSibling,
    attribute_types: String::from("attribute_item"),
    attribute_position: AttributePosition::PrecedingSibling,
    sig_fields: SigFields {
        parameters: Some("parameters"),
        return_type: Some("return_type"),
        type_parameters: Some("type_parameters"),
        bases: Some("trait"),
    },
    visibility_type: Some("visibility_modifier".to_string()),
    anon_resolvers: None, // TODO: determine correct
    comment_types: vec!["line_comment".to_string(), "block_comment".to_string()],
});

pub static PYTHON: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Python,
    block_types: HashMap::from([
        (
            "function_definition".to_string(),
            (BlockType::Atomic, NodeKind::Function),
        ),
        (
            "async_function_definition".to_string(),
            (BlockType::Atomic, NodeKind::Function),
        ),
        (
            "class_definition".to_string(),
            (BlockType::Container, NodeKind::Class),
        ),
        (
            "comment".to_string(),
            (BlockType::Comment, NodeKind::Comment),
        ),
        (
            "import_statement".to_string(),
            (BlockType::Import, NodeKind::Import),
        ),
        (
            "import_from_statement".to_string(),
            (BlockType::Container, NodeKind::Import),
        ),
        (
            "expression_statement".to_string(),
            (BlockType::Statement, NodeKind::Module),
        ),
        (
            "decorator".to_string(),
            (BlockType::Attribute, NodeKind::Attribute),
        ),
    ]),
    name_field: String::from("name"),
    body_field: String::from("body"),
    docstring_position: DocstringPosition::FirstBodyChild,
    anon_resolvers: None,

    attribute_types: String::from("decorator"),
    attribute_position: AttributePosition::ChildNode,
    sig_fields: SigFields {
        parameters: Some("parameters"),
        return_type: Some("return_type"),
        type_parameters: None,
        bases: Some("argument_list"),
    },
    visibility_type: None,

    comment_types: vec!["comment".to_string()],
});

pub static TYPESCRIPT: LazyLock<LangSyntax> = LazyLock::new(|| LangSyntax {
    language: SupportedLanguage::Typescript,
    block_types: HashMap::from([
        (
            "function_declaration".to_string(),
            (BlockType::Atomic, NodeKind::Function),
        ),
        (
            "arrow_function".to_string(),
            (BlockType::Atomic, NodeKind::Function),
        ),
        (
            "method_definition".to_string(),
            (BlockType::Atomic, NodeKind::Method),
        ),
        (
            "class_definition".to_string(),
            (BlockType::Container, NodeKind::Class),
        ),
        (
            "interface_definition".to_string(),
            (BlockType::Container, NodeKind::Interface),
        ),
        (
            "comment".to_string(),
            (BlockType::Comment, NodeKind::Comment),
        ),
        (
            "import_statement".to_string(),
            (BlockType::Import, NodeKind::Import),
        ),
        (
            "named_imports".to_string(),
            (BlockType::Import, NodeKind::Import),
        ),
        // export statements are imports / symbols with public visibility
        (
            "decorator".to_string(),
            (BlockType::Attribute, NodeKind::Attribute),
        ),
    ]),
    name_field: String::from("name"),
    body_field: String::from("body"),
    anon_resolvers: Some(AnonResolver {
        node_type: "arrow_function",
        parent_type: "variable_declarator",
        name_field: "name",
    }),

    attribute_types: String::from("decorator"),
    attribute_position: AttributePosition::ChildNode,
    sig_fields: SigFields {
        parameters: Some("parameters"),
        return_type: Some("return_type"),
        type_parameters: Some("type_parameters"),
        bases: Some("extends_clause"),
    },
    visibility_type: Some("accesibility_modifier".to_string()),
    docstring_position: DocstringPosition::FirstBodyChild, //TODO: determine correct
    comment_types: vec!["comment".to_string()],
});

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

// document
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

// chunk
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

// code document structs
// code location in file / explicit hierarchy
#[derive(Clone)]
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
        Self { segments: segments }
    }
}

#[derive(Clone)]
pub struct ScopeSegment {
    pub name: String,
    pub kind: BlockType,
    pub node_kind: NodeKind,
}

impl ScopePath {
    pub fn qualified(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("::")
    }
}

pub struct Signature {
    pub raw: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub type_params: Vec<String>,
    pub bases: Vec<String>,
}

pub struct Param {
    pub name: Option<String>,
    pub ty: Option<String>,
    pub default: Option<String>,
}

pub enum Visibility {
    Public,
    Crate,
    Module,
    Private,
    Other(String),
}

pub struct Symbol {
    pub name: String,
    pub kind: BlockType,
    pub byte_offset: usize,
}

pub struct CodeBlock {
    pub block_id: String,
    pub block_info: BlockInfo,
    pub range: Option<Range>,
    pub scope_path: ScopePath,
    pub root: String,
    pub text: String,
    pub lsp: Option<LspSymbolInfo>,

    pub symbol: Vec<Symbol>,
    pub parent_id: Option<String>,
    pub visibility: Visibility,

    pub signature: Option<Signature>,
    pub attributes: Vec<String>,
    pub comments: Option<String>,
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
            lsp: None,
        }
    }
}

impl CodeBlock {
    pub fn new(scope_path: ScopePath, range: Range, block_info: BlockInfo) -> Self {
        // gen block id from scope_path
        let root = &scope_path.clone().segments[0].name;

        Self {
            block_id: scope_path.qualified(),
            range: Some(range),
            scope_path: scope_path,
            root: root.to_string(),
            parent_id: None,
            block_info: block_info,
            ..Default::default()
        }
    }

    pub fn from_atomic() {
        //function
        //structs
        //enum
        //const
        //static
        //arrow function
        //method (ts)
    }

    pub fn from_import() {
        //import
    }

    pub fn from_statement(&self) {
        //import

        //expression
        //python module level docstrings
    }

    pub fn from_container(&self) {
        //class
        //impl
        //trait
        //mod
        //scope_use_list
    }

    pub fn from_comment(
        comment_node: Node,
        scope: &ScopePath,
        block_info: &BlockInfo,
        src: &[u8],
    ) -> CodeBlock {
        let mut comment = Self::new(scope.clone(), comment_node.range(), block_info.clone());
        comment.text = String::from(comment_node.utf8_text(src).unwrap());
        comment
    }

    pub fn from_attribute(&self) {
        //decorator
        //attribute item
    }
}

pub struct CodeSource {
    pub path: PathBuf,
    pub language: SupportedLanguage,
    pub module_coment: Option<String>,
    pub imports: Vec<ImportGroup>,
    pub blocks: Vec<CodeBlock>,
}

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

pub struct ImportBlock {
    pub base: String,
    pub imports: Vec<ModuleImport>,
}

pub struct ModuleImport {
    pub module_path: Vec<String>,
    pub symbol: String,
}

impl ModuleImport {
    pub fn new(module_path: Vec<String>, symbol: String) -> ModuleImport {
        Self {
            module_path: module_path,
            symbol: symbol,
        }
    }
}

pub struct LspSymbolInfo {
    pub block_id: String,
    pub calls: Vec<Symbol>,
    pub called_by: Vec<Symbol>,
    pub references: Vec<Symbol>,
    pub type_refs: Vec<Symbol>,
    pub overrides: Vec<Symbol>,
}

pub struct LspSymbolRequest {
    pub file: PathBuf,
    pub symbol_name: String,
    pub byte_offset: usize,
    pub block_id: String,
    pub scope_path: String,
}
