use std::{env, ffi::OsStr, io::Error, path::PathBuf, result::Result, str::FromStr};

use fosra::types::{
    AttributePosition, BlockInfo, BlockType, CodeBlock, ImportGroup, LANGUAGE_MAPPING, LangSyntax,
    PARSER_MAPPING, ScopePath, ScopeSegment, SupportedLanguage,
};
use tree_sitter::{Node, Parser, Tree};
use tree_sitter_rust;

struct CodeFile {
    //file
    source_text: String,
    syntax: &'static LangSyntax,
    path: PathBuf,
    tree: Tree,
    language: SupportedLanguage,

    // blocks
    blocks: Vec<CodeBlock>,
    imports: ImportGroup,

    //buffers
    scope_stack: ScopePath,
    comment_buffer: Vec<CodeBlock>,
    statement_buffer: Vec<String>,
    attribute_buffer: Vec<ScopeSegment>,

    last_byte: usize,
    next_id: usize,
}

struct CommentBuffer {
    items: Vec<CodeBlock>,
    pending_blank_line: bool,
}

impl Default for CommentBuffer {
    fn default() -> Self {
        Self {
            items: vec![],
            pending_blank_line: false,
        }
    }
}

impl CodeFile {
    pub fn new(path: &PathBuf) -> Result<CodeFile, std::io::Error> {
        let ext = &path
            .extension()
            .and_then(OsStr::to_str)
            .ok_or(Error::last_os_error())?;

        let module_name = &path.file_stem().and_then(OsStr::to_str).unwrap_or("");

        let lang = SupportedLanguage::from_str(ext).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid extension")
        })?;

        let source_code = std::fs::read_to_string(&path).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "File could not be read")
        })?;

        let syntax = LANGUAGE_MAPPING.get(&lang).unwrap();

        // create parser
        let mut parser = Parser::new();

        let _ = parser.set_language(PARSER_MAPPING.get(&lang).unwrap());

        //get tree and root
        let tree = parser.parse(&source_code, None).unwrap();
        let root = tree.root_node();

        // create and push module block to list
        let first_byte = root.range().start_byte;

        Ok(Self {
            source_text: source_code,
            syntax: syntax,
            tree: tree,
            path: path.clone(),
            language: lang,
            attribute_buffer: vec![],
            blocks: vec![],
            comment_buffer: vec![],
            imports: ImportGroup::new(),
            scope_stack: ScopePath::from_module(module_name),
            statement_buffer: vec![],
            last_byte: first_byte,
            next_id: 0,
        })
    }

    //parent passes module id
    fn _walk(&mut self, node: Node, parent: CodeBlock, src: &[u8]) {
        let mut cursor = self.tree.walk();
        for child in node.named_children(&mut cursor) {
            let child_kind = child.kind();
            let block_info = self.syntax.block_types.get(child_kind).unwrap();
            let (block_type, node_type) = block_info;

            // block type for category -> node kind for semantics
            match block_type {
                //comment - added to comment buffer
                BlockType::Comment => {
                    // rust  python  ts
                    self.comment_buffer.push(CodeBlock::from_comment(
                        child,
                        &self.scope_stack,
                        block_info,
                        src,
                    ));
                }

                // solo imports  | comments drained in
                BlockType::Import => {}

                BlockType::Statement => {
                    // list type imports as containers

                    // single import
                    ////  parse origin

                    // group import - container -> recurse w/ _walk -> added to import group
                    //// create import group  -> parse origin until exhausted -> push to buffer

                    // raw imports as statements
                }

                // decorator / derive like node
                BlockType::Attribute => {
                    if self.syntax.attribute_position == AttributePosition::PrecedingSibling {}
                }

                // self contained blocks
                BlockType::Atomic => {}

                // contain
                BlockType::Container => {}
            }
        }
    }

    // fn merge_ranges(&self, nodes: &[Node]) -> Range {
    //     let start = nodes.first().map(|n| n.start_byte()).unwrap_or(0);
    //     let end = nodes.last().map(|n| n.end_byte());
    // }

    // public parse method
    pub fn parse(&self, file_path: PathBuf) {
        // file details

        // init module node
    }
}

fn main() {
    let code_file = env::current_dir()
        .unwrap()
        .join("core")
        .join("examples")
        .join("ingest.rs");

    let code_text = code_file.to_str().unwrap();

    let mut rust_parser = tree_sitter::Parser::new();

    rust_parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Error loading Rust grammar");

    let mut tree = rust_parser.parse(code_text, None).unwrap();
}
