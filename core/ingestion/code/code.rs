use crate::languages::{
    BlockInfo, BlockType, CodeBlock, ImportBlock, ImportGroup, LANGUAGE_MAPPING, LangSyntax,
    NodeKind, PARSER_MAPPING, Param, Range, ScopePath, ScopeSegment, Signature, SupportedLanguage,
    Symbol, Visibility, flatten_import_container,
};

use crate::processing::embedding::EmbeddingEngine;
use crate::{DocumentMetadata, print_node_tree};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Read;
use std::sync::LazyLock;
use std::{ffi::OsStr, path::PathBuf, str::FromStr};

use serde::{Deserialize, Serialize};
use surrealdb::types::SurrealValue;
use tokio::fs;
use tree_sitter::{Node, Parser};
pub use tree_sitter::{Point as TSPoint, Range as TSRange, Tree as TSTree};

#[derive(Debug, Clone, SurrealValue)]
pub struct CodeSource {
    pub file_path: String,
    pub language: Option<SupportedLanguage>,
    pub content_hash: Option<i64>,
    pub module_comments: Option<String>,
    pub imports: Vec<ImportGroup>,
    pub inline_imports: Option<HashMap<String, Vec<String>>>,
    pub blocks: Vec<CodeBlock>,
    pub embedding: Vec<f32>,
    pub metadata: Option<DocumentMetadata>,
}
use tree_sitter::Node as TSNode;

pub struct CodeParser {
    path: String,
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
        path: String,
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

    fn extract_declared_symbols(&self, node: &TSNode, src: &[u8]) -> Option<Vec<Symbol>> {
        // 1. Try name_field (covers function_item, class_definition, etc.)
        let name = self.extract_name(node, src);

        if !node.parent().is_some() {
            // pass
        } else if node.parent().unwrap().kind() != "module" {
            println!("parent not module: {}", node.parent().unwrap().kind());
            return None;
        }

        if !name.is_empty() {
            return Some(vec![Symbol {
                name,
                kind: BlockType::Atomic,
                byte_offset: node.start_byte(),
            }]);
        }
        // 2. Try pattern: (let_declaration, variable_declarator, parameter)
        if let Some(pattern) = node.child_by_field_name("pattern") {
            if let Some(ident) = first_identifier(pattern, src) {
                return Some(vec![Symbol {
                    name: ident,
                    kind: BlockType::Atomic,
                    byte_offset: node.start_byte(),
                }]);
            }
        }
        // 3. Try left: (Python assignment)
        if let Some(left) = node.child_by_field_name("left") {
            if let Some(ident) = first_identifier(left, src) {
                return Some(vec![Symbol {
                    name: ident,
                    kind: BlockType::Atomic,
                    byte_offset: node.start_byte(),
                }]);
            }
        }
        // 4. Try name: in named_imports / import_specifier (TS)
        if let Some(name_child) = node.child_by_field_name("name") {
            if let Ok(text) = name_child.utf8_text(src) {
                if !text.is_empty() {
                    return Some(vec![Symbol {
                        name: text.to_string(),
                        kind: BlockType::Atomic,
                        byte_offset: node.start_byte(),
                    }]);
                }
            }
        }
        // 5. self parameter
        if node.kind() == "self_parameter" || node.kind() == "self" {
            return Some(vec![Symbol {
                name: "self".into(),
                kind: BlockType::Atomic,
                byte_offset: node.start_byte(),
            }]);
        }

        // 6. Walk named children for name/pattern/left (catches lexical_declaration→variable_declarator,
        //    expression_statement→assignment, etc.)
        for i in 0..node.named_child_count() as u32 {
            if let Some(child) = node.named_child(i) {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(text) = name_node.utf8_text(src) {
                        if !text.is_empty() {
                            return Some(vec![Symbol {
                                name: text.to_string(),
                                kind: BlockType::Atomic,
                                byte_offset: node.start_byte(),
                            }]);
                        }
                    }
                }
                // Try pattern: on child (let_declaration in some contexts)
                if let Some(pattern) = child.child_by_field_name("pattern") {
                    if let Some(ident) = first_identifier(pattern, src) {
                        return Some(vec![Symbol {
                            name: ident,
                            kind: BlockType::Atomic,
                            byte_offset: node.start_byte(),
                        }]);
                    }
                }
                // Try left: on child (Python assignment inside expression_statement)
                if let Some(left) = child.child_by_field_name("left") {
                    if let Some(ident) = first_identifier(left, src) {
                        return Some(vec![Symbol {
                            name: ident,
                            kind: BlockType::Atomic,
                            byte_offset: node.start_byte(),
                        }]);
                    }
                }
            }
        }
        Some(vec![])
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

                let mut buffer = String::new();
                let _ = src.clone().read_to_string(&mut buffer).unwrap();

                print_node_tree!(child, buffer.as_str());

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

                block.symbol = self
                    .extract_declared_symbols(&child, src)
                    .unwrap_or_default();

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
fn collect_inline_imports(
    root: TSNode,
    src: &[u8],
    imported_symbols: &[String],
    imported_modules: &[String],
    language: SupportedLanguage,
) -> HashMap<String, Vec<String>> {
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
    result.retain(|(module, _)| module.first().map_or(false, |head| !skip_module_head(head)));

    // Group by module path: accumulate all symbols per module
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for (module, symbol) in result {
        map.entry(module.join("::")).or_default().push(symbol);
    }
    // Deduplicate symbols per module
    for symbols in map.values_mut() {
        symbols.sort();
        symbols.dedup();
    }
    map
}

impl Default for CodeSource {
    fn default() -> Self {
        Self {
            file_path: "default code file".into(),
            language: None,
            content_hash: None,
            module_comments: None,
            imports: Vec::new(),
            blocks: Vec::new(),
            inline_imports: None,
            embedding: Vec::new(),
            metadata: None,
        }
    }
}
use owo_colors::OwoColorize;

impl CodeSource {
    pub async fn parse(path: String) -> Result<CodeSource> {
        println!("{}", format!("{:?}", "[555] source text").cyan());

        let source_text =
            std::fs::read_to_string(&path).map_err(|e| anyhow!("Error reading file: {}", e))?;

        let _path = PathBuf::from(&path);
        println!("{}", format!("{:?}", "[555] ext").cyan());
        let ext = _path
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| anyhow!("No file extension".to_string()))?;

        let lang = SupportedLanguage::from_str(ext)
            .map_err(|_| anyhow!("Unsupported language '{}'", ext))?;

        println!("{}", format!("{:?}", "[555] lang").cyan());

        let lang_for_ii = lang.clone();

        let module_name = _path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();

        println!("{} {:?}", format!("{:?}", "[555] syntax").cyan(), &lang);
        let syntax = LANGUAGE_MAPPING
            .get(&lang)
            .clone()
            .ok_or_else(|| anyhow!("error getting syntax"))?
            .to_owned();

        println!("{}", format!("{:?}", "[555] parser").cyan());
        let mut parser = Parser::new();
        let _ = parser.set_language(PARSER_MAPPING.get(&lang).unwrap());

        let tree = parser
            .parse(&source_text, None)
            .ok_or(anyhow!("Parse returned None".to_string()))?;

        let root = tree.root_node();
        let src = source_text.as_bytes();

        let mut code_parser =
            CodeParser::new(path.clone(), lang, syntax, src.to_vec(), module_name);

        let ext_decl = code_parser.extract_declared_symbols(&root, src);

        println!("EXT DACL{:?}", ext_decl);

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

        let imported_symbols = code_parser
            .imports
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
            .collect::<Vec<String>>();

        let imported_modules: Vec<String> = code_parser
            .imports
            .iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| {
                b.imports
                    .keys()
                    .filter_map(|k| k.split("::").last().map(String::from))
            })
            .collect();

        let inline_imports =
            collect_inline_imports(root, src, &imported_symbols, &imported_modules, lang_for_ii);

        // Filter: used_symbols should only reference symbols known to this file
        // (declared, imported, or referenced via inline :: paths).
        let known: Vec<String> = {
            let mut k: Vec<String> = code_parser
                .blocks
                .iter()
                .flat_map(|b| b.symbol.iter().map(|s| s.name.clone()))
                .collect();
            for syms in inline_imports.values() {
                k.extend(syms.iter().cloned());
            }
            k.extend(
                code_parser
                    .imports
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
                            .collect::<Vec<_>>()
                    }),
            );
            k.sort();
            k.dedup();
            k
        };

        for block in &mut code_parser.blocks {
            block.used_symbols.retain(|s| known.contains(&s.name));
        }
        let mut hasher = DefaultHasher::new();
        source_text.hash(&mut hasher);

        let content_hash = hasher.finish() as i64;

        let metadata = fs::metadata(&code_parser.path).await.unwrap();

        Ok(CodeSource {
            file_path: code_parser.path,
            language: Some(code_parser.language),
            content_hash: Some(content_hash),
            inline_imports: Some(inline_imports),
            module_comments: None,
            imports: code_parser.imports,
            blocks: code_parser.blocks,
            embedding: Vec::new(),
            metadata: Some(metadata.into()),
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
    pub fn all_dependencies(&self) -> Result<Vec<String>> {
        let mut deps: Vec<String> = self.imported_symbols();
        if let Some(ref inline_imports) = self.inline_imports {
            for (module, symbols) in inline_imports {
                if module.is_empty() {
                    deps.extend(symbols.iter().cloned());
                } else {
                    deps.extend(symbols.iter().map(|sym| format!("{}::{}", module, sym)));
                }
            }
        }
        deps.sort();
        deps.dedup();

        Ok(deps)
    }
    /// Parse source text directly (for testing, or when source is already in memory).
    /// `path` is used for extension detection and module naming.
    pub async fn parse_text(path: String, source_text: &str) -> Result<CodeSource> {
        let _path = PathBuf::from(&path);
        let ext = _path
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| anyhow!("No file extension".to_string()))?;
        let lang = SupportedLanguage::from_str(ext)
            .map_err(|_| anyhow!("Unsupported language '{}'", ext))?;

        let lang_for_ii = lang.clone();
        let module_name = _path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_string();

        let syntax = LANGUAGE_MAPPING.get(&lang).unwrap().clone();

        let mut parser = Parser::new();
        let _ = parser.set_language(PARSER_MAPPING.get(&lang).unwrap());

        let tree = parser
            .parse(source_text, None)
            .ok_or(anyhow!("Parse returned None".to_string()))?;
        let root = tree.root_node();

        let src = source_text.as_bytes();
        let mut code_parser = CodeParser::new(path, lang, syntax, src.to_vec(), module_name);
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
        let imported_symbols = code_parser
            .imports
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
            .collect::<Vec<String>>();
        let imported_modules: Vec<String> = code_parser
            .imports
            .iter()
            .flat_map(|g| g.imports.iter())
            .flat_map(|b| {
                b.imports
                    .keys()
                    .filter_map(|k| k.split("::").last().map(String::from))
            })
            .collect();
        let inline_imports =
            collect_inline_imports(root, src, &imported_symbols, &imported_modules, lang_for_ii);
        let mut hasher = DefaultHasher::new();
        source_text.hash(&mut hasher);
        let content_hash = hasher.finish() as i64;
        let mut dir = fs::read_dir(&code_parser.path).await.unwrap();

        let entry = dir
            .next_entry()
            .await
            .map_err(|e| anyhow!("read dir: {e}"))?
            .ok_or_else(|| anyhow!("no more entries"))?;
        let metadata = entry.metadata().await?;

        Ok(CodeSource {
            file_path: code_parser.path,
            language: Some(code_parser.language),
            content_hash: Some(content_hash),
            inline_imports: Some(inline_imports),
            module_comments: None,
            imports: code_parser.imports,
            blocks: code_parser.blocks,
            embedding: Vec::new(),
            metadata: Some(metadata.into()),
        })
    }

    pub fn embed_blocks_mut(
        &mut self,
        dims: usize,
        batch_size: usize,
        embedder: &mut EmbeddingEngine,
    ) -> Result<()> {
        self.embedding = embedder.embed_code_blocks_mut(&mut self.blocks, batch_size, dims)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_declared_symbols() {
        let source = r#"
fn foo() {}
fn bar() {
    let x = foo();
}
"#;
        let cs = CodeSource::parse_text("test.rs".to_string(), source).unwrap();

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
        let cs = CodeSource::parse_text("test.rs".to_string(), source).unwrap();
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
        let cs = CodeSource::parse_text("test.rs".to_string(), source).unwrap();
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
        let cs = CodeSource::parse_text("test.rs".to_string(), source).unwrap();

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
