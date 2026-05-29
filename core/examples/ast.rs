//! # Options
//! | Flag           | Description                                      |
//! |----------------|--------------------------------------------------|
//! | `<file>`       | Source file to parse                             |
//! | `--kind <K>`   | Tree-sitter node kind to match (required)        |
//! | `--lang <L>`   | Language override (auto-detected from extension)  |
//! | `--max <N>`    | Max matches to display (default: all)            |
//! | `--text`       | Show full text content (default: truncated to fit)|

use std::io::{self, Read, Write};
use std::str::FromStr;

use console::{Term, measure_text_width};
use tree_sitter::{Node, Parser};

use fosra::{ImportBlock, LANGUAGE_MAPPING, SupportedLanguage, flatten_import_container};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut file_path: Option<String> = None;
    let mut lang: Option<String> = None;
    let mut target_kind: Option<String> = None;
    let mut max_matches: Option<usize> = None;
    let mut full_text = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--kind" => {
                i += 1;
                if i < args.len() {
                    target_kind = Some(args[i].clone());
                }
            }
            "--lang" => {
                i += 1;
                if i < args.len() {
                    lang = Some(args[i].clone());
                }
            }
            "--max" => {
                i += 1;
                if i < args.len() {
                    max_matches = args[i].parse::<usize>().ok();
                }
            }
            "--text" => full_text = true,
            _ if args[i].starts_with("--") => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
            _ => file_path = Some(args[i].clone()),
        }
        i += 1;
    }

    let kind = target_kind.unwrap_or_else(|| {
        eprintln!("ERROR: --kind <node-kind> is required");
        eprintln!("Example: --kind function_item");
        std::process::exit(1);
    });

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
        "rust" | "rs" => (tree_sitter_rust::LANGUAGE.into(), "rust"),
        "python" | "py" => (tree_sitter_python::LANGUAGE.into(), "python"),
        "typescript" | "ts" => (
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "typescript",
        ),
        "tsx" => (tree_sitter_typescript::LANGUAGE_TSX.into(), "tsx"),
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
    let source = source_code.as_str();

    // ── Collect matching nodes ──────────────────────────────────────────
    let mut matches: Vec<Node> = Vec::new();
    collect_matching(root, &kind, &mut matches, max_matches);

    // ── Terminal setup ──────────────────────────────────────────────────
    let term = Term::stdout();
    let term_width = term.size().1 as usize;

    // ── Output ──────────────────────────────────────────────────────────
    let out = &mut io::stdout();

    let sep = "═".repeat(term_width.saturating_sub(2).min(54));
    writeln!(out, "╔{sep}╗").ok();
    writeln!(out, "║  AST Node Examiner").ok();
    writeln!(out, "╠{sep}╣").ok();
    writeln!(out, "║  Language  : {lang_label}").ok();
    writeln!(out, "║  Node kind : {kind}").ok();
    writeln!(
        out,
        "║  File      : {}",
        file_path.as_deref().unwrap_or("<stdin>")
    )
    .ok();
    writeln!(
        out,
        "║  Source    : {} bytes, {} lines",
        source.len(),
        source.lines().count()
    )
    .ok();
    writeln!(out, "║  Matches   : {}", matches.len()).ok();
    writeln!(out, "╚{sep}╝").ok();
    writeln!(out).ok();

    if matches.is_empty() {
        writeln!(out, "⛔ No nodes of kind '{kind}' found.").ok();
        writeln!(out).ok();
        writeln!(out, "Tip: use `--lang rust` to override auto-detection, or").ok();
        writeln!(
            out,
            "     check the exact kind name with `--kind _` to list all kinds."
        )
        .ok();
        return;
    }

    for (idx, node) in matches.iter().enumerate() {
        let s = node.start_position();
        let e = node.end_position();

        let header = format!(
            " Match {}: {kind} ({}:{}..{}:{}) ",
            idx + 1,
            s.row,
            s.column,
            e.row,
            e.column,
        );
        let pad = term_width.saturating_sub(measure_text_width(&header) + 2);
        let header_line = if pad > 1 {
            let left = pad / 2;
            let right = pad - left;
            format!("╔{}{}{}╗", "═".repeat(left), header, "═".repeat(right))
        } else {
            format!("╔{header}╗")
        };
        writeln!(out, "{header_line}").ok();

        // ── Recursive subtree dump ──────────────────────────────────
        print_subtree(out, *node, source, "", None, true, full_text, term_width).ok();
        writeln!(out).ok();

        // ── Custom parsing logic ────────────────────────────────────
        // writeln!(out, "│ Custom parse ─────────────────────────────────────").ok();
        // let custom = parse_custom(*node, source, &lang_name);

        // for line in custom.lines() {
        //     let text_width = measure_text_width(line);
        //     let max_text = term_width.saturating_sub(6).min(80);

        //     if text_width > max_text {
        //         let mut remaining = line;

        //         while !remaining.is_empty() {
        //             let chunk: String = remaining.chars().take(max_text).collect();
        //             let chunk_width = measure_text_width(&chunk);
        //             let rest = term_width.saturating_sub(chunk_width + 4);

        //             writeln!(out, "│   {chunk}{}", " ".repeat(rest)).ok();
        //             remaining = &remaining[chunk.len()..];
        //         }
        //     } else {
        //         let rest = term_width.saturating_sub(text_width + 4);
        //         writeln!(out, "│   {line}{}", " ".repeat(rest)).ok();
        //     }
        // }
        writeln!(
            out,
            "└{}┘",
            "─".repeat(term_width.saturating_sub(3).min(66))
        )
        .ok();

        // ── Raw metadata ────────────────────────────────────────────
        let s = node.start_position();
        let e = node.end_position();
        let text = node_text(node, source, full_text);
        writeln!(
            out,
            "  kind={kind} named={} error={} extra={} missing={} id={}",
            node.is_named(),
            node.has_error(),
            node.is_extra(),
            node.is_missing(),
            node.id(),
        )
        .ok();
        writeln!(
            out,
            "  bytes {}..{}  pos ({}:{}..{}:{})",
            node.start_byte(),
            node.end_byte(),
            s.row,
            s.column,
            e.row,
            e.column,
        )
        .ok();
        writeln!(
            out,
            "  children {} total / {} named",
            node.child_count(),
            node.named_child_count(),
        )
        .ok();
        // Compact field-children summary
        {
            let mut field_info = String::new();
            let mut cur = node.walk();
            if cur.goto_first_child() {
                let mut first = true;
                loop {
                    let field = cur.field_name().unwrap_or("*");
                    let child = cur.node();
                    if !first {
                        field_info.push_str(", ");
                    }
                    field_info.push_str(&format!("{field}:{}", child.kind()));
                    first = false;
                    if !cur.goto_next_sibling() {
                        break;
                    }
                }
            }
            if !field_info.is_empty() {
                writeln!(out, "  fields: {field_info}").ok();
            }
        }
        writeln!(out, "  text: {text}").ok();
        let footer = "═".repeat(term_width.saturating_sub(2).min(54));
        writeln!(out, "╚{footer}╝").ok();
        writeln!(out).ok();
    }

    // ── Bonus: show unique kind index ────────────────────────────────────
    if kind == "_" {
        writeln!(out, "── All unique node kinds ──").ok();
        let mut kinds: Vec<String> = Vec::new();
        collect_kinds(root, &mut kinds);
        kinds.sort();
        kinds.dedup();
        let max_col = term_width / 24; // ~20 chars per column
        for (i, k) in kinds.iter().enumerate() {
            if max_col > 0 && i > 0 && i % max_col == 0 {
                writeln!(out).ok();
            }
            write!(out, "  {k:24}").ok();
        }
        writeln!(out).ok();
        writeln!(out, "── {total} kinds ──", total = kinds.len()).ok();
    }
}

fn parse_custom(node: Node, source: &str, lang: &str) -> String {
    let lang = SupportedLanguage::from_str(&lang).unwrap();
    let syntax = LANGUAGE_MAPPING.get(&lang).unwrap();

    let block = syntax.block_determine(node.kind()).unwrap();

    let mut import_container = ImportBlock::default();
    let source_bytes = source.as_bytes();
    let extra_paths: Vec<String> = Vec::new();
    let mut cursor = node.walk();

    let container = node.child_by_field_name("argument").unwrap();

    flatten_import_container(
        container,
        source_bytes,
        extra_paths,
        &mut import_container,
        syntax,
        &mut cursor,
    );

    let output = if !import_container.imports.is_empty() {
        format!(
            "block_type={:?}, node_type={:?}, imports={:?}",
            block.0, block.1, import_container.imports,
        )
    } else {
        format!("block_kind={:?}, name={:?}", block.0, block.1,)
    };

    // for m in &import_container.imports {
    //     output.push_str(&format!("\n  - import: {:?}", m));
    // }
    output
}

fn print_subtree(
    out: &mut impl Write,
    node: Node,
    source: &str,
    prefix: &str,             // tree branch lines from ancestors: "│   │   "
    field_name: Option<&str>, // this node's field name in its parent
    is_last: bool,
    full_text: bool,
    term_width: usize,
) -> io::Result<()> {
    let connector = if is_last { "└── " } else { "├── " };
    let child_prefix = if is_last { "    " } else { "│   " };

    let kind = node.kind();

    let field_annot = match field_name {
        Some(f) if !f.is_empty() => format!("{f}: "),
        _ => String::new(),
    };

    let mut flags = String::new();
    if !node.is_named() {
        flags.push_str(" [anon]");
    }
    if node.has_error() {
        flags.push_str(" [error]");
    }
    if node.is_missing() {
        flags.push_str(" [missing]");
    }

    let s = node.start_position();
    let e = node.end_position();
    let loc = format!("({}:{}..{}:{})", s.row, s.column, e.row, e.column);

    let text = node_text(&node, source, full_text);

    // Core: always-shown part (prefix + connector + field + kind + flags)
    let core = format!("{prefix}{connector}{field_annot}{kind}{flags}");
    let core_width = measure_text_width(&core);

    // Optional decorations, in priority order
    let text_str = if text.is_empty() {
        String::new()
    } else {
        format!(" {:?}", text)
    };
    let loc_str = format!(" {loc}");

    let mut line = core;
    let mut remaining = term_width.saturating_sub(core_width);

    // 2. Text snippet
    if remaining > 2 && !text_str.is_empty() {
        let tw = measure_text_width(&text_str);
        if tw <= remaining {
            line.push_str(&text_str);
            remaining -= tw;
        } else if remaining > 6 {
            let room = remaining.saturating_sub(2);
            let mut truncated: String = text_str.chars().take(room.saturating_sub(2)).collect();

            if measure_text_width(&truncated) + 3 <= remaining {
                truncated.push_str("…\"");
                line.push(' ');
                line.push_str(&truncated);
            }
            remaining = 0;
        }
    }

    // 3. Location
    if !loc_str.trim().is_empty() {
        let lw = measure_text_width(&loc_str);
        if lw <= remaining {
            line.push_str(&loc_str);
        }
    }

    writeln!(out, "{line}")?;

    // Recurse into ALL children (named + anonymous) — nothing hidden
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        let mut children: Vec<(Option<String>, Node)> = Vec::new();
        loop {
            let field = cursor.field_name().map(String::from);
            children.push((field, cursor.node()));
            if !cursor.goto_next_sibling() {
                break;
            }
        }

        let total = children.len();
        for (i, (child_field, child)) in children.iter().enumerate() {
            let child_is_last = i == total - 1;
            let new_prefix = format!("{prefix}{child_prefix}");
            print_subtree(
                out,
                *child,
                source,
                &new_prefix,
                child_field.as_deref(),
                child_is_last,
                full_text,
                term_width,
            )?;
        }
    }

    Ok(())
}

fn node_text(node: &Node, source: &str, full_text: bool) -> String {
    let raw = source.get(node.byte_range()).unwrap_or("");
    let capped = if !full_text && raw.len() > 2000 {
        &raw[..2000]
    } else {
        raw
    };

    let mut out = String::new();
    let mut lines = 0;
    for c in capped.chars() {
        match c {
            '\n' => {
                out.push('↵');
                lines += 1;
                if !full_text && lines > 5 {
                    out.push_str("…");
                    break;
                }
            }
            '\r' => out.push('␍'),
            '\t' => out.push('⇥'),
            c if c.is_control() && c != '\n' => {}
            c => out.push(c),
        }
    }
    if !full_text && measure_text_width(&out) > 120 {
        let mut truncated = String::new();
        let mut w = 0;
        for c in out.chars() {
            let cw = measure_text_width(&c.to_string());
            if w + cw > 117 {
                truncated.push('…');
                break;
            }
            truncated.push(c);
            w += cw;
        }
        truncated
    } else {
        out
    }
}

fn collect_matching<'tree>(
    node: Node<'tree>,
    kind: &str,
    matches: &mut Vec<Node<'tree>>,
    max: Option<usize>,
) {
    if matches.is_empty() || max.map_or(true, |m| matches.len() < m) {
        if kind == "_" || node.kind() == kind {
            matches.push(node);
        }
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                collect_matching(child, kind, matches, max);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }
}

fn collect_kinds(node: Node, kinds: &mut Vec<String>) {
    kinds.push(node.kind().to_string());
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_kinds(cursor.node(), kinds);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}
