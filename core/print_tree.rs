//! Recursive tree-sitter node tree printer — used via the `print_node_tree!` macro.
//!
//! ```ignore
//! use fosra::print_node_tree;
//! print_node_tree!(node, source);
//! ```

use tree_sitter::Node;

/// Print the full recursive subtree of `node` to stdout.
///
/// Every child (named + anonymous) is shown with:
/// - Field name (if any)
/// - Node kind
/// - Flags: `[anon]`, `[error]`
/// - Text snippet (truncated)
/// - Source position `(row:col..row:col)`
pub fn print_subtree(node: &Node, source: &str) {
    fn inner(node: &Node, source: &str, prefix: &str, field: Option<&str>, last: bool) {
        let conn = if last { "└── " } else { "├── " };
        let child_pre = if last { "    " } else { "│   " };

        let field_annot = field.map_or(String::new(), |f| format!("{f}: "));

        let mut flags = String::new();
        if !node.is_named() {
            flags.push_str(" [anon]");
        }
        if node.has_error() {
            flags.push_str(" [error]");
        }

        let text = fmt_text(node, source);
        let s = node.start_position();
        let e = node.end_position();

        println!(
            "{prefix}{conn}{field_annot}{kind}{flags} \"{text}\" ({r}:{c}..{re}:{ce})",
            kind = node.kind(),
            r = s.row,
            c = s.column,
            re = e.row,
            ce = e.column,
        );

        let mut cur = node.walk();
        if cur.goto_first_child() {
            let mut kids: Vec<(Option<String>, Node)> = Vec::new();
            loop {
                kids.push((cur.field_name().map(String::from), cur.node()));
                if !cur.goto_next_sibling() {
                    break;
                }
            }
            for (i, (f, kid)) in kids.iter().enumerate() {
                let new_pre = format!("{prefix}{child_pre}");
                inner(kid, source, &new_pre, f.as_deref(), i == kids.len() - 1);
            }
        }
    }
    inner(node, source, "", None, true);
}

/// Format node source text for display: escape control chars, cap length.
fn fmt_text(node: &Node, source: &str) -> String {
    let raw = source.get(node.byte_range()).unwrap_or("");
    let mut out = String::new();
    let mut lines = 0;
    for c in raw.chars() {
        match c {
            '\n' => {
                out.push('↵');
                lines += 1;
                if lines > 3 {
                    out.push('…');
                    break;
                }
            }
            '\r' => out.push('␍'),
            '\t' => out.push('⇥'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    if out.len() > 100 {
        let mut s: String = out.chars().take(97).collect();
        s.push('…');
        s
    } else {
        out
    }
}

/// Print the full recursive subtree of a tree-sitter `Node`.
///
/// # Usage
/// ```ignore
/// use fosra::print_node_tree;
///
/// let node = tree.root_node();
/// let source = "fn main() {}";
/// print_node_tree!(node, source);
/// ```
#[macro_export]
macro_rules! print_node_tree {
    ($node:expr, $source:expr) => {{
        $crate::print_tree::print_subtree(&$node, $source);
    }};
}
