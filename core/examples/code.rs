use std::{env, path::PathBuf};

use fosra::ingestion::code_types::CodeSource;

fn main() {
    let args: Vec<String> = env::args().collect();
    let file_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        env::current_dir()
            .unwrap()
            .join("core")
            .join("examples")
            .join("ingest.rs")
    };

    let code_source = match CodeSource::parse(file_path) {
        Ok(cs) => cs,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    println!("=== Blocks ===");
    for block in &code_source.blocks {
        println!(
            "  block_id={:?} range={:?} lines={}-{} symbol={:?} used={:?}",
            block.block_id,
            block.range,
            block.line_start,
            block.line_end,
            block.symbol.iter().map(|s| &s.name).collect::<Vec<_>>(),
            block
                .used_symbols
                .iter()
                .map(|s| &s.name)
                .collect::<Vec<_>>(),
        );
    }

    println!("\n=== Imports ===");
    for group in &code_source.imports {
        for imp in &group.imports {
            println!(
                "  base_module={:?} imports={:?}",
                imp.base_module, imp.imports
            );
        }
    }

    println!("\n=== File-level symbol rollups ===");
    println!(
        "  declared: {:?}",
        code_source
            .declared_symbols()
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    println!(
        "  used:     {:?}",
        code_source
            .used_symbols()
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    println!("  imported: {:?}", code_source.imported_symbols());
    println!("\n=== Inline Imports (scoped-path) ===");
    if code_source.inline_imports.is_empty() {
        println!("  (none)");
    } else {
        for (module, symbols) in &code_source.inline_imports {
            println!("  {:?}: {:?}", module, symbols);
        }
    }
    println!("\n=== All Dependencies (combined) ===");
    for dep in &code_source.all_dependencies() {
        println!("  {dep}");
    }
    println!("  content_hash: {}", code_source.content_hash);
}
