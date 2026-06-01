use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use fosra::languages::{BlockType, CodeBlock, ImportBlock, SupportedLanguage, Symbol};
use owo_colors::{FgColorDisplay, OwoColorize};

use fosra::code::CodeSource;
use fosra::processing::embedding::EmbeddingEngine;
use pretty_assertions::{assert_eq, assert_ne};
use strum::VariantArray;
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::Ws;
use surrealdb::opt::auth::Root;

use fosra::{Document, Folder, SupportedFileTypes};

#[tokio::main]
async fn main() -> Result<()> {
    let db = Surreal::new::<Ws>("127.0.0.1:8000").await?;

    db.signin(Root {
        username: "root".to_string(),
        password: "secret".to_string(),
    })
    .await?;

    db.use_ns("main").use_db("main").await?;

    // ensure project root
    let base = "/home/roccoluxe/FOSRA";

    use ignore::WalkBuilder;
    let walker = WalkBuilder::new(base).git_ignore(true).build();
    println!("{}", "walk initialized".green().bold());

    // create all outer folders
    let mut folders: HashMap<PathBuf, Folder> = HashMap::new();

    fn default_folder(base: &Path) -> Folder {
        Folder {
            name: String::new(),
            base: base.to_string_lossy().into_owned(),
            ..Default::default()
        }
    }
    let model_dir = "/home/roccoluxe/models/embedding/onnx";

    let mut embedder = EmbeddingEngine::new(model_dir.into()).map_err(|e| anyhow!("{e}"))?;

    for entry in walker {
        let entry = entry?;
        let rel = entry.path().strip_prefix(base)?;

        match entry.file_type().ok_or_else(|| anyhow!("no file type"))? {
            ft if ft.is_dir() => {
                println!("{}", "is dir check".yellow().bold());
                folders.insert(
                    rel.to_path_buf(),
                    Folder::from_entry(&entry).ok_or_else(|| anyhow!("invalid dir"))?,
                );
            }
            ft if ft.is_file() => {
                println!("{}", "is file check".blue().bold());
                let parent = rel.parent().unwrap_or(&std::path::Path::new(""));

                let ext = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or_default();

                println!("{}", format!("{:?}", ext).red());
                if ext == "tsx" {
                    continue;
                }

                let meta = entry.metadata()?;
                let path_str = entry.path().to_string_lossy().into_owned();

                let parent_path = parent.to_path_buf();

                let will_run = Vec::from([SupportedLanguage::Rust, SupportedLanguage::Python]);

                let folder = folders
                    .entry(parent_path)
                    .or_insert_with(|| default_folder(Path::new(base)));
                if will_run // any supported file type
                    .iter()
                    .any(|f| f.as_ref() == ext)
                {
                    if ext == "tsx" {
                        continue;
                    }
                    println!("{}", "doc type check".yellow().bold());
                    // parse
                    let mut doc = Document::walk_md(path_str)?;

                    // embed
                    // println!("{}", "embed doc file".red().bold());

                    // _ = doc.embed_document(
                    //     &mut embedder,
                    //     896,
                    //     doc.content.clone().ok_or(anyhow!(""))?.len(),
                    // );

                    // push

                    doc.metadata = meta.into();
                    println!("{}", "place doc file".yellow().bold());
                    folder.docs.push(doc);
                } else if SupportedLanguage::VARIANTS // any supported language
                    .iter()
                    .any(|f| f.as_ref() == ext)
                {
                    println!("{}", "code file type check".yellow().bold());

                    // parse
                    let mut code = CodeSource::parse(path_str).await?;

                    // embed
                    // println!("{}", "embed code file".red().bold());
                    // _ = code.embed_blocks_mut(896, code.blocks.len(), &mut embedder);

                    // push
                    code.metadata = Some(meta.into());
                    println!("{}", "place code file".yellow().bold());
                    folder.code_docs.push(code);
                }
            }
            _ => {}
        }
    }

    use owo_colors::OwoColorize;

    for (_p, f) in &folders {
        println!("=== {} ===", f.name.to_uppercase().red().bold());
        println!("  {} ({})", f.name.blue(), f.base.yellow());
        println!("  total: {}", f.docs.len() + f.code_docs.len());
        println!();
        println!("{}", "=== Documents ===".bright_magenta().bold());
        for d in &f.docs {
            println!("  {}", d.file_path.bright_purple());
            println!(
                "  embedded: {}",
                (d.embedding != None).bright_yellow().bold()
            );
        }
        println!();
        println!("{}", "=== Code Sources ===".yellow().bold());
        fn type_name<T: ?Sized>(_: &T) -> &'static str {
            std::any::type_name::<T>()
        }

        for c in &f.code_docs {
            println!("  {}", c.file_path.yellow());
            let hash = &c.content_hash.ok_or(anyhow!("content hash empty"))?;

            println!(
                "  content hash: {} | type: {}",
                hash,
                type_name(hash).bold().green(),
            );
            if hash.to_string() == "0" {
                println!(" is empty: {}", true.bold().red());
            } else {
                println!(" is empty: {}", false.bold().green());
            };

            println!(
                "  embeddded: {}",
                (!c.embedding.is_empty()).bright_yellow().bold()
            );
        }
        println!("  total: {}", f.code_docs.len().green());

        println!();
    }

    // create base nodes for all folders

    use std::mem::take;
    for (p, mut f) in folders {
        // move doc & code files out
        let code_files = take(&mut f.code_docs);
        let doc_files = take(&mut f.docs);

        // raw folder
        let folder: Option<Folder> = db.create("folders").content(f).await?;

        // add code file links
        let seen: Vec<Symbol> = Vec::new();

        for f in &code_files {
            println!("==={}===", f.file_path.bold().yellow());
            let blocks = f.blocks.iter().map(|f| f.used_symbols.clone());

            let norm_imports: Vec<&ImportBlock> = f
                .imports
                .iter()
                .flat_map(|f| f.imports.iter().map(|i| i).collect::<Vec<&ImportBlock>>())
                .collect();

            let mut imports: Vec<String> = Vec::new();
            let mut bases: Vec<String> = Vec::new();

            for i in norm_imports {
                let imp: String = i
                    .imports
                    .iter()
                    .map(|i| {
                        let cat = String::new();
                        println!("base: {}", (i.0).bold().yellow());
                        bases.push(i.0.clone());
                        let b_split: Vec<&str> = i.0.split("::").collect();

                        for m in i.1.iter().map(|s| s.as_str()) {
                            let i_split: Vec<&str> = m.split(".").collect();

                            for r in &i_split {
                                imports.push(r.to_string());
                            }
                            if b_split == i_split {
                                println!("{}", b_split == i_split);
                                continue;
                            }
                            println!("{:?}", i_split)
                        }

                        cat
                    })
                    .collect();
            }

            imports = imports
                .iter()
                .map(|s| s.clone())
                .filter(|s| !bases.contains(s))
                .collect::<Vec<String>>();

            let decl_symbols: Vec<String> = f
                .declared_symbols()
                .iter()
                .map(|s| s.name.clone())
                .filter(|i| !f.imported_symbols().contains(i))
                .collect();

            for s in &decl_symbols {
                println!("decl: {}", s.bold().cyan());
            }

            let used_symbols: Vec<String> =
                f.used_symbols().iter().map(|s| s.name.clone()).collect();

            for s in &used_symbols {
                if decl_symbols.contains(s) {
                    println!("used decl sym: {}", s.bold().green());
                }
            }

            // Remove used_imports from used_symbols
            let used_imports: HashSet<&str> = imports
                .iter()
                .filter(|s| used_symbols.contains(s))
                .map(|s| s.as_str())
                .collect();

            let unused_imports: Vec<&String> = imports
                .iter()
                .filter(|x| !used_imports.contains(x.as_str()))
                .collect();

            for s in &used_imports {
                println!("used imports: {}", s.bold().yellow());
            }

            for u in unused_imports {
                println!("un-used imports: {}", u.bold().red());
            }

            let used_symbols: Vec<String> = used_symbols
                .clone()
                .into_iter()
                .filter(|s| !used_imports.contains(s.as_str()))
                .collect();

            let inl_imports = &f.inline_imports.clone().ok_or_else(|| anyhow!(""))?;

            for (k, v) in inl_imports {
                for s in v {
                    println!("{}: {}", k.bold().yellow(), s.bold().blue())
                }
            }
            let all_symbols: Vec<&String> = imports.iter().chain(decl_symbols.iter()).collect();

            // dbg!(norm_imports);
            // dbg!(inl_imports);
        }

        // add doc file links

        // dbg!(folder);
        // dbg!(code_files);
        // dbg!(doc_files);
    }

    // let code_doc_path = "/home/roccoluxe/FOSRA/core/examples/code.rs".to_string();
    // let mut code_doc = CodeSource::parse(code_doc_path).map_err(|e| anyhow!("{e}"))?;
    // // _ = code_doc.embed_blocks_mut(64, code_doc.blocks.len(), &mut embedder);
    // let code_doc_created: Option<CodeSource> = db.create("code").content(code_doc).await?.unwrap();

    // println!("{}", format!("{:?}", "=".repeat(250)).red());
    // dbg!(code_doc_created);
    // println!("{}", format!("{:?}", "=".repeat(250)).red());

    // let doc = Document::walk_md(PathBuf::from("/home/roccoluxe/FOSRA/ALGORITHMS.md")).unwrap();
    // println!("{}", format!("{:?}", "=".repeat(250)).yellow());
    // let doc_created: Option<Document> = db.create("document").content(doc).await?.unwrap();
    // println!("{}", format!("{:?}", "=".repeat(250)).yellow());

    Ok(())
}
