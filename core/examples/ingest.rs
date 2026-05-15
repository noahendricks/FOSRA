//html (deferred): kuchikiki -> treemd

// pdf-inspector

// markdown: treemd
use treemd::parse_markdown;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = parse_markdown(
        std::fs::read_to_string("/home/roccoluxe/fosra-rust/z-misc/sample-files/sample-md.md")?
            .as_str(),
    );

    println!("{:#?}", p);

    Ok(())
}
