pub mod code_types;
pub mod doc_types;

pub trait IsSection {
    fn text(&self) -> &str;
}
