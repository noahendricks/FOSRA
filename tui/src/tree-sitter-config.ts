// local parsers config for syntax highlighting
// replaces the monorepo-root parsers-config.ts
const parsers = {
  parsers: [
    {
      filetype: "python",
      wasm: "https://github.com/tree-sitter/tree-sitter-python/releases/download/v0.23.6/tree-sitter-python.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/tree-sitter/tree-sitter-python/master/queries/highlights.scm",
        ],
      },
    },
    {
      filetype: "typescript",
      wasm: "https://github.com/tree-sitter/tree-sitter-typescript/releases/download/v0.23.2/tree-sitter-typescript.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries/ecma/highlights.scm",
          "https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries/typescript/highlights.scm",
        ],
      },
    },
    {
      filetype: "javascript",
      wasm: "https://github.com/nicolo-ribaudo/tree-sitter-javascript/releases/download/v0.23.1/tree-sitter-javascript.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries/ecma/highlights.scm",
        ],
      },
    },
    {
      filetype: "go",
      wasm: "https://github.com/tree-sitter/tree-sitter-go/releases/download/v0.23.4/tree-sitter-go.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/tree-sitter/tree-sitter-go/master/queries/highlights.scm",
        ],
      },
    },
    {
      filetype: "rust",
      wasm: "https://github.com/tree-sitter/tree-sitter-rust/releases/download/v0.23.2/tree-sitter-rust.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/tree-sitter/tree-sitter-rust/master/queries/highlights.scm",
        ],
      },
    },
    {
      filetype: "json",
      wasm: "https://github.com/nicolo-ribaudo/tree-sitter-json/releases/download/v0.24.8/tree-sitter-json.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/tree-sitter/tree-sitter-json/master/queries/highlights.scm",
        ],
      },
    },
    {
      filetype: "bash",
      wasm: "https://github.com/nicolo-ribaudo/tree-sitter-bash/releases/download/v0.23.3/tree-sitter-bash.wasm",
      queries: {
        highlights: [
          "https://raw.githubusercontent.com/tree-sitter/tree-sitter-bash/master/queries/highlights.scm",
        ],
      },
    },
  ],
}

export default parsers
