// stub for @/tool/tool — only types used by the tui
export namespace Tool {
  // permissive — real tool definitions have more fields,
  // but tui stubs only carry { name: string }
  export type Info<_P = any, _M = any> = {
    name: string
    [key: string]: any
  }

  export type InferParameters<_T> = any
  export type InferMetadata<_T> = any
}
