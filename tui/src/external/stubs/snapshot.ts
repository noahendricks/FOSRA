// Stub for @/snapshot
export interface FileDiff {
  file: string
  before: string
  after: string
  additions: number
  deletions: number
}

export namespace Snapshot {
  export type Tree = any
  export const FileDiff = {} as FileDiff
}
