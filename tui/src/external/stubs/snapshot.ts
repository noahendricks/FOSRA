// Stub for @/snapshot
export interface FileDiff {
  file: string
  before: string
  after: string
  additions: number
  deletions: number
}

export const Snapshot = {
  FileDiff: {} as FileDiff,
}