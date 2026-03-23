// Re-export SDK types from the copied types file
export * from "../../../sdk-types"

// Stub client creation - will be replaced with FOSRA backend client
export const createOpencodeClient = (_options: {
  baseUrl?: string
  signal?: AbortSignal
  directory?: string
  fetch?: typeof fetch
  headers?: Record<string, string>
  experimental_workspaceID?: string
}) => {
  return {
    event: {
      subscribe: async function* () {},
    },
    config: {
      providers: async () => ({ providers: [], defaults: {} }),
      get: async () => ({}),
    },
    provider: {
      list: async () => ({ all: [], default: null, connected: [] }),
      auth: async () => [],
    },
    app: {
      agents: async () => [],
    },
    session: {
      list: async () => [],
      get: async () => null,
      messages: async () => [],
      todo: async () => [],
      diff: async () => [],
    },
    command: {
      list: async () => [],
    },
    lsp: {
      status: async () => [],
    },
    mcp: {
      status: async () => ({}),
    },
    experimental: {
      resource: {
        list: async () => ({}),
      },
    },
    formatter: {
      status: async () => [],
    },
    vcs: {
      get: async () => null,
    },
    path: {
      get: async () => ({ state: "", config: "", worktree: "", directory: "" }),
    },
  }
}