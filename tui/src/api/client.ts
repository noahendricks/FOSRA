import { createOpencodeClient, type Event } from "@fosra/sdk/v2"

export type ApiConfig = {
  baseUrl: string
  directory: string
  fetch?: typeof globalThis.fetch
  headers?: Record<string, string>
}

export type ApiClient = {
  client: ReturnType<typeof createOpencodeClient>
  url: string
  directory: string
  fetch: typeof globalThis.fetch
}

export function createApiClient(config: ApiConfig): ApiClient {
  const client = createOpencodeClient({
    baseUrl: config.baseUrl,
    directory: config.directory,
    fetch: config.fetch,
    headers: config.headers,
  })

  return {
    client,
    url: config.baseUrl,
    directory: config.directory,
    fetch: config.fetch ?? globalThis.fetch,
  }
}
