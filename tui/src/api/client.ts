import { createFosraClient, type Event } from "@fosra/sdk/v2"

export type ApiConfig = {
  baseUrl: string
  directory: string
  fetch?: typeof globalThis.fetch
  headers?: Record<string, string>
}

export type ApiClient = {
  fosra: ReturnType<typeof createFosraClient>
  url: string
  directory: string
  fetch: typeof globalThis.fetch
}

export function createApiClient(config: ApiConfig): ApiClient {
  const fosra = createFosraClient({
    baseUrl: config.baseUrl,
    directory: config.directory,
    fetch: config.fetch,
    headers: config.headers,
  })

  return {
    fosra,
    url: config.baseUrl,
    directory: config.directory,
    fetch: config.fetch ?? globalThis.fetch,
  }
}
