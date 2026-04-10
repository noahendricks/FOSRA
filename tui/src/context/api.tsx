import { createSimpleContext } from "./helper"
import { createFosraClient } from "@fosra/api/v2"
import { createEventChannel, type ExternalEvents } from "@tui/bus/channel"

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

const ctx = createSimpleContext({
  name: "Api",
  init: (props: { config: ApiConfig; events?: ExternalEvents }) => {
    const fosra = createFosraClient({
      baseUrl: props.config.baseUrl,
      directory: props.config.directory,
      fetch: props.config.fetch,
      headers: props.config.headers,
    })

    const channel = createEventChannel(
      props.events
        ? { mode: "external", source: props.events }
        : { mode: "sse", client: { fosra, url: props.config.baseUrl, directory: props.config.directory, fetch: props.config.fetch ?? globalThis.fetch } }
    )

    channel.start()

    return {
      fosra,
      url: props.config.baseUrl,
      directory: props.config.directory,
      fetch: props.config.fetch ?? globalThis.fetch,
      channel,
      dispose() {
        channel.stop()
      },
    }
  },
})

export const ApiProvider = ctx.provider
export const useApi = ctx.use
