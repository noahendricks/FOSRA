import { createSimpleContext } from "./helper"
import { createApiClient, type ApiConfig } from "../api/client"
import { createEventChannel, type ExternalEvents } from "../events/channel"

export type { ApiClient } from "../api/client"

const ctx = createSimpleContext({
  name: "Api",
  init: (props: { config: ApiConfig; events?: ExternalEvents }) => {
    const api = createApiClient(props.config)

    const channel = createEventChannel(
      props.events
        ? { mode: "external", source: props.events }
        : { mode: "sse", client: api }
    )

    channel.start()

    return {
      fosra: api.fosra,
      url: api.url,
      directory: api.directory,
      fetch: api.fetch,
      channel,
      dispose() {
        channel.stop()
      },
    }
  },
})

export const ApiProvider = ctx.provider
export const useApi = ctx.use