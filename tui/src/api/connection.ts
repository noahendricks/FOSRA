import type { ApiClient } from "./client"
import type { Event } from "@fosra/sdk/v2"

export type SSEConfig = {
  client: ApiClient
  signal?: AbortSignal
  onEvent: (event: Event) => void
  onError?: (err: unknown) => void
}

export async function connectSSE(config: SSEConfig): Promise<() => void> {
  let backoff = 1000
  let abort = false

  const run = async () => {
    while (!abort && !config.signal?.aborted) {
      try {
        const subscription = await config.client.fosra.event.subscribe({}, { signal: config.signal })
        backoff = 1000
        for await (const event of subscription.stream) {
          if (abort) break
          config.onEvent(event)
        }
      } catch (err) {
        if (abort || config.signal?.aborted) return
        config.onError?.(err)
        console.warn(`sse disconnected, reconnecting in ${backoff}ms`, err)
        await new Promise((r) => setTimeout(r, backoff))
        backoff = Math.min(backoff * 2, 30_000)
      }
    }
  }

  run()
  return () => { abort = true }
}