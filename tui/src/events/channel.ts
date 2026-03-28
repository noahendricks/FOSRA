import { batch } from "solid-js"
import type { Event } from "@fosra/sdk/v2"
import type { ApiClient } from "../api/client"

export type ExternalEvents = {
  on: (handler: (event: Event) => void) => () => void
  setDirectory?: (directory?: string) => void
}

export type ChannelConfig =
  | { mode: "sse"; client: ApiClient; signal?: AbortSignal }
  | { mode: "external"; source: ExternalEvents }

function createBatcher(
  windowMs: number,
  flush: (events: Event[]) => void
) {
  let queue: Event[] = []
  let timer: ReturnType<typeof setTimeout> | undefined
  let lastFlush = 0

  function drain() {
    if (queue.length === 0) return
    const events = queue
    queue = []
    timer = undefined
    lastFlush = Date.now()
    flush(events)
  }

  function enqueue(event: Event) {
    queue.push(event)
    if (timer) return
    const elapsed = Date.now() - lastFlush
    if (elapsed < windowMs) {
      timer = setTimeout(drain, windowMs)
    } else {
      drain()
    }
  }

  function dispose() {
    if (timer) clearTimeout(timer)
    queue = []
  }

  return { enqueue, dispose }
}

async function connectSSE(
  client: ApiClient,
  onEvent: (event: Event) => void,
  signal: AbortSignal
) {
  let backoff = 1000

  while (!signal.aborted) {
    try {
      const subscription = await client.fosra.event.subscribe({}, { signal })
      backoff = 1000
      for await (const event of subscription.stream) {
        onEvent(event)
      }
    } catch (err) {
      if (signal.aborted) return
      console.warn(`sse disconnected, reconnecting in ${backoff}ms`, err)
      await new Promise((r) => setTimeout(r, backoff))
      backoff = Math.min(backoff * 2, 30_000)
    }
  }
}

export function createEventChannel(config: ChannelConfig) {
  const listeners = new Set<(event: Event) => void>()
  let cleanup: (() => void) | undefined

  const batcher = createBatcher(16, (events) => {
    batch(() => {
      for (const event of events) {
        for (const listener of listeners) {
          listener(event)
        }
      }
    })
  })

  function start() {
    if (config.mode === "sse") {
      const ctrl = new AbortController()
      connectSSE(config.client, batcher.enqueue, ctrl.signal)
      cleanup = () => ctrl.abort()
    } else {
      cleanup = config.source.on(batcher.enqueue)
    }
  }

  function stop() {
    cleanup?.()
    batcher.dispose()
  }

  return {
    subscribe(handler: (event: Event) => void) {
      listeners.add(handler)
      return () => listeners.delete(handler)
    },
    start,
    stop,
    restart() {
      stop()
      start()
    },
  }
}

export type EventChannel = ReturnType<typeof createEventChannel>