import type { Event } from "@fosra/sdk/v2"

type EventOfType<T extends Event["type"]> = Extract<Event, { type: T }>

export type EventHandler<T extends Event["type"]> = (
  properties: EventOfType<T>["properties"],
  raw: EventOfType<T>
) => void

export type EventHandlerMap = {
  [K in Event["type"]]?: EventHandler<K>
}

export function createHandlerRegistry() {
  const handlers: EventHandlerMap = {}

  function on<T extends Event["type"]>(
    type: T,
    handler: EventHandler<T>
  ) {
    ;(handlers as any)[type] = handler
  }

  function dispatch(event: Event) {
    const handler = handlers[event.type]
    if (handler) {
      ;(handler as any)(event.properties, event)
    }
  }

  return { on, dispatch, handlers }
}
