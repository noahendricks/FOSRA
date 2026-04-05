import type { Event } from "@fosra/api/v2"

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
    ;(handlers as Record<T, EventHandler<T>>)[type] = handler
  }

  function dispatch(event: Event) {
    const handler = handlers[event.type]
    if (handler) {
      ;(handler as EventHandler<typeof event.type>)(event.properties, event)
    }
  }

  return { on, dispatch, handlers }
}
