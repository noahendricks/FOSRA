import type { Session, Message, Part, Todo } from "../external/sdk-types"

export type EntityKey = "sessions" | "messages" | "parts" | "todos"

export function createStoreActions(state: {
  sessions: Map<string, Session>
  messages: Map<string, Message[]>
  parts: Map<string, Part[]>
  todos: Map<string, Todo[]>
  permission: Map<string, unknown[]>
  question: Map<string, unknown[]>
}) {
  return {
    set<K extends EntityKey>(collection: K, id: string, value: any) {
      (state[collection] as Map<string, any>).set(id, value)
    },

    remove<K extends EntityKey>(collection: K, id: string) {
      (state[collection] as Map<string, any>).delete(id)
    },

    setMessage(sessionId: string, message: Message) {
      const current = state.messages.get(sessionId) ?? []
      const idx = current.findIndex((m) => m.id === message.id)
      if (idx >= 0) {
        current[idx] = message
        state.messages.set(sessionId, [...current])
      } else {
        state.messages.set(sessionId, [...current, message])
      }
    },

    removeMessage(sessionId: string, messageId: string) {
      const current = state.messages.get(sessionId) ?? []
      state.messages.set(
        sessionId,
        current.filter((m) => m.id !== messageId)
      )
    },

    setPart(messageId: string, part: Part) {
      const current = state.parts.get(messageId) ?? []
      const idx = current.findIndex((p) => p.id === part.id)
      if (idx >= 0) {
        current[idx] = part
        state.parts.set(messageId, [...current])
      } else {
        state.parts.set(messageId, [...current, part])
      }
    },

    removePart(messageId: string, partId: string) {
      const current = state.parts.get(messageId) ?? []
      state.parts.set(
        messageId,
        current.filter((p) => p.id !== partId)
      )
    },

    setTodo(sessionId: string, todo: Todo) {
      const current = state.todos.get(sessionId) ?? []
      const idx = current.findIndex((t) => t.content === todo.content)
      if (idx >= 0) {
        current[idx] = todo
        state.todos.set(sessionId, [...current])
      } else {
        state.todos.set(sessionId, [...current, todo])
      }
    },

    removeTodo(sessionId: string, todoContent: string) {
      const current = state.todos.get(sessionId) ?? []
      state.todos.set(
        sessionId,
        current.filter((t) => t.content !== todoContent)
      )
    },

    setPermission(sessionId: string, request: unknown) {
      const current = state.permission.get(sessionId) ?? []
      state.permission.set(sessionId, [...current, request as any])
    },

    clearPermission(sessionId: string) {
      state.permission.delete(sessionId)
    },

    setQuestion(sessionId: string, request: unknown) {
      state.question.set(sessionId, [request as any])
    },

    clearQuestion(sessionId: string) {
      state.question.delete(sessionId)
    },

    applyDelta(messageId: string, partId: string, delta: string, field: string = "text") {
      const parts = state.parts.get(messageId)
      if (!parts) return
      const idx = parts.findIndex((p) => p.id === partId)
      if (idx < 0) return
      const part = parts[idx] as any
      const existing = part[field] as string | undefined
      part[field] = (existing ?? "") + delta
      state.parts.set(messageId, [...parts])
    },
  }
}

export type StoreActions = ReturnType<typeof createStoreActions>
