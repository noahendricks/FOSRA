import type {
  Session,
  Message,
  Part,
  Todo,
  PermissionRequest,
  QuestionRequest,
} from "../fosra/sdk-types";
import { Accessor, Setter } from "solid-js";
import { Log } from "@/util/log";

export type EntityKey = "sessions" | "messages" | "parts" | "todos";

export function createStoreActions(state: {
  sessions: Map<string, Session>;
  sessionsArray: Accessor<Session[]>;
  setSessionsArray: Setter<Session[]>;
  messages: Map<string, Message[]>;
  parts: Map<string, Part[]>;
  todos: Map<string, Todo[]>;
  permission: Map<string, PermissionRequest[]>;
  question: Map<string, QuestionRequest[]>;
}) {
  function syncSessionsArray() {
    state.setSessionsArray([...state.sessions.values()]);
  }

  return {
    get<K extends EntityKey>(collection: K, id: string) {
      return (state[collection] as Map<string, any>).get(id);
    },

    set<K extends EntityKey>(collection: K, id: string, value: any) {
      (state[collection] as Map<string, any>).set(id, value);
      if (collection === "sessions") syncSessionsArray();
    },

    remove<K extends EntityKey>(collection: K, id: string) {
      (state[collection] as Map<string, any>).delete(id);
      if (collection === "sessions") syncSessionsArray();
    },

    setMessage(sessionId: string, message: Message) {
      const current = state.messages.get(sessionId) ?? [];
      const idx = current.findIndex((m) => m.id === message.id);

      if (idx >= 0) {
        current[idx] = message;
        state.messages.set(sessionId, [...current]);
      } else {
        state.messages.set(sessionId, [...current, message]);
      }
    },

    removeMessage(sessionId: string, messageId: string) {
      const current = state.messages.get(sessionId) ?? [];
      state.messages.set(
        sessionId,
        current.filter((m) => m.id !== messageId),
      );
    },

    setPart(messageId: string, part: Part) {
      const current = state.parts.get(messageId) ?? [];
      const idx = current.findIndex((p) => p.id === part.id);
      if (idx >= 0) {
        current[idx] = part;
        state.parts.set(messageId, [...current]);
      } else {
        state.parts.set(messageId, [...current, part]);
      }
    },

    removePart(messageId: string, partId: string) {
      const current = state.parts.get(messageId) ?? [];
      state.parts.set(
        messageId,
        current.filter((p) => p.id !== partId),
      );
    },

    setTodo(sessionId: string, todo: Todo) {
      const current = state.todos.get(sessionId) ?? [];
      const idx = current.findIndex((t) => t.content === todo.content);

      if (idx >= 0) {
        current[idx] = todo;
        state.todos.set(sessionId, [...current]);
      } else {
        state.todos.set(sessionId, [...current, todo]);
      }
    },

    removeTodo(sessionId: string, todoContent: string) {
      const current = state.todos.get(sessionId) ?? [];
      state.todos.set(
        sessionId,
        current.filter((t) => t.content !== todoContent),
      );
    },

    setPermission(sessionId: string, request: PermissionRequest) {
      const current = state.permission.get(sessionId) ?? [];
      const idx = current.findIndex((r) => r.id === request.id);

      if (idx >= 0) {
        current[idx] = request;
        state.permission.set(sessionId, [...current]);
      } else {
        state.permission.set(sessionId, [...current, request]);
      }
    },

    clearPermission(sessionId: string) {
      state.permission.delete(sessionId);
    },

    setQuestion(sessionId: string, request: QuestionRequest) {
      state.question.set(sessionId, [request]);
    },

    clearQuestion(sessionId: string) {
      state.question.delete(sessionId);
    },

    applyDelta(
      messageId: string,
      partId: string,
      delta: string,
      field: string = "text",
    ) {
      const parts = state.parts.get(messageId);

      if (!parts) return;

      const idx = parts.findIndex((p) => p.id === partId);
      if (idx < 0) return;

      const part = parts[idx] as any;
      const existing = part[field] as string | undefined;

      part[field] = (existing ?? "") + delta;
      state.parts.set(messageId, [...parts]);
    },
  };
}

export type StoreActions = ReturnType<typeof createStoreActions>;
