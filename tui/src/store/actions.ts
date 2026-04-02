import type {
  Session,
  Message,
  Part,
  Todo,
  PermissionRequest,
  QuestionRequest,
} from "../fosra/sdk-types";
import type { ReactiveRecord } from "./state";
import { Accessor, Setter } from "solid-js";
import { Log } from "@/util/log";

export type EntityKey = "sessions" | "messages" | "parts" | "todos";

export function createStoreActions(state: {
  sessions: ReactiveRecord<Session>;
  sessionsArray: Accessor<Session[]>;
  setSessionsArray: Setter<Session[]>;
  messages: ReactiveRecord<Message[]>;
  parts: ReactiveRecord<Part[]>;
  todos: ReactiveRecord<Todo[]>;
  permission: ReactiveRecord<PermissionRequest[]>;
  question: ReactiveRecord<QuestionRequest[]>;
}) {
  function syncSessionsArray() {
    state.setSessionsArray(state.sessions.values());
  }

  return {
    get<K extends EntityKey>(collection: K, id: string) {
      return state[collection].get(id);
    },

    set<K extends EntityKey>(collection: K, id: string, value: any) {
      state[collection].set(id, value);
      if (collection === "sessions") syncSessionsArray();
    },

    remove<K extends EntityKey>(collection: K, id: string) {
      state[collection].delete(id);
      if (collection === "sessions") syncSessionsArray();
    },

    setMessage(sessionId: string, message: Message) {
      const current = state.messages.get(sessionId) ?? [];
      const idx = current.findIndex((m) => m.id === message.id);

      if (idx >= 0) {
        const updated = [...current];
        updated[idx] = message;
        state.messages.set(sessionId, updated);
        Log.Default.info("[STORE] setMessage UPDATE:", message.role, "id:", message.id, "session:", sessionId, "total:", current.length)
      } else {
        state.messages.set(sessionId, [...current, message]);
        Log.Default.info("[STORE] setMessage INSERT:", message.role, "id:", message.id, "session:", sessionId, "total:", current.length + 1)
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
        const updated = [...current];
        updated[idx] = part;
        state.parts.set(messageId, updated);
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
        const updated = [...current];
        updated[idx] = todo;
        state.todos.set(sessionId, updated);
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
        const updated = [...current];
        updated[idx] = request;
        state.permission.set(sessionId, updated);
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
      sessionId: string,
      messageId: string,
      partId: string,
      delta: string,
      field: string = "text",
    ) {
      let parts = state.parts.get(messageId);

      if (!parts) {
        parts = [];
      }

      let idx = parts.findIndex((p) => p.id === partId);

      if (idx < 0) {
        const newPart: Part = {
          id: partId,
          sessionID: sessionId,
          messageID: messageId,
          type: "text",
          text: field === "text" ? delta : "",
          time: { start: Date.now() },
        };
        if (field !== "text") {
          (newPart as any)[field] = delta;
        }
        parts = [...parts, newPart];
        Log.Default.info("[STORE] applyDelta: created new part", { messageId, partId, field, delta });
        state.parts.set(messageId, parts);
        return;
      }

      const part = parts[idx] as any;
      const existing = part[field] as string | undefined;

      const updatedPart = { ...part, [field]: (existing ?? "") + delta };
      const newParts = [...parts];
      newParts[idx] = updatedPart;
      Log.Default.info("[STORE] applyDelta: updated part", { messageId, partId, field, newText: updatedPart[field]?.slice(0, 50) });
      state.parts.set(messageId, newParts);
    },
  };
}

export type StoreActions = ReturnType<typeof createStoreActions>;
