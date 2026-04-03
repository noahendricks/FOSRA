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
import { log } from "@/util/log";

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
        log.store.debug("STORE_SET_MESSAGE", { operation: "update", sessionId, messageId: message.id, role: message.role, total: current.length });
      } else {
        state.messages.set(sessionId, [...current, message]);
        log.store.debug("STORE_SET_MESSAGE", { operation: "insert", sessionId, messageId: message.id, role: message.role, total: current.length + 1 });
      }
    },

    removeMessage(sessionId: string, messageId: string) {
      const current = state.messages.get(sessionId) ?? [];
      state.messages.set(
        sessionId,
        current.filter((m) => m.id !== messageId),
      );
      log.store.debug("STORE_REMOVE_MESSAGE", {
        sessionId,
        messageId,
        remaining: current.length - 1,
      });
    },

    setPart(messageId: string, part: Part) {
      const current = state.parts.get(messageId) ?? [];
      const idx = current.findIndex((p) => p.id === part.id);
      if (idx >= 0) {
        const updated = [...current];
        updated[idx] = part;
        state.parts.set(messageId, updated);
        log.store.debug("STORE_SET_PART", {
          messageId,
          partId: part.id,
          operation: "update",
          total: current.length,
        });
      } else {
        state.parts.set(messageId, [...current, part]);
        log.store.debug("STORE_SET_PART", {
          messageId,
          partId: part.id,
          operation: "insert",
          total: current.length + 1,
        });
      }
    },

    removePart(messageId: string, partId: string) {
      const current = state.parts.get(messageId) ?? [];
      state.parts.set(
        messageId,
        current.filter((p) => p.id !== partId),
      );
      log.store.debug("STORE_REMOVE_PART", {
        messageId,
        partId,
        remaining: current.length - 1,
      });
    },

    setTodo(sessionId: string, todo: Todo) {
      const current = state.todos.get(sessionId) ?? [];
      const idx = current.findIndex((t) => t.content === todo.content);

      if (idx >= 0) {
        const updated = [...current];
        updated[idx] = todo;
        state.todos.set(sessionId, updated);
        log.store.debug("STORE_SET_TODO", {
          sessionId,
          operation: "update",
          total: current.length,
        });
      } else {
        state.todos.set(sessionId, [...current, todo]);
        log.store.debug("STORE_SET_TODO", {
          sessionId,
          operation: "insert",
          total: current.length + 1,
        });
      }
    },

    removeTodo(sessionId: string, todoContent: string) {
      const current = state.todos.get(sessionId) ?? [];
      state.todos.set(
        sessionId,
        current.filter((t) => t.content !== todoContent),
      );
      log.store.debug("STORE_REMOVE_TODO", {
        sessionId,
        todoContent,
        remaining: current.length - 1,
      });
    },

    setPermission(sessionId: string, request: PermissionRequest) {
      const current = state.permission.get(sessionId) ?? [];
      const idx = current.findIndex((r) => r.id === request.id);

      if (idx >= 0) {
        const updated = [...current];
        updated[idx] = request;
        state.permission.set(sessionId, updated);
        log.store.debug("STORE_SET_PERMISSION", {
          sessionId,
          permissionId: request.id,
          operation: "update",
          total: current.length,
        });
      } else {
        state.permission.set(sessionId, [...current, request]);
        log.store.debug("STORE_SET_PERMISSION", {
          sessionId,
          permissionId: request.id,
          operation: "insert",
          total: current.length + 1,
        });
      }
    },

    clearPermission(sessionId: string) {
      state.permission.delete(sessionId);
      log.store.debug("STORE_CLEAR_PERMISSION", { sessionId });
    },

    setQuestion(sessionId: string, request: QuestionRequest) {
      state.question.set(sessionId, [request]);
      log.store.debug("STORE_SET_QUESTION", {
        sessionId,
        questionId: request.id,
      });
    },

    clearQuestion(sessionId: string) {
      state.question.delete(sessionId);
      log.store.debug("STORE_CLEAR_QUESTION", { sessionId });
    },

    applyDelta(
      sessionId: string,
      messageId: string,
      partId: string,
      delta: string,
      field: string = "text",
      partType: string = "text",
    ) {
      let parts = state.parts.get(messageId);

      if (!parts) {
        parts = [];
      }

      let idx = parts.findIndex((p) => p.id === partId);

      if (idx < 0) {
        const newPart = {
          id: partId,
          sessionID: sessionId,
          messageID: messageId,
          type: partType,
          text: field === "text" ? delta : "",
          time: { start: Date.now() },
        } as Part;
        if (field !== "text") {
          (newPart as any)[field] = delta;
        }
        parts = [...parts, newPart];
        log.store.debug("STORE_APPLY_DELTA", {
          messageId,
          partId,
          field,
          partType,
          operation: "create",
          resultLength: delta.length,
        });
        state.parts.set(messageId, parts);
        return;
      }

      const part = parts[idx] as any;
      const existing = part[field] as string | undefined;

      const updatedPart = { ...part, [field]: (existing ?? "") + delta };
      const newParts = [...parts];
      newParts[idx] = updatedPart;
      log.store.debug("STORE_APPLY_DELTA", {
        messageId,
        partId,
        field,
        operation: "update",
        resultLength: (updatedPart[field] as string)?.length ?? 0,
      });
      state.parts.set(messageId, newParts);
    },
  };
}

export type StoreActions = ReturnType<typeof createStoreActions>;
