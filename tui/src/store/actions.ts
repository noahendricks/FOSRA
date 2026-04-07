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

  function upsertItem<T>(
    collection: ReactiveRecord<T[]>,
    parentId: string,
    item: T,
    findBy: (existing: T) => boolean,
    logTag: string,
    extra: (
      updated: T[],
      op: "update" | "insert",
    ) => Record<string, unknown> = () => ({}),
  ) {
    const current = collection.get(parentId) ?? [];
    const idx = current.findIndex(findBy);
    const op: "update" | "insert" = idx >= 0 ? "update" : "insert";
    const updated =
      idx >= 0
        ? (() => {
            const r = [...current];
            r[idx] = item;
            return r;
          })()
        : [...current, item];
    collection.set(parentId, updated);
    log.store.debug(logTag, {
      operation: op,
      total: updated.length,
      ...extra(updated, op),
    });
  }

  function getMessageCompleted(message: Message): number | null {
    if (message.role !== "assistant") return null;
    const t = message.time;
    if (!t) return null;
    return "completed" in t
      ? ((t as { completed?: number }).completed ?? null)
      : null;
  }

  function getPartTextLength(part: Part): number {
    if (part.type !== "text") return 0;
    return (part as { text?: string }).text?.length ?? 0;
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
      log.store.debug("CHECKING_STORE_CONFIG", {
        message,
      });
      upsertItem(
        state.messages,
        sessionId,
        message,
        (m) => m.id === message.id,
        "STORE_SET_MESSAGE",
        (updated) => ({
          sessionId,
          messageId: message.id,
          role: message.role,
          messageList: updated.map((m) => ({
            id: m.id,
            role: m.role,
            completed: getMessageCompleted(m),
          })),
        }),
      );
    },

    removeMessage(sessionId: string, messageId: string) {
      const current = state.messages.get(sessionId) ?? [];
      const updated = current.filter((m) => m.id !== messageId);
      state.messages.set(sessionId, updated);
      log.store.debug("STORE_REMOVE_MESSAGE", {
        sessionId,
        messageId,
        remaining: updated.length,
        messageList: updated.map((m) => ({
          id: m.id,
          role: m.role,
          completed: getMessageCompleted(m),
        })),
      });
    },

    setPart(messageId: string, part: Part) {
      upsertItem(
        state.parts,
        messageId,
        part,
        (p) => p.id === part.id,
        "STORE_SET_PART",
        (updated) => ({
          messageId,
          partId: part.id,
          type: part.type,
          partsList: updated.map((p) => ({
            id: p.id,
            type: p.type,
            textLen: getPartTextLength(p),
          })),
        }),
      );
    },

    removePart(messageId: string, partId: string) {
      const current = state.parts.get(messageId) ?? [];
      const updated = current.filter((p) => p.id !== partId);
      state.parts.set(messageId, updated);
      log.store.debug("STORE_REMOVE_PART", {
        messageId,
        partId,
        remaining: updated.length,
        partsList: updated.map((p) => ({
          id: p.id,
          type: p.type,
          textLen: getPartTextLength(p),
        })),
      });
    },

    setTodo(sessionId: string, todo: Todo) {
      upsertItem(
        state.todos,
        sessionId,
        todo,
        (t) => t.content === todo.content,
        "STORE_SET_TODO",
        () => ({ sessionId }),
      );
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
      upsertItem(
        state.permission,
        sessionId,
        request,
        (r) => r.id === request.id,
        "STORE_SET_PERMISSION",
        () => ({
          sessionId,
          permissionId: request.id,
        }),
      );
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
      let parts = state.parts.get(messageId) ?? [];

      let idx = parts.findIndex((p) => p.id === partId);

      if (idx < 0) {
        const newPart: Record<string, unknown> = {
          id: partId,
          sessionID: sessionId,
          messageID: messageId,
          type: partType,
          text: field === "text" ? delta : "",
          time: { start: Date.now() },
        };
        if (field !== "text") {
          newPart[field] = delta;
        }
        parts = [...parts, newPart as Part];
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

      const part = parts[idx];
      const existing = (part as Record<string, unknown>)[field] as
        | string
        | undefined;

      const updatedPart: Record<string, unknown> = {
        ...part,
        [field]: (existing ?? "") + delta,
        ...(partType && partType !== part.type ? { type: partType } : {}),
      };
      const newParts = [...parts];
      newParts[idx] = updatedPart as Part;
      log.store.debug("STORE_APPLY_DELTA", {
        messageId,
        partId,
        field,
        operation: "update",
        resultLength: String(updatedPart[field] ?? "").length,
      });
      state.parts.set(messageId, newParts);
    },
  };
}

export type StoreActions = ReturnType<typeof createStoreActions>;
