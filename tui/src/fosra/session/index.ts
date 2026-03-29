// Stub for @/session
import { SessionID } from "./schema"

export const Session = {
  isDefaultTitle: (title: string) => title.startsWith("New session"),
  Event: {
    Deleted: { type: "session.deleted" },
    Error: { type: "session.error" },
  },
}

export { SessionID }