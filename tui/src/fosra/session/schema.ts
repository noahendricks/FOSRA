// stub for @/session/schema
import z from "zod"

export type SessionID = string & { readonly __brand: unique symbol }
export type MessageID = string & { readonly __brand: unique symbol }
export type PartID = string & { readonly __brand: unique symbol }

// ULID GENERATION
// backend uses python-ulid which produces uppercase crockford base32
// we must match the format so IDs sort consistently in Binary.search
const CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

function generateULID(): string {
  const now = Date.now()
  let ts = ""
  let t = now
  for (let i = 0; i < 10; i++) {
    ts = CROCKFORD[t % 32] + ts
    t = Math.floor(t / 32)
  }
  let rand = ""
  for (let i = 0; i < 16; i++) {
    rand += CROCKFORD[Math.floor(Math.random() * 32)]
  }
  return ts + rand
}

export const SessionID = {
  generate: () => generateULID() as SessionID,
  zod: z.string() as unknown as z.ZodType<SessionID>,
}
export const MessageID = {
  generate: () => generateULID() as MessageID,
  zod: z.string() as unknown as z.ZodType<MessageID>,
}
export const PartID = {
  generate: () => generateULID() as PartID,
  ascending: (a: PartID, b: PartID) => a.localeCompare(b),
  zod: z.string() as unknown as z.ZodType<PartID>,
}
