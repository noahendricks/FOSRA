// Stub for @/session/schema
export type SessionID = string & { readonly __brand: unique symbol }
export type MessageID = string & { readonly __brand: unique symbol }
export type PartID = string & { readonly __brand: unique symbol }

export const SessionID = {
  generate: () => crypto.randomUUID() as SessionID,
}
export const MessageID = {
  generate: () => crypto.randomUUID() as MessageID,
}
export const PartID = {
  generate: () => crypto.randomUUID() as PartID,
}