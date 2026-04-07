// stub for @/session/schema
import z from "zod";

import { ulid } from "ulidx";

export type SessionID = string & { readonly __brand: unique symbol };
export type MessageID = string & { readonly __brand: unique symbol };
export type PartID = string & { readonly __brand: unique symbol };

export const SessionID = {
  generate: () => ulid() as SessionID,
  zod: z.string() as unknown as z.ZodType<SessionID>,
};
export const MessageID = {
  zod: z.string() as unknown as z.ZodType<MessageID>,
};
export const PartID = {
  ascending: (a: PartID, b: PartID) => a.localeCompare(b),
  zod: z.string() as unknown as z.ZodType<PartID>,
};
