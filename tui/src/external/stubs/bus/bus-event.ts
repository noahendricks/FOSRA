// Stub for @/bus/bus-event
import { z } from "zod"

export interface BusEventDefinition<T extends z.ZodType> {
  type: string
  properties: T
}

export const BusEvent = {
  define: <T extends z.ZodType>(def: { type: string; properties: T }): BusEventDefinition<T> => def,
}