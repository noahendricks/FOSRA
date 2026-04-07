// Stub for @/bus/bus-event
import { z } from "zod"

interface BusEventDefinition<T extends z.ZodType> {
  type: string
  properties: T
}

export const BusEvent = {
  define: <T extends z.ZodType>(type: string, properties: T): BusEventDefinition<T> => ({
    type,
    properties,
  }),
}