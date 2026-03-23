// Stub for @/cli/ui
import { RGBA } from "@opentui/core"

const AGENT_COLORS: Record<string, RGBA> = {}

export const UI = {
  Agent: {
    color: (name: string): RGBA => {
      if (AGENT_COLORS[name]) return AGENT_COLORS[name]
      // Generate a consistent color based on the name
      const hash = name.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0)
      const hue = (hash * 137) % 360
      const r = Math.round(128 + 127 * Math.cos((hue * Math.PI) / 180))
      const g = Math.round(128 + 127 * Math.cos(((hue - 120) * Math.PI) / 180))
      const b = Math.round(128 + 127 * Math.cos(((hue - 240) * Math.PI) / 180))
      AGENT_COLORS[name] = RGBA.fromInts(r, g, b)
      return AGENT_COLORS[name]
    },
  },
}