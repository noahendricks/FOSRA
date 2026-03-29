// Stub for @/cli/ui
import { RGBA } from "@opentui/core"

const AGENT_COLORS: Record<string, RGBA> = {}

export const UI = {
  Agent: {
    color: (name: string): RGBA => {
      if (AGENT_COLORS[name]) return AGENT_COLORS[name]
      const hash = name.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0)
      const hue = (hash * 137) % 360
      const r = Math.round(128 + 127 * Math.cos((hue * Math.PI) / 180))
      const g = Math.round(128 + 127 * Math.cos(((hue - 120) * Math.PI) / 180))
      const b = Math.round(128 + 127 * Math.cos(((hue - 240) * Math.PI) / 180))
      AGENT_COLORS[name] = RGBA.fromInts(r, g, b)
      return AGENT_COLORS[name]
    },
  },
  logo: (padding: string): string => {
    return [
      padding + "╔════════════════════════════════════════════════════════════════════════╗",
      padding + "║ ███████╗  █████╗  ███████╗ ██████╗   █████╗                             ║",
      padding + "║ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗                            ║",
      padding + "║ █████╗   ██║  ██║ ███████╗ ██████╔╝ ███████║                            ║",
      padding + "║ ██╔══╝   ██║  ██║ ╚════██║ ██╔══██╗ ██╔══██║                            ║",
      padding + "║ ██║      ╚█████╔╝ ███████║ ██║  ██║ ██║  ██║                            ║",
      padding + "║ ╚═╝       ╚════╝  ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝                            ║",
      padding + "╚════════════════════════════════════════════════════════════════════════╝",
    ].join("\n")
  },
  Style: {
    TEXT_DIM: "\x1b[2m",
    TEXT_NORMAL: "\x1b[0m",
    TEXT_NORMAL_BOLD: "\x1b[1m",
  },
}
