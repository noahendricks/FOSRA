import * as fs from "fs"
import * as path from "path"
import * as os from "os"
import chalk from "chalk"

// Force chalk to emit ANSI codes even when writing to file (not a TTY)
chalk.level = 3

const LOG_FILE =
  process.env.FOSRA_LOG_FILE ??
  path.join(os.homedir(), ".fosra-tui.log")

const CATEGORIES = [
  "startup",
  "sse",
  "store",
  "api",
  "route",
  "ui",
  "keybind",
  "theme",
  "config",
  "prompt",
  "session",
  "provider",
  "mcp",
  "error",
] as const

type Category = (typeof CATEGORIES)[number]
type Level = "debug" | "info" | "warn" | "error"

const LEVEL_ORDER: Record<Level, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
}

const LOG_LEVEL = (process.env.FOSRA_LOG_LEVEL ?? "info") as Level
const LOG_FORMAT = process.env.FOSRA_LOG_FORMAT ?? "pretty"

const LEVEL_COLORS: Record<Level, (s: string) => string> = {
  debug: chalk.gray,
  info: chalk.green,
  warn: chalk.yellow,
  error: (s) => chalk.red.bold(s),
}

const CAT_COLORS: Record<Category, (s: string) => string> = {
  startup: chalk.magenta,
  sse: chalk.cyan,
  store: chalk.blue,
  api: chalk.magenta,
  route: chalk.white,
  ui: chalk.green,
  keybind: chalk.yellow,
  theme: chalk.magenta,
  config: chalk.white,
  prompt: chalk.cyan,
  session: chalk.blue,
  provider: chalk.green,
  mcp: chalk.yellow,
  error: (s) => chalk.red.bold(s),
}

const CAT_PATHS: Record<Category, string> = {
  startup: "index.tsx",
  sse: "events/channel.ts",
  store: "store/actions.ts",
  api: "fosra/client/v2.ts",
  route: "context/route.tsx",
  ui: "ui/dialog.tsx",
  keybind: "context/keybind.tsx",
  theme: "context/theme.tsx",
  config: "context/kv.tsx",
  prompt: "component/prompt/index.tsx",
  session: "context/store.tsx",
  provider: "context/local.tsx",
  mcp: "component/dialog-mcp.tsx",
  error: "component/dialog-provider.tsx",
}

let dirReady = false

function ensureLogDir() {
  const dir = path.dirname(LOG_FILE)
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
}

// truncate only string values over maxLen, leave arrays/objects/numbers intact
function truncateStrings(obj: unknown, maxLen = 2000): unknown {
  if (obj === null || obj === undefined) return obj
  if (typeof obj === "string") {
    if (obj.length <= maxLen) return obj
    return obj.slice(0, maxLen) + `...[truncated ${obj.length} chars]`
  }
  if (Array.isArray(obj)) return obj.map((v) => truncateStrings(v, maxLen))
  if (typeof obj === "object") {
    const result: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      result[k] = truncateStrings(v, maxLen)
    }
    return result
  }
  return obj
}

function formatTimestamp(iso: string): string {
  // Strip to HH:MM:SS.mmm
  const d = new Date(iso)
  const h = d.getHours().toString().padStart(2, "0")
  const m = d.getMinutes().toString().padStart(2, "0")
  const s = d.getSeconds().toString().padStart(2, "0")
  const ms = d.getMilliseconds().toString().padStart(3, "0")
  return `${h}:${m}:${s}.${ms}`
}

function write(cat: Category, lvl: Level, msg: string, data?: unknown) {
  if (LEVEL_ORDER[lvl] < LEVEL_ORDER[LOG_LEVEL]) {
    return
  }

  if (!dirReady) {
    ensureLogDir()
    dirReady = true
  }

  if (data !== undefined) {
    try {
      data = truncateStrings(data)
    } catch {
      data = "[unserializable]"
    }
  }

  const ts = new Date().toISOString()

  if (LOG_FORMAT === "json") {
    const entry: Record<string, unknown> = { ts, cat, lvl, msg }
    if (data !== undefined) entry.data = data
    try {
      fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + "\n")
    } catch (_) {}
    return
  }

  const tsStr = chalk.dim(formatTimestamp(ts))
  const lvlStr = LEVEL_COLORS[lvl](lvl.padEnd(5))
  const catStr = CAT_COLORS[cat](CAT_PATHS[cat].padEnd(24))
  const msgStr = msg

  let out = `${tsStr} [${catStr}] ${lvlStr} ${msgStr}`

  if (data !== undefined) {
    out += "\n" + chalk.dim(JSON.stringify(data, null, 2))
  }

  out += "\n\n"

  try {
    fs.appendFileSync(LOG_FILE, out)
  } catch {
    // if logging itself fails, don't crash the app
  }
}

function scoped(cat: Category) {
  return {
    debug: (msg: string, data?: unknown) => write(cat, "debug", msg, data),
    info: (msg: string, data?: unknown) => write(cat, "info", msg, data),
    warn: (msg: string, data?: unknown) => write(cat, "warn", msg, data),
    error: (msg: string, data?: unknown) => write(cat, "error", msg, data),
  }
}

export const log = {
  startup: scoped("startup"),
  sse: scoped("sse"),
  store: scoped("store"),
  api: scoped("api"),
  route: scoped("route"),
  ui: scoped("ui"),
  keybind: scoped("keybind"),
  theme: scoped("theme"),
  config: scoped("config"),
  prompt: scoped("prompt"),
  session: scoped("session"),
  provider: scoped("provider"),
  mcp: scoped("mcp"),
  error: scoped("error"),
}

// backwards compat bridge
// log.default.info(msg, data) maps to startup category
// log.default.warn(msg, data) maps to error category
const bridge = {
  debug: (msg: string, ...args: unknown[]) =>
    write("startup", "debug", msg, args.length === 1 ? args[0] : args.length > 0 ? args : undefined),
  info: (msg: string, ...args: unknown[]) =>
    write("startup", "info", msg, args.length === 1 ? args[0] : args.length > 0 ? args : undefined),
  warn: (msg: string, ...args: unknown[]) =>
    write("error", "warn", msg, args.length === 1 ? args[0] : args.length > 0 ? args : undefined),
  error: (msg: string, ...args: unknown[]) =>
    write("error", "error", msg, args.length === 1 ? args[0] : args.length > 0 ? args : undefined),
}

export const Log = {
  Default: bridge,
}
