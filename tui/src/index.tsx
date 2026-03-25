// FOSRA TUI ENTRY POINT
import { tui } from "./app"
import { TuiConfig } from "@/config/tui"
import type { Args } from "./context/args"
import * as fs from "fs"
import * as path from "path"

const LOG_FILE = process.env.FOSRA_LOG_FILE ?? path.join(process.env.HOME ?? "/tmp", ".fosra-tui.log")

function logAction(action: string, details?: Record<string, unknown>) {
  const timestamp = new Date().toISOString()
  const entry = {
    timestamp,
    action,
    ...details,
  }
  const line = JSON.stringify(entry) + "\n"
  fs.appendFileSync(LOG_FILE, line)
}

function logError(action: string, error: unknown, details?: Record<string, unknown>) {
  const timestamp = new Date().toISOString()
  const entry = {
    timestamp,
    action,
    error: error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
    ...details,
  }
  const line = JSON.stringify(entry) + "\n"
  fs.appendFileSync(LOG_FILE, line)
}

// LOG RUN SEPARATOR
logAction("TUI_START", {
  separator: "═".repeat(60),
  pid: process.pid,
  cwd: process.cwd(),
  argv: process.argv,
  env: { FOSRA_BACKEND_URL: process.env.FOSRA_BACKEND_URL },
})

const args: Args = {}

// PARSE CLI ARGS
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i]
  switch (arg) {
    case "--agent":
    case "-a":
      args.agent = process.argv[++i]
      logAction("ARG_PARSED", { arg, value: args.agent })
      break
    case "--model":
    case "-m":
      args.model = process.argv[++i]
      logAction("ARG_PARSED", { arg, value: args.model })
      break
    case "--session":
    case "-s":
      args.sessionID = process.argv[++i]
      logAction("ARG_PARSED", { arg, value: args.sessionID })
      break
    case "--continue":
    case "-c":
      args.continue = true
      logAction("ARG_PARSED", { arg, value: true })
      break
    case "--fork":
      args.fork = true
      logAction("ARG_PARSED", { arg, value: true })
      break
    case "--prompt":
    case "-p":
      args.prompt = process.argv[++i]
      logAction("ARG_PARSED", { arg, value: args.prompt })
      break
  }
}

const url = process.env.FOSRA_BACKEND_URL ?? "http://localhost:8000/oc"
logAction("CONFIG_RESOLVED", { backendUrl: url, logFile: LOG_FILE })

try {
  const result = await tui({
    url,
    args,
    config: TuiConfig.Info,
    directory: process.cwd(),
  })
  logAction("TUI_EXIT", { result })
} catch (err) {
  logError("TUI_FATAL", err)
  throw err
}

logAction("TUI_END", { separator: "═".repeat(60) })
process.exit(0)
