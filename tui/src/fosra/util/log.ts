import * as fs from "fs"
import * as path from "path"
import * as os from "os"

const LOG_FILE = path.join(os.homedir(), ".fosra-tui.log")

function writeLog(level: string, ...args: unknown[]) {
  const timestamp = new Date().toISOString()
  const entry = {
    timestamp,
    level,
    ...((typeof args[0] === "string" ? { action: args[0] } : args[0]) as Record<string, unknown>),
  }
  const line = JSON.stringify(entry) + "\n"
  try {
    fs.appendFileSync(LOG_FILE, line)
  } catch {}
  const method = level === "ERROR" ? console.error : level === "WARN" ? console.warn : console.info
  method(`[${level}]`, ...args)
}

export const Log = {
  Default: {
    error: (...args: unknown[]) => writeLog("ERROR", ...args),
    warn: (...args: unknown[]) => writeLog("WARN", ...args),
    info: (...args: unknown[]) => writeLog("INFO", ...args),
    debug: (...args: unknown[]) => writeLog("DEBUG", ...args),
  },
}