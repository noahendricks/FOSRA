// FOSRA TUI ENTRY POINT
import { tui } from "./app"
import { TuiConfig } from "@/config/tui"
import type { Args } from "./context/args"
import { log } from "./fosra/util/log"

log.startup.info("TUI_START", {
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
      break
    case "--model":
    case "-m":
      args.model = process.argv[++i]
      break
    case "--session":
    case "-s":
      args.sessionID = process.argv[++i]
      break
    case "--continue":
    case "-c":
      args.continue = true
      break
    case "--fork":
      args.fork = true
      break
    case "--prompt":
    case "-p":
      args.prompt = process.argv[++i]
      break
  }
}

log.startup.info("CLI_ARGS_PARSED", { args })

const url = process.env.FOSRA_BACKEND_URL ?? "http://localhost:8000/oc"
log.startup.info("CONFIG_RESOLVED", { backendUrl: url })

try {
  const result = await tui({
    url,
    args,
    config: TuiConfig.Info,
    directory: process.cwd(),
  })
  log.startup.info("TUI_EXIT", { result })
} catch (err) {
  log.startup.error("TUI_FATAL", { error: err instanceof Error ? { message: err.message, stack: err.stack } : String(err) })
  throw err
}

log.startup.info("TUI_END")
process.exit(0)
