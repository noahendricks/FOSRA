import { spawnSync } from "child_process"
import path from "path"

export function which(cmd: string, env?: NodeJS.ProcessEnv): string | null {
  const base = env?.PATH ?? env?.Path ?? process.env.PATH ?? process.env.Path ?? ""
  const full = base ? base + path.delimiter + path.join(process.env.HOME ?? "", ".fosra", "bin") : path.join(process.env.HOME ?? "", ".fosra", "bin")
  try {
    const result = spawnSync("which", [cmd], {
      encoding: "utf-8",
      env: { ...process.env, PATH: full, PATHEXT: env?.PATHEXT ?? env?.PathExt ?? process.env.PATHEXT ?? process.env.PathExt },
    })
    if (result.status === 0) return result.stdout.trim()
  } catch {}
  return null
}