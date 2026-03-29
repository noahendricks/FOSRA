// stub for @/util/process
import { spawn, type SpawnOptions } from "bun"

export namespace Process {
  interface RunOptions {
    nothrow?: boolean
    cwd?: string
    env?: Record<string, string>
  }

  interface RunResult {
    stdout: Buffer
    stderr: Buffer
    exitCode: number
    text: string
  }

  export async function run(
    cmd: string[],
    opts?: RunOptions,
  ): Promise<RunResult> {
    const proc = Bun.spawn(cmd, {
      stdout: "pipe",
      stderr: "pipe",
      cwd: opts?.cwd,
      env: opts?.env,
    })
    const stdout = Buffer.from(await new Response(proc.stdout).arrayBuffer())
    const stderr = Buffer.from(await new Response(proc.stderr).arrayBuffer())
    const exitCode = await proc.exited
    if (exitCode !== 0 && !opts?.nothrow) {
      throw new Error(`Command failed: ${cmd.join(" ")}`)
    }
    return { stdout, stderr, exitCode, text: stdout.toString() }
  }

  export async function text(
    cmd: string[],
    opts?: RunOptions,
  ): Promise<{ text: string }> {
    const result = await run(cmd, { ...opts, nothrow: true })
    return { text: result.stdout.toString().trim() }
  }

  export function spawn(
    cmd: string[],
    opts?: { stdin?: "pipe"; stdout?: "pipe" | "ignore"; stderr?: "pipe" | "ignore" },
  ) {
    const proc = Bun.spawn(cmd, {
      stdin: opts?.stdin === "pipe" ? "pipe" : undefined,
      stdout: opts?.stdout === "pipe" ? "pipe" : "ignore",
      stderr: opts?.stderr === "pipe" ? "pipe" : "ignore",
    })
    return {
      stdin: proc.stdin,
      stdout: proc.stdout,
      stderr: proc.stderr,
      exited: proc.exited,
    }
  }
}
