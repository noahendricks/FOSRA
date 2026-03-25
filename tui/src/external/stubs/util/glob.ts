import { Glob as BunGlob } from "bun"
import { minimatch } from "minimatch"

export namespace Glob {
  export interface Options {
    cwd?: string
    absolute?: boolean
    include?: "file" | "all"
    dot?: boolean
    symlink?: boolean
  }

  export async function scan(
    pattern: string,
    options: Options = {},
  ): Promise<string[]> {
    const glob = new BunGlob(pattern)
    const results: string[] = []
    for await (const match of glob.scan({
      cwd: options.cwd ?? ".",
      absolute: options.absolute,
      dot: options.dot,
      followSymlinks: options.symlink,
      onlyFiles: options.include !== "all",
    })) {
      results.push(match)
    }
    return results
  }

  export function scanSync(pattern: string, options: Options = {}): string[] {
    const glob = new BunGlob(pattern)
    return [...glob.scanSync({
      cwd: options.cwd ?? ".",
      absolute: options.absolute,
      dot: options.dot,
      followSymlinks: options.symlink,
      onlyFiles: options.include !== "all",
    })]
  }

  export function match(pattern: string, filepath: string): boolean {
    return minimatch(filepath, pattern, { dot: true })
  }
}