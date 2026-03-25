// stub for @/util/filesystem
import nodePath from "path"

export const Filesystem = {
  read: async (path: string): Promise<string> => {
    const file = Bun.file(path)
    return file.text()
  },
  readBytes: async (path: string): Promise<Buffer> => {
    const file = Bun.file(path)
    const ab = await file.arrayBuffer()
    return Buffer.from(ab)
  },
  write: async (path: string, content: string): Promise<void> => {
    await Bun.write(path, content)
  },
  readJson: async <T>(path: string): Promise<T> => {
    const file = Bun.file(path)
    return await file.json()
  },
  writeJson: async (path: string, data: unknown): Promise<void> => {
    await Bun.write(path, JSON.stringify(data, null, 2))
  },
  exists: async (path: string): Promise<boolean> => {
    const file = Bun.file(path)
    return file.exists()
  },
  mkdir: async (path: string): Promise<void> => {
    const fs = require("fs")
    fs.mkdirSync(path, { recursive: true })
  },
  readText: async (path: string): Promise<string> => {
    const file = Bun.file(path)
    return file.text()
  },
  readArrayBuffer: async (path: string): Promise<ArrayBuffer> => {
    const file = Bun.file(path)
    return file.arrayBuffer()
  },
  mimeType: (path: string): string => {
    const ext = path.split(".").pop()?.toLowerCase() ?? ""
    const map: Record<string, string> = {
      txt: "text/plain", md: "text/markdown", json: "application/json",
      js: "text/javascript", ts: "text/typescript", html: "text/html",
      css: "text/css", png: "image/png", jpg: "image/jpeg", svg: "image/svg+xml",
    }
    return map[ext] ?? "application/octet-stream"
  },
  // walk upward from `start` looking for directories containing any of `targets`
  async *up(opts: {
    targets: string[]
    start: string
  }): AsyncGenerator<string> {
    let dir = opts.start
    while (true) {
      for (const target of opts.targets) {
        const candidate = nodePath.join(dir, target)
        const file = Bun.file(candidate)
        try {
          const stat = await file.exists()
          if (stat) yield candidate
        } catch {}
      }
      const parent = nodePath.dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  },
}
