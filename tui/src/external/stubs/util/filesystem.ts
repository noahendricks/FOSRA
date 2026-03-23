// Stub for @/util/filesystem
export const Filesystem = {
  read: async (path: string): Promise<string> => {
    const file = Bun.file(path)
    return file.text()
  },
  write: async (path: string, content: string): Promise<void> => {
    await Bun.write(path, content)
  },
  readJson: async <T>(path: string): Promise<T | undefined> => {
    try {
      const file = Bun.file(path)
      return await file.json()
    } catch {
      return undefined
    }
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
}