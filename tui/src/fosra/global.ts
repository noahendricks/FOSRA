// Stub for @/global
import * as path from "path"
import * as os from "os"

const homeDir = os.homedir()

export const Global = {
  Path: {
    home: homeDir,
    config: path.join(homeDir, ".config", "fosra"),
    state: path.join(homeDir, ".local", "state", "fosra"),
    worktree: process.cwd(),
    directory: process.cwd(),
  },
}
