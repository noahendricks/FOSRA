import { createMemo } from "solid-js"
import { useSyncCompat } from "./compat/sync"
import { Global } from "@/global"

export function useDirectory() {
  const sync = useSyncCompat()
  return createMemo(() => {
    const directory = (sync.data.path as any).directory || process.cwd()
    const result = directory.replace(Global.Path.home, "~")
    if ((sync.data.vcs as any)?.branch) return result + ":" + (sync.data.vcs as any).branch
    return result
  })
}
