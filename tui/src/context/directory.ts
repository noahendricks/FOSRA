import { createMemo } from "solid-js"
import { useStore } from "./store"
import { Global } from "@/global"

export function useDirectory() {
  const store = useStore()
  return createMemo(() => {
    const directory = process.cwd()
    const result = directory.replace(Global.Path.home, "~")
    return result
  })
}

