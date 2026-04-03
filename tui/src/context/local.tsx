import { createStore } from "solid-js/store"
import { batch, createEffect, createMemo } from "solid-js"
import { useStore } from "./store"
import { useApi } from "./api"
import { useTheme } from "@tui/context/theme"
import { uniqueBy } from "remeda"
import path from "path"
import { Global } from "@/global"
import { iife } from "@/util/iife"
import { createSimpleContext } from "./helper"
import { useToast } from "../ui/toast"
import { Provider } from "@/provider/provider"
import { useArgs } from "./args"
import { RGBA } from "@opentui/core"
import { Filesystem } from "@/util/filesystem"
import { log } from "@/util/log"

export const { use: useLocal, provider: LocalProvider } = createSimpleContext({
  name: "Local",
  init: () => {
    const store = useStore()
    const api = useApi()
    const toast = useToast()

    function isModelValid(model: { providerID: string; modelID: string }) {
      const provider = store.state.providers().find((x) => x.id === model.providerID)
      return !!provider?.models[model.modelID]
    }

    function getFirstValidModel(...modelFns: (() => { providerID: string; modelID: string } | undefined)[]) {
      for (const modelFn of modelFns) {
        const model = modelFn()
        if (!model) continue
        if (isModelValid(model)) return model
      }
    }

    // fallback when agents haven't loaded yet
    const EMPTY_AGENT = { name: "", mode: "", hidden: false } as any

    const agent = iife(() => {
      const agents = createMemo(() => store.state.agents().filter((x) => x.mode !== "subagent" && !x.hidden))
      const visibleAgents = createMemo(() => store.state.agents().filter((x) => !x.hidden))
      const [agentStore, setAgentStore] = createStore<{
        current: string
      }>({
        current: agents()[0]?.name ?? "",
      })
      const { theme } = useTheme()
      const colors = createMemo(() => [
        theme.secondary,
        theme.accent,
        theme.success,
        theme.warning,
        theme.primary,
        theme.error,
        theme.info,
      ])
      return {
        list() {
          return agents()
        },
        current() {
          return agents().find((x) => x.name === agentStore.current) ?? agents()[0] ?? EMPTY_AGENT
        },
        set(name: string) {
          if (!agents().some((x) => x.name === name))
            return toast.show({
              variant: "warning",
              message: `Agent not found: ${name}`,
              duration: 3000,
            })
          setAgentStore("current", name)
          log.provider.info("AGENT_SET", { name, agents: agents().map((x) => x.name) })
        },
        move(direction: 1 | -1) {
          batch(() => {
            const list = agents()
            if (!list.length) return
            let next = list.findIndex((x) => x.name === agentStore.current) + direction
            if (next < 0) next = list.length - 1
            if (next >= list.length) next = 0
            setAgentStore("current", list[next].name)
          })
        },
        color(name: string) {
          const index = visibleAgents().findIndex((x) => x.name === name)
          if (index === -1) return colors()[0]
          const agent = visibleAgents()[index]

          if (agent?.color) {
            const color = agent.color
            if (color.startsWith("#")) return RGBA.fromHex(color)
            // already validated by config, just satisfying TS here
            return theme[color as keyof typeof theme] as RGBA
          }
          return colors()[index % colors().length]
        },
      }
    })

    const model = iife(() => {
      const [modelStore, setModelStore] = createStore<{
        ready: boolean
        model: Record<
          string,
          {
            providerID: string
            modelID: string
          }
        >
        recent: {
          providerID: string
          modelID: string
        }[]
        favorite: {
          providerID: string
          modelID: string
        }[]
        variant: Record<string, string | undefined>
      }>({
        ready: false,
        model: {},
        recent: [],
        favorite: [],
        variant: {},
      })

      const filePath = path.join(Global.Path.state, "model.json")
      const state = {
        pending: false,
      }

      let writeTimer: ReturnType<typeof setTimeout> | null = null
      function flushWrite() {
        if (writeTimer) {
          clearTimeout(writeTimer)
          writeTimer = null
        }
        Filesystem.writeJson(filePath, {
          model: modelStore.model,
          recent: modelStore.recent,
          favorite: modelStore.favorite,
          variant: modelStore.variant ?? {},
        }).catch(() => {})
      }
      function scheduleWrite() {
        if (writeTimer) clearTimeout(writeTimer)
        writeTimer = setTimeout(flushWrite, 500)
      }
      if (typeof process !== "undefined" && process.on) {
        process.on("SIGTERM", flushWrite)
        process.on("SIGINT", flushWrite)
        process.on("exit", flushWrite)
      }

      function save() {
        if (!modelStore.ready) {
          state.pending = true
          return
        }
        state.pending = false
        scheduleWrite()
      }

      Filesystem.readJson(filePath)
        .then((x: any) => {
          if (typeof x.model === "object" && !Array.isArray(x.model)) setModelStore("model", x.model)
          if (Array.isArray(x.recent)) setModelStore("recent", x.recent)
          if (Array.isArray(x.favorite)) setModelStore("favorite", x.favorite)
          if (typeof x.variant === "object") setModelStore("variant", x.variant ?? {})
        })
        .catch(() => {})
        .finally(() => {
          setModelStore("ready", true)
          if (state.pending) save()
        })

      const args = useArgs()
      const fallbackModel = createMemo(() => {
        if (args.model) {
          const { providerID, modelID } = Provider.parseModel(args.model)
          if (isModelValid({ providerID, modelID })) {
            return {
              providerID,
              modelID,
            }
          }
        }

        if (store.state.config()?.model) {
          const { providerID, modelID } = Provider.parseModel(store.state.config()!.model)
          if (isModelValid({ providerID, modelID })) {
            return {
              providerID,
              modelID,
            }
          }
        }

        for (const item of modelStore.recent) {
          if (isModelValid(item)) {
            return item
          }
        }

        const provider = store.state.providers()[0] as any
        if (!provider) return undefined
        const defaultModel = store.state.providerDefault()[provider.id]
        const firstModel = Object.values(provider.models as any)[0] as any
        const model = defaultModel ?? firstModel?.id
        if (!model) return undefined
        return {
          providerID: provider.id,
          modelID: model,
        }
      })

      const currentModel = createMemo(() => {
        const a = agent.current()
        return (
          getFirstValidModel(
            () => modelStore.model[a?.name],
            () => a?.model,
            fallbackModel,
          ) ?? undefined
        )
      })

      return {
        current: currentModel,
        get ready() {
          return modelStore.ready
        },
        recent() {
          return modelStore.recent
        },
        favorite() {
          return modelStore.favorite
        },
        parsed: createMemo(() => {
          const value = currentModel()
          if (!value) {
            return {
              provider: "Connect a provider",
              model: "No provider selected",
              reasoning: false,
            }
          }
          const provider = store.state.providers().find((x) => x.id === value.providerID)
          const info = provider?.models[value.modelID]
          return {
            provider: provider?.name ?? value.providerID,
            model: info?.name ?? value.modelID,
            reasoning: info?.capabilities?.reasoning ?? false,
          }
        }),
        cycle(direction: 1 | -1) {
          const current = currentModel()
          if (!current) return
          const recent = modelStore.recent
          const index = recent.findIndex((x) => x.providerID === current.providerID && x.modelID === current.modelID)
          if (index === -1) return
          let next = index + direction
          if (next < 0) next = recent.length - 1
          if (next >= recent.length) next = 0
          const val = recent[next]
          if (!val) return
          const curr = agent.current()
          if (!curr) return
          setModelStore("model", curr.name, { ...val })
          log.provider.info("MODEL_CYCLED", { direction, model: current, recent: modelStore.recent })
        },
        cycleFavorite(direction: 1 | -1) {
          const favorites = modelStore.favorite.filter((item) => isModelValid(item))
          if (!favorites.length) {
            toast.show({
              variant: "info",
              message: "Add a favorite model to use this shortcut",
              duration: 3000,
            })
            return
          }
          const current = currentModel()
          let index = -1
          if (current) {
            index = favorites.findIndex((x) => x.providerID === current.providerID && x.modelID === current.modelID)
          }
          if (index === -1) {
            index = direction === 1 ? 0 : favorites.length - 1
          } else {
            index += direction
            if (index < 0) index = favorites.length - 1
            if (index >= favorites.length) index = 0
          }
          const next = favorites[index]
          if (!next) return
          const curr = agent.current()
          if (!curr) return
          setModelStore("model", curr.name, { ...next })
          const uniq = uniqueBy([next, ...modelStore.recent], (x) => `${x.providerID}/${x.modelID}`)
          if (uniq.length > 10) uniq.pop()
          setModelStore(
            "recent",
            uniq.map((x) => ({ providerID: x.providerID, modelID: x.modelID })),
          )
          save()
          log.provider.info("MODEL_FAVORITE_CYCLED", { direction, model: next, favorites: modelStore.favorite })
        },
        set(model: { providerID: string; modelID: string }, options?: { recent?: boolean }) {
          batch(() => {
            if (!isModelValid(model)) {
              toast.show({
                message: `Model ${model.providerID}/${model.modelID} is not valid`,
                variant: "warning",
                duration: 3000,
              })
              return
            }
            const curr = agent.current()
            if (!curr) return
            setModelStore("model", curr.name, model)
            if (options?.recent) {
              const uniq = uniqueBy([model, ...modelStore.recent], (x) => `${x.providerID}/${x.modelID}`)
              if (uniq.length > 10) uniq.pop()
              setModelStore(
                "recent",
                uniq.map((x) => ({ providerID: x.providerID, modelID: x.modelID })),
              )
            }
            save()
          })
          log.provider.info("MODEL_SET", { model, options, recent: modelStore.recent })
        },
        toggleFavorite(model: { providerID: string; modelID: string }) {
          batch(() => {
            if (!isModelValid(model)) {
              toast.show({
                message: `Model ${model.providerID}/${model.modelID} is not valid`,
                variant: "warning",
                duration: 3000,
              })
              return
            }
            const exists = modelStore.favorite.some(
              (x) => x.providerID === model.providerID && x.modelID === model.modelID,
            )
            const next = exists
              ? modelStore.favorite.filter((x) => x.providerID !== model.providerID || x.modelID !== model.modelID)
              : [model, ...modelStore.favorite]
            setModelStore(
              "favorite",
              next.map((x) => ({ providerID: x.providerID, modelID: x.modelID })),
            )
            save()
          })
          log.provider.info("MODEL_FAVORITE_TOGGLED", { model, favorited: !modelStore.favorite.some((x) => x.providerID === model.providerID && x.modelID === model.modelID), favorites: modelStore.favorite })
        },
        variant: {
          current() {
            const m = currentModel()
            if (!m) return undefined
            const key = `${m.providerID}/${m.modelID}`
            if (!modelStore.variant) return undefined
            return modelStore.variant[key]
          },
          list() {
            const m = currentModel()
            if (!m) return []
            const provider = store.state.providers().find((x) => x.id === m.providerID)
            const info = provider?.models[m.modelID]
            if (!info?.variants) return []
            return Object.keys(info.variants)
          },
          set(value: string | undefined) {
            const m = currentModel()
            if (!m) return
            const key = `${m.providerID}/${m.modelID}`
            setModelStore("variant", key, value)
            save()
            log.provider.debug("MODEL_VARIANT_SET", { variant: value, model: m })
          },
          cycle() {
            const variants = this.list()
            if (variants.length === 0) return
            const current = this.current()
            if (!current) {
              this.set(variants[0])
              log.provider.debug("MODEL_VARIANT_CYCLED", { variant: variants[0], model: currentModel(), variants })
              return
            }
            const index = variants.indexOf(current)
            if (index === -1 || index === variants.length - 1) {
              this.set(undefined)
              log.provider.debug("MODEL_VARIANT_CYCLED", { variant: undefined, model: currentModel(), variants })
              return
            }
            this.set(variants[index + 1])
            log.provider.debug("MODEL_VARIANT_CYCLED", { variant: variants[index + 1], model: currentModel(), variants })
          },
        },
      }
    })

    const mcp = {
      isEnabled(name: string) {
        const status = store.state.mcp.get(name)
        return status?.status === "connected"
      },
      async toggle(name: string) {
        const status = store.state.mcp.get(name)
        const willEnable = status?.status !== "connected"
        if (status?.status === "connected") {
          await (api.fosra.mcp as any).disconnect({ name })
        } else {
          await (api.fosra.mcp as any).connect({ name })
        }
        log.mcp.info("MCP_TOGGLED", { name, enabled: willEnable, status: store.state.mcp.get(name) })
      },
    }

    // Automatically update model when agent changes
    createEffect(() => {
      const value = agent.current()
      if (!value) return
      if (value.model) {
        if (isModelValid(value.model))
          model.set({
            providerID: value.model.providerID,
            modelID: value.model.modelID,
          })
        else
          toast.show({
            variant: "warning",
            message: `Agent ${value.name}'s configured model ${value.model.providerID}/${value.model.modelID} is not valid`,
            duration: 3000,
          })
      }
    })

    const result = {
      model,
      agent,
      mcp,
    }
    return result
  },
})
