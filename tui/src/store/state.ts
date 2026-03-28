import { ReactiveMap } from "@solid-primitives/map"
import { createSignal } from "solid-js"
import type {
  Session,
  Message,
  Part,
  Todo,
  Provider,
  Agent,
  Config,
  PermissionRequest,
  QuestionRequest,
} from "../external/sdk-types"

export function createAppState() {
  const sessions = new ReactiveMap<string, Session>()
  const messages = new ReactiveMap<string, Message[]>()
  const parts = new ReactiveMap<string, Part[]>()
  const todos = new ReactiveMap<string, Todo[]>()

  const permissions = new ReactiveMap<string, PermissionRequest[]>()
  const questions = new ReactiveMap<string, QuestionRequest[]>()

  const [providers, setProviders] = createSignal<Provider[]>([])
  const [agents, setAgents] = createSignal<Agent[]>([])
  const [config, setConfig] = createSignal<Config | null>(null)

  const [directory, setDirectory] = createSignal("")

  const loadedSessions = new Set<string>()

  return {
    sessions,
    messages,
    parts,
    todos,
    permissions,
    questions,

    providers,
    setProviders,
    agents,
    setAgents,
    config,
    setConfig,

    directory,
    setDirectory,

    loadedSessions,
  }
}

export type AppState = ReturnType<typeof createAppState>