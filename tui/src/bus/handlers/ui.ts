import type { EventRouter } from "../router"
import type { Route } from "../../context/route"
import type { ToastContext } from "@tui/components/dialogs/toast"
import { log } from "@/util/log"
import { EventSessionInfoSchema, EventSessionErrorSchema, EventInstallationUpdateAvailableSchema, EventTuiToastShowSchema, EventTuiCommandExecuteSchema, EventTuiPromptAppendSchema, EventTuiSessionSelectSchema } from "@/schemas"

type UIHandlerDeps = {
  toast: ToastContext
  commands: { trigger: (command: string) => void }
  prompt: { append: (text: string) => void }
  navigate?: (sessionID: string) => void
  route?: {
    data: Route
    navigate: (route: Route) => void
  }
}

export function registerUIHandlers(
  router: EventRouter["ui"],
  deps: UIHandlerDeps
) {
  router.on("session.deleted" as any, (props: any) => {
    const result = EventSessionInfoSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_SESSION_DELETED_INVALID", { error: result.error })
      return
    }
    log.ui.info("SESSION_DELETED", { sessionID: result.data.info.id })
    if (
      deps.route &&
      deps.route.data.type === "session" &&
      deps.route.data.sessionID === result.data.info.id
    ) {
      deps.route.navigate({ type: "home" })
      deps.toast.show({
        variant: "info",
        message: "The current session was deleted",
      })
    }
  })

  router.on("session.error" as any, (props: any) => {
    const result = EventSessionErrorSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_SESSION_ERROR_INVALID", { error: result.error })
      return
    }
    log.ui.info("SESSION_ERROR", { error: result.data.error })
    const error = result.data.error
    if (!error || typeof error !== "object") return
    if (error.name === "MessageAbortedError") return
    const data = (error as any).data
    const message = data?.message ?? String(error)
    deps.toast.show({ variant: "error", message, duration: 5000 })
  })

  router.on("installation.update-available" as any, (props: any) => {
    const result = EventInstallationUpdateAvailableSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_UPDATE_AVAILABLE_INVALID", { error: result.error })
      return
    }
    log.ui.info("UPDATE_AVAILABLE", { version: result.data.latestVersion })
    deps.toast.show({
      variant: "info",
      title: "Update Available",
      message: `OpenCode v${result.data.latestVersion} is available. Run 'opencode upgrade' to update manually.`,
      duration: 10000,
    })
  })

  router.on("tui.toast.show" as any, (props: any) => {
    const result = EventTuiToastShowSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_TOAST_SHOW_INVALID", { error: result.error })
      return
    }
    log.ui.info("TUI_TOAST_SHOW", { props: result.data })
    deps.toast.show({
      title: result.data.title,
      message: result.data.message,
      variant: result.data.variant,
      duration: result.data.duration,
    })
  })

  router.on("tui.command.execute" as any, (props: any) => {
    const result = EventTuiCommandExecuteSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_COMMAND_EXECUTE_INVALID", { error: result.error })
      return
    }
    log.ui.debug("TUI_COMMAND_EXECUTE", { command: result.data.command })
    deps.commands.trigger(result.data.command)
  })

  router.on("tui.prompt.append" as any, (props: any) => {
    const result = EventTuiPromptAppendSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_PROMPT_APPEND_INVALID", { error: result.error })
      return
    }
    log.ui.debug("TUI_PROMPT_APPEND", { textLength: result.data.text?.length })
    deps.prompt.append(result.data.text)
  })

  router.on("tui.session.select" as any, (props: any) => {
    const result = EventTuiSessionSelectSchema.safeParse(props)
    if (!result.success) {
      log.ui.warn("UI_SESSION_SELECT_INVALID", { error: result.error })
      return
    }
    log.ui.info("TUI_SESSION_SELECT", { sessionID: result.data.sessionID })
    deps.navigate?.(result.data.sessionID)
  })
}