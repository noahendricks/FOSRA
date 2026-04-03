import type { EventRouter } from "../router"
import { log } from "@/util/log"

export type UIHandlerDeps = {
  toast: { show: (opts: { title?: string; message?: string; description?: string; variant?: string; duration?: number }) => void }
  commands: { trigger: (command: string) => void }
  prompt: { append: (text: string) => void }
  navigate?: (sessionID: string) => void
}

export function registerUIHandlers(
  router: EventRouter["ui"],
  deps: UIHandlerDeps
) {
  router.on("tui.toast.show" as any, (props: any) => {
    log.ui.info("TUI_TOAST_SHOW", { props })
    deps.toast.show({
      title: props.title,
      message: props.message,
      description: props.description,
      variant: props.variant,
      duration: props.duration,
    })
  })

  router.on("tui.command.execute" as any, (props: any) => {
    log.ui.debug("TUI_COMMAND_EXECUTE", { command: props.command })
    deps.commands.trigger(props.command)
  })

  router.on("tui.prompt.append" as any, (props: any) => {
    log.ui.debug("TUI_PROMPT_APPEND", { textLength: props.text?.length })
    deps.prompt.append(props.text)
  })

  router.on("tui.session.select" as any, (props: any) => {
    log.ui.info("TUI_SESSION_SELECT", { sessionID: props.sessionID })
    deps.navigate?.(props.sessionID)
  })
}