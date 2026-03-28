import type { EventRouter } from "../router"

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
    deps.toast.show({
      title: props.title,
      message: props.message,
      description: props.description,
      variant: props.variant,
      duration: props.duration,
    })
  })

  router.on("tui.command.execute" as any, (props: any) => {
    deps.commands.trigger(props.command)
  })

  router.on("tui.prompt.append" as any, (props: any) => {
    deps.prompt.append(props.text)
  })

  router.on("tui.session.select" as any, (props: any) => {
    deps.navigate?.(props.sessionID)
  })
}