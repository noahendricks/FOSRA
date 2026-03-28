import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"

export function registerMessageHandlers(
  router: EventRouter["store"],
  actions: StoreActions
) {
  router.on("message.updated", (props: any) => {
    actions.setMessage(props.metadata?.sessionID, props)
  })

  router.on("message.removed", (props: any) => {
    actions.removeMessage(props.sessionID, props.messageID)
  })

  router.on("message.part.updated", (props: any) => {
    actions.setPart(props.messageID, props.part)
  })

  router.on("message.part.removed", (props: any) => {
    actions.removePart(props.messageID, props.partID)
  })

  router.on("message.part.delta", (props: any) => {
    actions.applyDelta(props.messageID, props.partID, props.delta, props.field)
  })
}