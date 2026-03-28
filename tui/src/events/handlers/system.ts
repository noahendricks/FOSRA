import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"

export function registerSystemHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
  loadInitialState: () => Promise<void>
) {
  router.on("permission.asked", (props: any) => {
    actions.setPermission(props.sessionID, props)
  })

  router.on("permission.replied", (props: any) => {
    actions.clearPermission(props.sessionID)
  })

  router.on("question.asked", (props: any) => {
    actions.setQuestion(props.sessionID, props)
  })

  router.on("question.replied", (props: any) => {
    actions.clearQuestion(props.sessionID)
  })

  router.on("question.rejected", (props: any) => {
    actions.clearQuestion(props.sessionID)
  })

  router.on("server.instance.disposed", () => {
    loadInitialState()
  })
}