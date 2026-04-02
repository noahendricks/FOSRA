import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"

export function registerSessionHandlers(
  router: EventRouter["store"],
  actions: StoreActions
) {
  router.on("session.created", (props: any) => {
    actions.set("sessions", props.info.id, props.info)
  })

  router.on("session.updated", (props: any) => {
    const session = actions.get("sessions", props.info.id)
    if (session) {
      actions.set("sessions", props.info.id, { ...session, ...props.info })
    }
  })

  router.on("session.deleted", (props: any) => {
    actions.remove("sessions", props.info.id)
  })

  router.on("session.status", (props: any) => {
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status })
    }
  })

  router.on("session.error", (props: any) => {
    const id = props.sessionID
    if (!id) return
    const session = actions.get("sessions", id)
    if (session) {
      actions.set("sessions", id, { ...session, error: props.error })
    }
  })

  // backend sends "busy" and "idle" as standalone event types
  // in addition to the "session.status" envelope
  router.on("busy" as any, (props: any) => {
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "busy" } })
    }
  })

  router.on("idle" as any, (props: any) => {
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "idle" } })
    }
  })
}