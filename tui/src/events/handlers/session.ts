import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"
import { log } from "@/util/log"

export function registerSessionHandlers(
  router: EventRouter["store"],
  actions: StoreActions
) {
  router.on("session.created", (props: any) => {
    log.store.info("SESSION_CREATED", { props })
    actions.set("sessions", props.info.id, props.info)
  })

  router.on("session.updated", (props: any) => {
    log.store.debug("SESSION_UPDATED", { props })
    const session = actions.get("sessions", props.info.id)
    if (session) {
      actions.set("sessions", props.info.id, { ...session, ...props.info })
    }
  })

  router.on("session.deleted", (props: any) => {
    log.store.info("SESSION_DELETED", { sessionID: props.info.id })
    actions.remove("sessions", props.info.id)
  })

  router.on("session.status", (props: any) => {
    log.store.debug("SESSION_STATUS", { sessionID: props.sessionID, status: props.status })
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status })
    }
  })

  router.on("session.error", (props: any) => {
    log.store.error("SESSION_ERROR", { sessionID: props.sessionID, error: props.error })
    const id = props.sessionID
    if (!id) return
    const session = actions.get("sessions", id)
    if (session) {
      actions.set("sessions", id, { ...session, error: props.error })
    }
  })

  router.on("busy" as any, (props: any) => {
    log.store.debug("SESSION_BUSY", { sessionID: props.sessionID, status: props.status })
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "busy" } })
    }
  })

  router.on("idle" as any, (props: any) => {
    log.store.debug("SESSION_IDLE", { sessionID: props.sessionID, status: props.status })
    const session = actions.get("sessions", props.sessionID)
    if (session) {
      actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "idle" } })
    }
  })
}