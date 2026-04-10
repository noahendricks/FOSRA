import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"
import { log } from "@/util/log"
import { EventSessionInfoSchema, EventSessionStatusSchema, EventSessionErrorSchema } from "@/schemas"

export function registerSessionHandlers(
  router: EventRouter["store"],
  actions: StoreActions
) {
  router.on("session.created", (props: any) => {
    const result = EventSessionInfoSchema.safeParse(props)
    if (!result.success) {
      log.store.warn("SESSION_CREATED_INVALID", { error: result.error.format() })
      return
    }
    const data = result.data
    log.store.info("SESSION_CREATED", { props: data })
    actions.set("sessions", data.info.id, data.info)
  })

  router.on("session.updated", (props: any) => {
    const result = EventSessionInfoSchema.safeParse(props)
    if (!result.success) {
      log.store.warn("SESSION_UPDATED_INVALID", { error: result.error.format() })
      return
    }
    const data = result.data
    log.store.debug("SESSION_UPDATED", { props: data })
    const session = actions.get("sessions", data.info.id)
    if (session) {
      actions.set("sessions", data.info.id, { ...session, ...data.info })
    }
  })

  router.on("session.deleted", (props: any) => {
    const result = EventSessionInfoSchema.safeParse(props)
    if (!result.success) {
      log.store.warn("SESSION_DELETED_INVALID", { error: result.error.format() })
      return
    }
    const data = result.data
    log.store.info("SESSION_DELETED", { sessionID: data.info.id })
    actions.remove("sessions", data.info.id)
  })

  router.on("session.status", (props: any) => {
    const result = EventSessionStatusSchema.safeParse(props)
    if (!result.success) {
      log.store.warn("SESSION_STATUS_INVALID", { error: result.error.format() })
      return
    }
    const data = result.data
    log.store.debug("SESSION_STATUS", { sessionID: data.sessionID, status: data.status })
    const session = actions.get("sessions", data.sessionID)
    if (session) {
      actions.set("sessions", data.sessionID, { ...session, status: data.status })
    }
  })

  router.on("session.error", (props: any) => {
    const result = EventSessionErrorSchema.safeParse(props)
    if (!result.success) {
      log.store.warn("SESSION_ERROR_INVALID", { error: result.error.format() })
      return
    }
    const data = result.data
    log.store.error("SESSION_ERROR", { sessionID: data.sessionID, error: data.error })
    if (!data.sessionID) return
    const session = actions.get("sessions", data.sessionID)
    if (session) {
      actions.set("sessions", data.sessionID, { ...session, error: data.error })
    }
  })

  router.on("busy" as any, (props: any) => {
    const result = EventSessionStatusSchema.safeParse(props)
    if (!result.success) {
      log.store.debug("SESSION_BUSY_INVALID", { error: result.error.format() })
      const session = actions.get("sessions", props.sessionID)
      if (session) {
        actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "busy" } })
      }
      return
    }
    const data = result.data
    log.store.debug("SESSION_BUSY", { sessionID: data.sessionID, status: data.status })
    const session = actions.get("sessions", data.sessionID)
    if (session) {
      actions.set("sessions", data.sessionID, { ...session, status: data.status })
    }
  })

  router.on("idle" as any, (props: any) => {
    const result = EventSessionStatusSchema.safeParse(props)
    if (!result.success) {
      log.store.debug("SESSION_IDLE_INVALID", { error: result.error.format() })
      const session = actions.get("sessions", props.sessionID)
      if (session) {
        actions.set("sessions", props.sessionID, { ...session, status: props.status ?? { type: "idle" } })
      }
      return
    }
    const data = result.data
    log.store.debug("SESSION_IDLE", { sessionID: data.sessionID, status: data.status })
    const session = actions.get("sessions", data.sessionID)
    if (session) {
      actions.set("sessions", data.sessionID, { ...session, status: data.status })
    }
  })
}