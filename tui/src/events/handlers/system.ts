import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"
import type { AppState } from "../../store/state"
import { log } from "@/util/log"

export function registerSystemHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
  loadInitialState: () => Promise<void>,
  state: AppState
) {
  router.on("permission.asked", (props: any) => {
    log.store.info("PERMISSION_ASKED", { props })
    actions.setPermission(props.sessionID, props)
  })

  router.on("permission.replied", (props: any) => {
    log.store.debug("PERMISSION_REPLIED", { sessionID: props.sessionID })
    actions.clearPermission(props.sessionID)
  })

  router.on("question.asked", (props: any) => {
    log.store.info("QUESTION_ASKED", { props })
    actions.setQuestion(props.sessionID, props)
  })

  router.on("question.replied", (props: any) => {
    log.store.debug("QUESTION_REPLIED", { sessionID: props.sessionID })
    actions.clearQuestion(props.sessionID)
  })

  router.on("question.rejected", (props: any) => {
    log.store.debug("QUESTION_REJECTED", { sessionID: props.sessionID })
    actions.clearQuestion(props.sessionID)
  })

  router.on("server.instance.disposed", () => {
    log.store.info("SERVER_INSTANCE_DISPOSED", {})
    loadInitialState()
  })

  router.on("lsp.updated", (_props: any) => {
    // LSP state is managed via REST polling, not SSE
  })

  router.on("vcs.branch.updated", (props: any) => {
    log.store.debug("VCS_BRANCH_UPDATED", { branch: props.branch })
    state.setVcs({ branch: props.branch ?? "" })
  })

  router.on("mcp.tools.changed", (_props: any) => {
    // MCP tools changed — could trigger a re-fetch of MCP status
  })
}