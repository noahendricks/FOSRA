import type { EventRouter } from "../router"
import type { StoreActions } from "../../store/actions"
import type { AppState } from "../../store/state"

export function registerSystemHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
  loadInitialState: () => Promise<void>,
  state: AppState
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

  router.on("lsp.updated", (_props: any) => {
    // LSP state is managed via REST polling, not SSE
    // Server sends this event to trigger a re-fetch
  })

  router.on("vcs.branch.updated", (props: any) => {
    state.setVcs({ branch: props.branch ?? "" })
  })

  router.on("mcp.tools.changed", (_props: any) => {
    // MCP tools changed — could trigger a re-fetch of MCP status
  })
}