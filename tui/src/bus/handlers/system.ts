import type { EventRouter } from "../router";
import type { StoreActions } from "../../store/actions";
import type { AppState } from "../../store/state";
import { log } from "@/util/log";
import {
  PermissionRequestSchema,
  EventPermissionRepliedSchema,
  QuestionRequestSchema,
  EventQuestionRepliedSchema,
  EventQuestionRejectedSchema,
  EventVcsBranchUpdatedSchema,
} from "@/schemas";

export function registerSystemHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
  loadInitialState: () => Promise<void>,
  state: AppState,
) {
  router.on("permission.asked", (props: any) => {
    const result = PermissionRequestSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("PERMISSION_ASKED_INVALID", { props });
      return;
    }
    log.store.info("PERMISSION_ASKED", { props: result.data });
    actions.setPermission(result.data.sessionID, result.data);
  });

  router.on("permission.replied", (props: any) => {
    const result = EventPermissionRepliedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("PERMISSION_REPLIED_INVALID", { props });
      return;
    }
    log.store.debug("PERMISSION_REPLIED", { sessionID: result.data.sessionID });
    actions.clearPermission(result.data.sessionID);
  });

  router.on("question.asked", (props: any) => {
    const result = QuestionRequestSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("QUESTION_ASKED_INVALID", { props });
      return;
    }
    log.store.info("QUESTION_ASKED", { props: result.data });
    actions.setQuestion(result.data.sessionID, result.data);
  });

  router.on("question.replied", (props: any) => {
    const result = EventQuestionRepliedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("QUESTION_REPLIED_INVALID", { props });
      return;
    }
    log.store.debug("QUESTION_REPLIED", { sessionID: result.data.sessionID });
    actions.clearQuestion(result.data.sessionID);
  });

  router.on("question.rejected", (props: any) => {
    const result = EventQuestionRejectedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("QUESTION_REJECTED_INVALID", { props });
      return;
    }
    log.store.debug("QUESTION_REJECTED", { sessionID: result.data.sessionID });
    actions.clearQuestion(result.data.sessionID);
  });

  router.on("server.instance.disposed", () => {
    log.store.info("SERVER_INSTANCE_DISPOSED", {});
    loadInitialState();
  });

  router.on("lsp.updated", (_props: any) => {
    // LSP state is managed via REST polling, not SSE
  });

  router.on("vcs.branch.updated", (props: any) => {
    const result = EventVcsBranchUpdatedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("VCS_BRANCH_UPDATED_INVALID", { props });
      return;
    }
    log.store.debug("VCS_BRANCH_UPDATED", { branch: result.data.branch });
    state.setVcs({ branch: result.data.branch ?? "" });
  });

  router.on("mcp.tools.changed", (_props: any) => {
    // MCP tools changed — could trigger a re-fetch of MCP status
  });
}
