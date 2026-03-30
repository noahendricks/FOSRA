import type { EventRouter } from "../router";
import type { StoreActions } from "../../store/actions";
import { Log } from "@/util/log";

export function registerMessageHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
) {
  router.on("message.updated", (props: any) => {
    actions.setMessage(props.info.sessionID, props.info);
  });

  router.on("message.removed", (props: any) => {
    actions.removeMessage(props.sessionID, props.messageID);
  });

  router.on("message.part.updated", (props: any) => {
    actions.setPart(props.part.messageID, props.part);
  });

  router.on("message.part.removed", (props: any) => {
    actions.removePart(props.messageID, props.partID);
  });

  router.on("message.part.delta", (props: any) => {
    Log.Default.debug("[SSE HANDLER] message.part.delta:", JSON.stringify({ messageID: props.messageID, partID: props.partID, delta: props.delta?.slice(0, 50), field: props.field }))
    actions.applyDelta(props.messageID, props.partID, props.delta, props.field);
  });
}
