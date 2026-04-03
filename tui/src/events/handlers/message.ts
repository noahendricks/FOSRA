import type { EventRouter } from "../router";
import type { StoreActions } from "../../store/actions";
import { log } from "@/util/log";

export function registerMessageHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
) {
  router.on("message.updated", (props: any) => {
    log.store.info("MESSAGE_UPDATED", { props });
    actions.setMessage(props.info.sessionID, props.info);
  });

  router.on("message.removed", (props: any) => {
    log.store.debug("MESSAGE_REMOVED", { sessionID: props.sessionID, messageID: props.messageID });
    actions.removeMessage(props.sessionID, props.messageID);
  });

  router.on("message.part.updated", (props: any) => {
    log.store.debug("MESSAGE_PART_UPDATED", {
      messageID: props.part.messageID,
      partID: props.part.id,
      field: props.field,
      partType: props.partType,
    });
    actions.setPart(props.part.messageID, props.part);
  });

  router.on("message.part.removed", (props: any) => {
    log.store.debug("MESSAGE_PART_REMOVED", { messageID: props.messageID, partID: props.partID });
    actions.removePart(props.messageID, props.partID);
  });

  router.on("message.part.delta", (props: any) => {
    log.store.debug("MESSAGE_PART_DELTA", {
      sessionID: props.sessionID,
      messageID: props.messageID,
      partID: props.partID,
      field: props.field,
      deltaLength: props.delta?.length ?? 0,
    });
    actions.applyDelta(
      props.sessionID,
      props.messageID,
      props.partID,
      props.delta,
      props.field,
      props.partType,
    );
  });
}
