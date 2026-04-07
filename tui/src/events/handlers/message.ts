import type { EventRouter } from "../router";
import type { StoreActions } from "../../store/actions";
import { log } from "@/util/log";
import {
  EventMessageRemovedSchema,
  EventMessagePartRemovedSchema,
  MessageSchema,
  PartSchema,
} from "@/schemas";

export function registerMessageHandlers(
  router: EventRouter["store"],
  actions: StoreActions,
) {
  router.on("message.updated", (props: any) => {
    log.store.info("MESSAGE_UPDATED", { props });
    const parsed = MessageSchema.safeParse(props.info);
    if (!parsed.success) {
      log.store.warn("MESSAGE_UPDATED_INVALID", {
        error: parsed.error.format(),
        sessionID: props.info?.sessionID,
      });
      return;
    }
    actions.setMessage(props.info.sessionID, props.info);
  });

  router.on("message.removed", (props: any) => {
    log.store.debug("MESSAGE_REMOVED", {
      sessionID: props.sessionID,
      messageID: props.messageID,
    });
    const result = EventMessageRemovedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("MESSAGE_REMOVED_INVALID", { error: result.error.format() });
      return;
    }
    actions.removeMessage(result.data.sessionID, result.data.messageID);
  });

  router.on("message.part.updated", (props: any) => {
    log.store.debug("MESSAGE_PART_UPDATED", {
      messageID: props.part.messageID,
      partID: props.part.id,
      field: props.field,
      partType: props.partType,
    });
    const parsed = PartSchema.safeParse(props.part);
    if (!parsed.success) {
      log.store.warn("MESSAGE_PART_UPDATED_INVALID", {
        error: parsed.error.format(),
        partID: props.part?.id,
        messageID: props.part?.messageID,
      });
      return;
    }
    actions.setPart(props.part.messageID, props.part);
  });

  router.on("message.part.removed", (props: any) => {
    log.store.debug("MESSAGE_PART_REMOVED", {
      messageID: props.messageID,
      partID: props.partID,
    });
    const result = EventMessagePartRemovedSchema.safeParse(props);
    if (!result.success) {
      log.store.warn("MESSAGE_PART_REMOVED_INVALID", { error: result.error.format() });
      return;
    }
    actions.removePart(result.data.messageID, result.data.partID);
  });

  router.on("message.part.delta", (props: any) => {
    log.store.debug("MESSAGE_PART_DELTA", {
      sessionID: props.sessionID,
      messageID: props.messageID,
      partID: props.partID,
      field: props.field,
      deltaLength: props.delta?.length ?? 0,
    });
    // delta events carry partial data (e.g., text chunks) that can't be
    // validated against full schemas — validation happens on the store side
    // when the full part state is assembled from accumulated deltas.
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
