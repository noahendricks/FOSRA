import {
  MessageFromBackendType,
  NewMessageRequestType,
} from "@/components/schemas/domain";
import { DataShape, MyUIMessage } from "@/components/schemas/ui";
import { DataUIPart, TextUIPart, UIMessage, UIMessagePart } from "ai";
import { decodeTime } from "ulid";
import {
  SourceResponseDeep,
  ChunkWithScoreResponse,
  ChunkResponse,
  SourceGroupResponse,
  MessageResponse,
} from "../../lib/api/types.gen";
import { zSourceGroupResponse, zUiMessage } from "../../lib/api/zod.gen";
import z from "zod";

export function normalizeRole(role: string): "user" | "assistant" | "system" {
  const normalized = role.toLowerCase().trim();
  if (
    normalized === "user" ||
    normalized === "assistant" ||
    normalized === "system"
  ) {
    return normalized as "user" | "assistant" | "system";
  }
  console.warn(`Unknown role: ${role}, defaulting to 'user'`);
  return "user";
}

function parseAttachedSources(
  sources: Array<{ [key: string]: unknown }> | undefined,
): SourceGroupResponse[] {
  if (!sources || !Array.isArray(sources) || sources.length === 0) {
    return [];
  }

  try {
    const parsed = sources
      .map((source) => {
        try {
          return zSourceGroupResponse.parse(source) as SourceGroupResponse;
        } catch (error) {
          return null;
        }
      })
      .filter((s): s is SourceGroupResponse => s !== null);

    return parsed;
  } catch (error) {
    console.error("Error parsing attached sources:", error);
    return [];
  }
}

export function backendMessageTransform(msg: MessageResponse): MyUIMessage {
  const parts: MyUIMessage["parts"] = [];

  if (msg.Text) {
    parts.push({
      type: "text",
      text: msg.Text,
      state: "done",
    } as TextUIPart);
  }

  const parsedSources = parseAttachedSources(msg.AttachedSources);

  if (parsedSources.length > 0) {
    parts.push({
      type: "data-sources",
      data: parsedSources,
    } as any);

    parts.push({
      type: "data-notification",
      data: {
        "rag-status": {
          stage: "complete" as const,
          progress: 1,
        },
      },
    } as any);
  }
  if (parts.length === 0) {
    parts.push({
      type: "text",
      text: "",
      state: "done",
    } as TextUIPart);
  }

  const body = {
    id: msg.MessageId || crypto.randomUUID(),
    role: normalizeRole(msg.Role),
    parts,
    messageMetadata: {
      timestamp: msg.Timestamp ? new Date(msg.Timestamp) : undefined,
      sourceId: parsedSources.map((group) => group.Source.SourceId).join(","),
      convoId: msg.ConvoId,
    },
  };

  return body;
}
export function backendToUIMessages(
  messages: MessageResponse[],
): MyUIMessage[] {
  if (!messages || messages.length === 0) {
    return [];
  }

  const messageMap = new Map<string, MessageResponse>();
  messages.forEach((msg) => {
    const id = msg.MessageId;
    if (!id) return;

    const existing = messageMap.get(id);
    if (
      !existing ||
      (msg.Timestamp && msg.Timestamp > (existing.Timestamp || ""))
    ) {
      messageMap.set(id, msg);
    }
  });

  const deduped = Array.from(messageMap.values());

  deduped.sort(
    (a, b) => decodeTime(a.MessageId ?? "") - decodeTime(b.MessageId ?? ""),
  );

  return deduped.map(backendMessageTransform);
}

export function extractTextFromUIMessage(msg: MyUIMessage): string {
  if (!msg.parts || !Array.isArray(msg.parts)) {
    return "";
  }

  const textparts = msg.parts.filter(
    (part): part is TextUIPart => part.type === "text",
  );

  if (textparts.length === 0) {
    return "";
  }

  return textparts.map((part) => part.text || "").join("");
}

export function extractSourcesFromUIMessage(
  msg: MyUIMessage,
): SourceGroupResponse {
  if (!msg.parts) return [];
  const sources: SourceGroupResponse[] = [];

  for (const part of msg.parts) {
    if (part.type === "source-url" && "data" in part) {
      const data = part.data as any;
      if (data) {
        const result = zSourceGroupResponse.safeParse(data);
        if (result.success) {
          return result.data;
        }
      }
    }
  }
}

export function hasSourcesInUIMessage(msg: MyUIMessage): boolean {
  if (!msg.parts) return false;
  return msg.parts.some((p) => p.type === "data-rag-source");
}

export function hasFilesInUIMessage(msg: MyUIMessage): boolean {
  if (!msg.parts) return false;
  return msg.parts.some((p) => p.type === "file");
}
