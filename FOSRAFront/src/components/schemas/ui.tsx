/**
 * UI Message types that extend AI SDK's UIMessage.
 * These bridge your backend types with the AI SDK.
 */
import {
  UIMessage,
  InferUITools,
  UIMessagePart,
  UIDataTypes,
  UITools,
} from "ai";
import { ulidDateInfo } from "@/hooks/utils/ulid";

import {
  SourceResponseDeep,
  ChunkWithScoreResponse,
  SourceGroupResponse,
} from "../../lib/api/types.gen";
import z from "zod";
import { zSourceGroupResponse } from "../../lib/api/zod.gen";

export type ulidDateInfoType = z.infer<typeof ulidDateInfo>;

export type SourceGroupType = z.infer<typeof zSourceGroupResponse>;

export type SourceDataShape = { sources: SourceGroupResponse[] };

// export type MessageDataType = UIMessagePart<
//   {
//     sources: SourceDataShape;
//     notification: {
//       "rag-status": {
//         stage:
//         | "fetching"
//         | "embedding"
//         | "searching"
//         | "reranking"
//         | "complete"
//         | "error";
//         progress?: number;
//         message?: string;
//       };
//     };
//   },
//   UITools
// >;
//
export interface DataShape extends UIMessage {
  timestamp?: Date;
  sourceIds?: string;
  convoId?: string;
  [key: string]: unknown;
  data: {
    sources: SourceGroupResponse[];
    notification: {
      "rag-status": {
        stage:
        | "fetching"
        | "embedding"
        | "searching"
        | "reranking"
        | "complete"
        | "error";
        progress?: number;
        message?: string;
      };
    };
  };
}

export type MyUIMessage = UIMessage<
  {
    timestamp?: Date | null;

    sourceIds?: string | null;

    convoId?: string | null;

    [key: string]: unknown;
  },
  {
    sources: SourceGroupResponse[] | null;
    notification: {
      "rag-status": {
        stage:
        | "fetching"
        | "embedding"
        | "searching"
        | "reranking"
        | "complete"
        | "error";
        progress?: number;
        message?: string;
      } | null;
    };
  }
>;

// export type MyUIMessage = UIMessage<
//   MessageMetadata,
//   MessageDataParts,
//   MessageTools
// >;

/**
 * Type guard to check if a part is a source URL part
 */
export function isSourcePart(
  part: MyUIMessage["parts"][number],
): part is Extract<MyUIMessage["parts"][number], { type: "source-url" }> {
  return part.type === "source-url";
}

/**
 * Type guard to check if a part is a source document part
 */
export function isSourceDocumentPart(
  part: MyUIMessage["parts"][number],
): part is Extract<MyUIMessage["parts"][number], { type: "source-document" }> {
  return part.type === "source-document";
}
