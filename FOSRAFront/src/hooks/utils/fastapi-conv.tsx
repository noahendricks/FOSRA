/**
 * ChatTransport for FastAPI backend.
 * Handles SSE streaming and request/response transformation.
 */
import { HttpChatTransport, UIMessageChunk } from "ai";
import type { MyUIMessage } from "@/components/schemas/ui";
import {
  zMessageRequest,
  zSourceGroupResponse,
  zUiMessage,
} from "../../lib/api/zod.gen";
import { SourceGroupResponse } from "../../lib/api/types.gen";
import z from "zod";
import { nanoid } from "nanoid";
import { transformSourceGroups } from "./converters";
import { printObjTypes } from "./debug";

export interface FastAPITransportOptions {
  baseUrl?: string;
  userId: string;
  workspaceId: string;
  convoId: string;
  headers?: Record<string, string>;
  credentials?: RequestCredentials;
}

/**
 * Custom transport that connects to FastAPI backend and transforms SSE events.
 */

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000; // 32KB chunks
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

type BackendMessagePart =
  | { type: "text"; text: string }
  | {
    type: "file";
    name: string;
    size: number;
    filename: string;
    bytes: ArrayBuffer;
    media_type: string;
    url: string | null;
  };

export class FastAPITransport extends HttpChatTransport<MyUIMessage> {
  private readonly baseUrl: string;
  private readonly userId: string;
  private readonly workspaceId: string;
  private readonly convoId: string;

  constructor({
    baseUrl = "http://localhost:8000",
    userId,
    workspaceId,
    convoId,
    headers = {},
    credentials = "same-origin",
  }: FastAPITransportOptions) {
    super({
      api: baseUrl,
      credentials,
      headers,

      prepareSendMessagesRequest: async ({
        messages,
        trigger,
        messageId,
        headers: requestHeaders,
        body: requestBody,
      }) => {
        console.log(`all messages: FastAPI`, messages);
        console.log(
          `types all messages: FastAPI`,
          messages.map((ins) => typeof ins),
        );
        console.log(`prior to parse`);
        console.log(messages);
        printObjTypes({ obj: messages });

        // Transform messages to match backend UIMessage format
        //
        try {
          const normalized = messages.map((msg) => ({
            Id: msg.id,
            Role: msg.role,
            Parts: msg.parts.map((part) => {
              if (part.type === "text") {
                return {
                  Type: part.type,
                  Text: part.text,
                };
              } else if (part.type === "file") {
                return {
                  Type: part.type,
                  Name: part.filename,
                  Size: part.bytes?.byteLength || 0,
                  Filename: part.filename,
                  Bytes: arrayBufferToBase64(part.bytes),
                  MediaType: part.mediaType,
                  Url: part.url,
                };
              }
              return {};
            }),
            MessageMetadata: msg.metadata || null,
          }));

          const ui_messages = await z
            .array(zUiMessage)
            .safeParseAsync(normalized);

          console.log("after parse");
          console.log(ui_messages);

          if (!ui_messages.success) {
            console.error("Validation failed:", ui_messages.error);
            throw ui_messages.error;
          }

          const body = {
            messages: ui_messages.data,
            trigger,
            MessageId: messageId || nanoid(),
            ConvoId: convoId,
            Role: "user",
            WorkspaceId: workspaceId,
            UserId: userId,
            ...requestBody,
          };

          const formatted = {
            api: `${baseUrl}/workspaces/${convoId}/send_message/`,
            headers: {
              "Content-Type": "application/json",
              "x-user-id": userId,
              "x-workspace-id": workspaceId,
              ...requestHeaders,
            },
            body: body,
          };

          return formatted;
        } catch (e) {
          console.log(e);
          throw e;
        }
      },

      prepareReconnectToStreamRequest: async ({ headers: requestHeaders }) => {
        return {
          api: `${baseUrl}/workspaces/${convoId}/reconnect/`,
          headers: {
            "Content-Type": "application/json",
            "x-user-id": userId,
            "x-workspace-id": workspaceId,
            ...requestHeaders,
          },
        };
      },
    });

    this.baseUrl = baseUrl;
    this.userId = userId;
    this.workspaceId = workspaceId;
    this.convoId = convoId;
  }

  /**
   *NOTE: Transform FastAPI SSE events into AI SDK Objects.
   * Handles: data events (text streaming), source events (RAG results), and status events (RAG progress)
   */
  protected processResponseStream(
    stream: ReadableStream<Uint8Array>,
  ): ReadableStream<any> {
    const decoder = new TextDecoder();
    let buffer = "";

    return new ReadableStream<UIMessageChunk>({
      async start(controller) {
        const reader = stream.getReader();

        try {
          while (true) {
            const { done, value } = await reader.read();

            if (done) {
              break;
            }

            buffer += decoder.decode(value, { stream: true }); //WARN: Could still be location of message bug

            // Process complete SSE events (separated by \n\n)
            const events = buffer.split("\n\n");

            buffer = events.pop() || ""; // Keep incomplete event in buffer

            for (const event of events) {
              if (!event.trim()) continue;

              try {
                // Parse SSE event line by line
                const lines: string[] = event.split("\n");

                for (const line of lines) {
                  // Source Events
                  //NOTE: Source Handling -
                  //WARN: Ensure sources hitting here, type may fallthrough
                  if (line.startsWith("source:")) {
                    const jsonStr = line.slice(7).trim(); // Remove "source:" prefix

                    try {
                      const parsed = JSON.parse(jsonStr);

                      const sourceGroups: SourceGroupResponse[] = z
                        .array(zSourceGroupResponse)
                        .parse(parsed);

                      // Enqueue all source groups in a single event
                      const PROVIDER_KEY = "rag-engine";

                      controller.enqueue({
                        type: "source-document",
                        sourceId: sourceGroups[0]?.Source.SourceId ?? "unknown",
                        mediaType: "application/json",
                        title:
                          sourceGroups[0]?.Source.Name ?? "Source Documents",

                        // 2. Wrap your NestedJSONMap inside the provider key
                        providerMetadata: {
                          [PROVIDER_KEY]: {
                            sourceGroups: transformSourceGroups(sourceGroups), // Your transformed data
                          },
                        },
                      });
                    } catch (parseError) {
                      console.warn(
                        "Failed to parse source event:",
                        parseError,
                        jsonStr,
                      );
                    }
                  }
                  //Status events
                  else if (line.startsWith("status:")) {
                    const jsonStr = line.slice(7).trim(); // Remove "status:" prefix

                    try {
                      const statusData = JSON.parse(jsonStr);

                      // Enqueue status update for UI feedback
                      //TODO: Format correctly and UI Component creation
                      controller.enqueue({
                        type: "data-notification",
                        data: {
                          status: statusData.status,
                          message: statusData.message,
                          step: statusData.step,
                          timestamp: Date.now(),
                        },
                      });
                    } catch (parseError) {
                      console.warn(
                        "Failed to parse status event:",
                        parseError,
                        jsonStr,
                      );
                    }
                  }
                  // Data / Text Events - text streaming
                  else if (line.startsWith("data:")) {
                    const jsonStr = line.slice(5).trim(); // Remove "data:" prefix

                    try {
                      const data = JSON.parse(jsonStr);

                      // Map backend events to AI SDK events
                      switch (data.type) {
                        case "start":
                        case "text-start":
                          controller.enqueue({
                            type: "text-start",
                            id: data.messageId || data.id,
                            providerMetadata: {
                              time: { timestamp: Date.now() },
                            },
                          });
                          break;

                        case "text-delta":
                          if (data.delta) {
                            controller.enqueue({
                              type: "text-delta",
                              delta: data.delta,
                              id: data.id,
                            });
                          }
                          break;

                        case "finish":
                          controller.enqueue({
                            type: "finish",
                            finishReason: data.finishReason || "stop",
                            messageMetadata: data.usage || {
                              promptTokens: 0,
                              completionTokens: 0,
                            },
                          });
                          return; // End stream after finish

                        case "error":
                          throw new Error(data.error || "Stream error");
                      }
                    } catch (dataParseError) {
                      console.warn(
                        "Failed to parse data event:",
                        dataParseError,
                        jsonStr,
                      );
                    }
                  }
                }
              } catch (eventError) {
                console.warn("Failed to process SSE event:", eventError);
                // Continue to other events
              }
            }
          }
        } catch (error) {
          controller.error(error);
        } finally {
          controller.close();
          reader.releaseLock();
        }
      },
      cancel() {
        // Stream cancellation handled by the base class
      },
    });
  }

  getUserId(): string {
    return this.userId;
  }

  getWorkspaceId(): string {
    return this.workspaceId;
  }

  getConvoId(): string {
    return this.convoId;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }
}

export function createFastAPITransport(
  options: FastAPITransportOptions,
): FastAPITransport {
  if (!options.userId) {
    throw new Error("FastAPITransport requires userId");
  }
  if (!options.workspaceId) {
    throw new Error("FastAPITransport requires workspaceId");
  }
  if (!options.convoId) {
    throw new Error("FastAPITransport requires convoId");
  }

  return new FastAPITransport(options);
}
