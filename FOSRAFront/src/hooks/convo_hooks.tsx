import * as React from "react";
import axios from "axios";
import * as z from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  ConvoFromBackend,
  ConvoFromBackendType,
  ConvoListItemDisplay,
  WorkspaceFromBackend,
} from "../components/schemas/domain";
import { toast } from "sonner";
import { NewMessageRequestType } from "@/components/schemas/domain";
import { useCallback, useMemo, useState } from "react";
import { Chat, useChat } from "@ai-sdk/react";
import { createFastAPITransport } from "./utils/fastapi-conv";
import { backendToUIMessages } from "@/hooks/utils/message-converters";
import type { MyUIMessage } from "@/components/schemas/ui";
import { useNavStore } from "./state-hooks";

import { DataUIPart, FileUIPart, UITools } from "ai";

import AsyncStorage, {
  useAsyncStorage,
} from "@react-native-async-storage/async-storage";
import {
  getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGet,
  MessageRequest,
} from "../lib/api";
import { queryClient } from "../main";
export interface UseProperChatOptions {
  convoId: string;
  onFinish?: (options: {
    message: MyUIMessage;
    messages: MyUIMessage[];
    finishReason?: string;
  }) => void;
  onError?: (error: Error) => void;
}

const api = axios.create({ baseURL: "http://localhost:8000" });

export function useProperChat({
  onFinish,
  onError,
  convoId,
}: UseProperChatOptions) {
  const queryClient = useQueryClient();
  const { activeUserId, activeWorkspaceId } = useNavStore();
  const { data: convoData, isLoading: isLoadingConvo } = useGetConvo(convoId);
  const [input, setInput] = useState("");

  const initialMessages = useMemo<MyUIMessage[]>(() => {
    if (!convoData?.Messages || convoData.Messages.length === 0) {
      return [];
    }

    return backendToUIMessages(convoData.Messages);
  }, [convoData?.Messages]);

  // Create transport
  const transport = useMemo(
    () =>
      createFastAPITransport({
        userId: activeUserId,
        workspaceId: activeWorkspaceId,
        baseUrl: "http://localhost:8000",
        convoId: convoId,
      }),
    [activeUserId, activeWorkspaceId, convoId],
  );

  // Create Chat instance with initial messages
  const chatInstance = useMemo(() => {
    return new Chat<MyUIMessage>({
      id: convoId,
      transport,
      messages: initialMessages,
    });
  }, [convoId, transport, initialMessages]);

  // Stable callbacks
  const handleFinish = useCallback(
    (event: {
      message: MyUIMessage;
      messages: MyUIMessage[];
      finishReason?: string;
    }) => {
      queryClient.invalidateQueries({ queryKey: ["convos", convoId] });
      toast.success("Message sent");
      onFinish?.(event);
    },
    [convoId, queryClient, onFinish],
  );

  const handleError = useCallback(
    (error: Error) => {
      toast.error("Failed to send message", { description: error.message });
      onError?.(error);
    },
    [onError],
  );

  // Use the chat hook with the Chat instance
  const chat = useChat<MyUIMessage>({
    chat: chatInstance,
    onFinish: handleFinish,
    onError: handleError,
  });

  // Custom sendMessage that uses our local input state
  //NOTE: Message Files are present when called?
  const sendMessage = useCallback(
    async (text: string, files?: FileList | FileUIPart[]) => {
      console.log("2. sendMessage called with:", { text, files });
      if (!text.trim() && !files?.length) {
        toast.error("Cannot send empty message");
        return;
      }

      await chat.sendMessage({
        text,
        files: files,
      });
      console.log("3. After chat.sendMessage");

      setInput("");
    },
    [chat],
  );

  const regenerateLastMessage = useCallback(async () => {
    const lastAssistantMessage = [...chat.messages]
      .reverse()
      .find((m) => m.role === "assistant");

    if (!lastAssistantMessage) {
      toast.error("No assistant message to regenerate");
      return;
    }

    //NOTE: Not working
    //TODO: Add functionality
    await chat.regenerate({
      messageId: lastAssistantMessage.id,
    });
  }, [chat]);

  // Input change handler
  //NOTE: May not be being called
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setInput(e.target.value);
    },
    [],
  );

  return {
    // Message state
    messages: chat.messages,
    setMessages: chat.setMessages,

    // Status
    status: chat.status,
    error: chat.error,
    isLoading: isLoadingConvo || chat.status === "streaming",

    // Actions
    sendMessage,
    regenerateLastMessage,
    stop: chat.stop,
    resumeStream: chat.resumeStream,
    clearError: chat.clearError,

    // Input management (custom implementation)
    chat: {
      input,
      setInput,
      handleInputChange,
    },
  };
}

/**
 * Safe version that handles missing convoId.
 */
export function useProperChatSafe(
  convoId: string | undefined,
  options?: Omit<UseProperChatOptions, "convoId">,
) {
  if (!convoId) {
    throw new Error("No conversation ID provided");
  }

  const fullOptions = useMemo(
    () => ({
      ...options,
      convoId: convoId,
    }),
    [options, convoId],
  );

  return useProperChat(fullOptions);
}

export const useCreateNewConversation = () => {
  const queryClient = useQueryClient();

  const { activeWorkspaceId, activeUserId } = useNavStore();

  return useMutation({
    mutationFn: async () => {
      const url = `/${activeUserId}/${activeWorkspaceId}/new_convo/`;

      const { data } = await api.post(
        url,
        {},
        {
          params: {
            user_id: activeUserId,
            workspace_id: activeWorkspaceId,
            title: "New Convo",
          },
        },
      );

      const convo_response = z.array(ConvoFromBackend).parse(data);
    },
    onSuccess: () => {
      // Refresh the sidebar list after creating
      queryClient.invalidateQueries({ queryKey: ["convos"] });
    },
  });
};

//NOTE: May not be being called
export const postNewUserMessage = async (
  convoId: string,
  message: MessageRequest,
) => {
  const url = `/workspaces/${convoId}/send_message/`;

  console.log(`message text: `, message.Text);
  const payload = {
    user_id: message.UserId,
    Role: message.Role,
    convo_id: convoId,
    text: message.Text,
    metadata: message.MessageMetadata,
    attached_files: [""],
  };

  const { data } = await api.post(url, payload);

  //NOTE: Currently returning the whole conversation
  const message_response = z.array(ConvoFromBackend).parse(data);

  return message_response;
};

//NOTE: Can be replaced with autogenned mutate
export const fetchConvo = async (
  ConvoId: string | null,
): Promise<ConvoFromBackendType | undefined> => {
  const { activeUserId } = useNavStore.getState();

  console.log(`Convo ID Init Fetch Convo: ${ConvoId}`);

  if (ConvoId) {
    const response = await api.get(
      `/workspaces/${activeUserId}/${ConvoId}/get_convo`,
    );

    const rawData = response.data;

    if (!rawData) {
      throw Error("No or Invalid Convo Response");
    }

    //TODO: Update type to synced backend type
    const parsed = ConvoFromBackend.safeParse(rawData);

    // console.log(parsed.data);

    if (!parsed.success) {
      console.log(parsed.error.cause);
      return ConvoFromBackend.parse(rawData);
    }
    return parsed.data;
  }
  return undefined;
};

export const useGetConvo = (ConvoId: string) => {
  return useQuery({
    queryKey: ["convos", ConvoId],
    queryFn: async () => fetchConvo(ConvoId),
  });
};

export const useGetConvoList = () => {
  const { activeUserId, activeWorkspaceId } = useNavStore();
  const getConvoList = React.useEffect(() => {
    if (!activeUserId) {
      throw Error("No User ID ");
    }

    const fetch = async () => {
      const results =
        await getListOfConvosWorkspacesUserIdWorkspaceIdGetConvosListGet({
          path: { user_id: activeUserId, workspace_id: activeWorkspaceId },
        });

      queryClient.invalidateQueries();
    };

    fetch();
  }, [activeUserId, activeWorkspaceId]);
};

//NOTE: replace with autogenned mutate
export const useGetWorkspaceList = () => {
  const { activeUserId } = useNavStore.getState();
  return useQuery({
    queryKey: ["workspaces", activeUserId],
    queryFn: async () => {
      const request = await api.get(
        `/workspaces/${activeUserId}/list_workspaces`,
      );

      const rawData = request.data;

      if (!rawData) {
        console.log(`Not RawData ${rawData}`);
      } else {
        console.log(`WS RawData ${rawData}`);
      }

      const parsed = z.array(WorkspaceFromBackend).parse(rawData);

      return parsed;
    },
    throwOnError: true,
  });
};
