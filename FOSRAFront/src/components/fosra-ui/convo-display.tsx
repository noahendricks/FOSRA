import { createFileRoute, useParams } from "@tanstack/react-router";
("use client");

import * as React from "react";
import { useState, useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  CheckIcon,
  CopyIcon,
  GlobeIcon,
  MicIcon,
  RefreshCcwIcon,
} from "lucide-react";

// AI Element Imports
import {
  MessageAction,
  MessageActions,
  MessageBranch,
} from "@/components/ai-elements/message";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputHeader,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorEmpty,
  ModelSelectorGroup,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorLogo,
  ModelSelectorLogoGroup,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "@/components/ai-elements/model-selector";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import {
  Source,
  Sources,
  SourcesContent,
  SourcesTrigger,
} from "@/components/ai-elements/sources";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";

import { Shimmer } from "@/components/ai-elements/shimmer";

// Hooks and Utils
import { useProperChatSafe } from "@/hooks/convo_hooks";
import {
  extractTextFromUIMessage,
  extractSourcesFromUIMessage,
  hasSourcesInUIMessage,
} from "@/hooks/utils/message-converters";
import { MyUIMessage } from "@/components/schemas/ui";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useNavStore } from "../../hooks/state-hooks";
import { Dialog } from "@radix-ui/react-dialog";
import { zSourceGroupResponse } from "../../lib/api/zod.gen";
import { SourceGroupResponse } from "../../lib/api";
import axios from "axios";
import { useQuery } from "@tanstack/react-query";
const models = [
  {
    id: "gpt-4o",
    name: "GPT-4o",
    chef: "OpenAI",
    chefSlug: "openai",
    providers: ["openai", "azure"],
  },
  {
    id: "gpt-4o-mini",
    name: "GPT-4o Mini",
    chef: "OpenAI",
    chefSlug: "openai",
    providers: ["openai", "azure"],
  },
  {
    id: "claude-opus-4-20250514",
    name: "Claude 4 Opus",
    chef: "Anthropic",
    chefSlug: "anthropic",
    providers: ["anthropic", "azure", "google", "amazon-bedrock"],
  },
  {
    id: "claude-sonnet-4-20250514",
    name: "Claude 4 Sonnet",
    chef: "Anthropic",
    chefSlug: "anthropic",
    providers: ["anthropic", "azure", "google", "amazon-bedrock"],
  },
  {
    id: "gemini-2.0-flash-exp",
    name: "Gemini 2.0 Flash",
    chef: "Google",
    chefSlug: "google",
    providers: ["google"],
  },
];

const suggestions = ["System Prompt", "Parameters"];

export const EnhancedChatDisplay = ({ linkId }: { linkId: string }) => {
  const [model, setModel] = useState<string>(models[0].id);
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [useWebSearch, setUseWebSearch] = useState<boolean>(false);
  const [useMicrophone, setUseMicrophone] = useState<boolean>(false);

  const { activeConvoId } = useNavStore(
    useShallow((state) => ({
      activeConvoId: state.activeConvoId,
    })),
  );

  // Stable callback references with proper typing
  const handleChatFinish = useCallback(
    (event: {
      message: MyUIMessage;
      messages: MyUIMessage[];
      messageId: string;
      finishReason?: string;
    }) => {
      console.log("Message completed:", event);
    },
    [],
  );

  const handleChatError = useCallback((err: Error) => {
    console.error("Chat error:", err);
  }, []);

  const chatOptions = useMemo(
    () => ({
      onFinish: handleChatFinish,
      onError: handleChatError,
    }),
    [handleChatFinish, handleChatError],
  );

  //Create Chat Instance
  const { messages, sendMessage, isLoading, chat } = useProperChatSafe(
    linkId || activeConvoId,
    chatOptions,
  );

  const ollamaFetch = async () => {
    const response = await axios.get("http://localhost:11434/api/tags");
    const data = response.data;
    return data;
  };

  const modelFetch = () => {
    const { activeUserId } = useNavStore.getState();
    return useQuery({
      queryKey: ["workspaces", activeUserId],
      queryFn: ollamaFetch,
      throwOnError: true,
    });
  };

  const log_out = React.useCallback(() => {
    console.log(modelFetch);
  }, [modelFetch]);

  //
  // // Query with parameters
  // function User({ userId }) {
  //   const { data, error, isLoading } = useQuery({
  //     queryKey: ['user', userId],
  //     queryFn: () => fetchUser(userId),
  //     enabled: !!userId // Only run if userId exists
  //   })
  //
  //   return <div>{data?.name}</div>
  // }
  //
  const selectedModelData = models.find((m) => m.id === model);

  //NOTE: File bug location
  const handleSubmit = useCallback(
    async (msg: PromptInputMessage) => {
      console.log("1. Received in handleSubmit:", msg);
      if (!msg.text && !msg.files?.length) return;
      await sendMessage(msg.text || "", msg.files);
    },
    [sendMessage],
  );

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      sendMessage(suggestion);
    },
    [sendMessage],
  );

  React.useEffect(() => {
    if (messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      // console.log("Last Message State:", lastMsg.role, lastMsg.parts?.length);
    }
  }, [messages]);

  //NOTE: Message List Component
  const messageListContent = useMemo(() => {
    //Loop over Messages
    return messages.map((message: MyUIMessage, index: number) => {
      //Check if last message is assistant
      const isLastMessage = index === messages.length - 1;

      if (isLastMessage) {
        // console.log(message);
      }
      const isLastMessageAssistant =
        isLastMessage && message.role === "assistant";

      const sources = hasSourcesInUIMessage(message)
        ? extractSourcesFromUIMessage(message)
        : [];

      //Message Text
      const text = extractTextFromUIMessage(message);

      // Handle reasoning metadata
      const reasoning = message.metadata?.reasoning;

      const reasoningContent =
        reasoning && typeof reasoning === "object" && "content" in reasoning
          ? String((reasoning as { content: unknown }).content)
          : reasoning
            ? String(reasoning)
            : null;

      const reasoningDuration =
        reasoning && typeof reasoning === "object" && "duration" in reasoning
          ? Number((reasoning as { duration: unknown }).duration)
          : 0;

      return (
        <MessageBranch defaultBranch={0} key={message.id}>
          <Message role={message.role}>
            {/* Sources List */}
            <div className="flex flex-col gap-2">
              {message.role === "assistant" && sources.length > 0 && (
                <Sources>
                  <SourcesTrigger count={sources.length} />
                  <SourcesContent>
                    {sources.map((source: SourceGroupResponse, idx: number) => (
                      <Source
                        key={`${message.id}-source-${idx}`}
                        path={source.Source.Name ?? "no name provided"}
                      />
                    ))}
                  </SourcesContent>
                </Sources>
              )}
              {message.role === "assistant" && reasoningContent && (
                <Reasoning duration={reasoningDuration}>
                  <ReasoningTrigger />
                  <ReasoningContent>{reasoningContent}</ReasoningContent>
                </Reasoning>
              )}
              <MessageContent>
                {isLastMessageAssistant && text === "" ? (
                  <div className="flex items-center space-x-2">
                    <div className="flex">
                      <div className="w-1 h-1 mx-0.5 bg-gray-500 rounded-full animate-pulse [animation-delay:0ms]"></div>
                      <div className="w-1 h-1 mx-0.5 bg-gray-500 rounded-full animate-pulse [animation-delay:150ms]"></div>
                      <div className="w-1 h-1 mx-0.5 bg-gray-500 rounded-full animate-pulse [animation-delay:300ms]"></div>
                    </div>
                    <Shimmer duration={1.5} spread={1.2} className="text-sm">
                      Assistant is thinking...
                    </Shimmer>
                  </div>
                ) : (
                  <MessageResponse>{text}</MessageResponse>
                )}
              </MessageContent>
            </div>
          </Message>
          <MessageActions messageRole={message.role}>
            <MessageAction
              // TODO: Add Regen Here
              // onClick={() => regenerate()}
              label="Retry"
            >
              <RefreshCcwIcon className="size-3" />
            </MessageAction>
            <MessageAction
              // TODO: Add copy here
              // onClick={() =>
              //   navigator.clipboard.writeText(part.text)
              // }
              label="Copy"
            >
              <CopyIcon className="size-3" />
            </MessageAction>
          </MessageActions>
        </MessageBranch>
      );
    });
  }, [messages]);

  return (
    // Conversation Component
    <div className="flex size-full flex-col divide-y overflow-hidden bg-background text-foreground">
      <div className="mx-auto w-full max-w-5xl flex-1 overflow-hidden flex flex-col scrollbar-none">
        <Conversation style={{ scrollbarWidth: "none" }}>
          {/* Messages Container */}
          <ConversationContent className={"scrollbar-none"}>
            {messageListContent}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>
      </div>

      <div
        className="mx-auto w-full max-w-5xl flex-none overflow-hidden flex flex-col"
        style={{ scrollbarWidth: "none" }}
      >
        {/* Suggestions / Settings Buttons */}
        {messages.length === 0 && (
          <Suggestions className="px-4">
            {suggestions.map((suggestion) => (
              <Suggestion
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                suggestion={suggestion}
              />
            ))}
          </Suggestions>
        )}

        <div className="mx-auto w-full max-w-5xl flex-1 overflow-hidden flex flex-col p-4">
          {/* Prompt Input Area */}
          <PromptInput globalDrop multiple onSubmit={handleSubmit}>
            <PromptInputHeader>
              <PromptInputAttachments>
                {/*TODO: Stylize */}
                {(attachment) => <PromptInputAttachment data={attachment} />}
              </PromptInputAttachments>
            </PromptInputHeader>

            {/* Text Area */}
            <PromptInputBody>
              {/*NOTE: Causing renders, optimize potential */}
              <PromptInputTextarea
                value={chat.input}
                onChange={chat.handleInputChange}
                placeholder="Type your message..."
              />
            </PromptInputBody>

            {/* Prompt Footer */}
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>

                {/* Prompt Buttons */}
                <PromptInputButton
                  onClick={() => setUseMicrophone(!useMicrophone)}
                  variant={useMicrophone ? "default" : "ghost"}
                >
                  <MicIcon size={16} />
                </PromptInputButton>

                <PromptInputButton
                  onClick={() => setUseWebSearch(!useWebSearch)}
                  variant={useWebSearch ? "default" : "ghost"}
                >
                  <GlobeIcon size={16} />
                </PromptInputButton>

                {/* Model Selector Component */}
                <ModelSelector
                  onOpenChange={setModelSelectorOpen}
                  open={modelSelectorOpen}
                >
                  <ModelSelectorTrigger asChild>
                    {/* Model Selector Button */}
                    <PromptInputButton>
                      {selectedModelData?.chefSlug && (
                        <ModelSelectorLogo
                          provider={selectedModelData.chefSlug}
                        />
                      )}
                      <ModelSelectorName>
                        {selectedModelData?.name || "Select Model"}
                      </ModelSelectorName>
                    </PromptInputButton>
                  </ModelSelectorTrigger>
                  {/*TODO: Add Provider Query for list */}
                  {/* Model List  */}
                  <ModelSelectorContent>
                    <ModelSelectorInput placeholder="Search models..." />
                    <ModelSelectorList>
                      <ModelSelectorEmpty>No models found.</ModelSelectorEmpty>
                      {["OpenAI", "Anthropic", "Google"].map((chef) => (
                        <ModelSelectorGroup key={chef} heading={chef}>
                          {models
                            .filter((m) => m.chef === chef)
                            .map((m) => (
                              <ModelSelectorItem
                                key={m.id}
                                onSelect={() => {
                                  setModel(m.id);
                                  setModelSelectorOpen(false);
                                }}
                                value={m.id}
                              >
                                <ModelSelectorLogo provider={m.chefSlug} />
                                <ModelSelectorName>{m.name}</ModelSelectorName>
                                <ModelSelectorLogoGroup>
                                  {m.providers.map((p) => (
                                    <ModelSelectorLogo key={p} provider={p} />
                                  ))}
                                </ModelSelectorLogoGroup>
                                {model === m.id && (
                                  <CheckIcon className="ml-auto size-4" />
                                )}
                              </ModelSelectorItem>
                            ))}
                        </ModelSelectorGroup>
                      ))}
                    </ModelSelectorList>
                  </ModelSelectorContent>
                </ModelSelector>
                {/* Tool List */}
              </PromptInputTools>

              {/* Submit Button */}
              <PromptInputSubmit
                disabled={!chat.input.trim() || isLoading}
                status={isLoading ? "streaming" : "ready"}
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
};

export default EnhancedChatDisplay;
