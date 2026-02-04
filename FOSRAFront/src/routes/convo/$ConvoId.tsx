import { createFileRoute, redirect, useParams } from "@tanstack/react-router";
import { fetchConvo } from "../../hooks/convo_hooks";
import z from "zod";
import { queryOptions } from "@tanstack/react-query";
import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { useCallback } from "react";
import EnhancedChatDisplay from "../../components/fosra-ui/convo-display";
import {
  PersistedClient,
  Persister,
} from "@tanstack/react-query-persist-client";
import { useNavStore } from "../../hooks/state-hooks";

/**
 * TanStack Router route for conversation view with enhanced chat.
 * Prefetches conversation data and integrates with AI SDK streaming.
 */

export const Route = createFileRoute("/convo/$ConvoId")({
  params: {
    parse: (params) => ({
      ConvoId: z.string().parse(params.ConvoId),
    }),
    stringify: ({ ConvoId }) => ({
      ConvoId,
    }),
  },

  // Prefetch conversation data before rendering
  loader: async (opts) => {
    await opts.context.queryClient.prefetchQuery(
      queryOptions({
        queryKey: ["convos", opts.params.ConvoId],
        queryFn: () => fetchConvo(opts.params.ConvoId),
        staleTime: 1000 * 60 * 5, // 5 minutes
        refetchOnMount: true,
      }),
    );

    return null;
  },

  beforeLoad: async ({ location }) => {
    const { activeUserId } = useNavStore.getState();

    if (!activeUserId) {
      throw redirect({
        to: "/login",
        search: {
          redirect: location.href,
        },
      });
    }
  },

  shouldReload: true,
  preload: true,
  // Show loading state while prefetching
  pendingComponent: () => (
    <div className="flex items-center justify-center h-full ">
      <div className="text-muted-foreground">Loading conversation...</div>
    </div>
  ),

  // Error boundary
  errorComponent: ({ error }) => (
    <div className="flex items-center justify-center h-full">
      <div className="text-destructive">
        <h2 className="text-lg font-semibold mb-2">
          Error loading conversation
        </h2>
        <p className="text-sm">{error.message}</p>
      </div>
    </div>
  ),

  component: ConvoMessagesDisplay,
});

function ConvoMessagesDisplay() {
  const params = Route.useParams();

  let linkId = params.ConvoId;

  return <EnhancedChatDisplay linkId={linkId} />;
}
