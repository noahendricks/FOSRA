import { z } from "zod";

export const TodoSchema = z.object({
  content: z.string(),
  status: z.string(),
  priority: z.string(),
});

export const FileDiffSchema = z.object({
  file: z.string(),
  before: z.string(),
  after: z.string(),
  additions: z.number(),
  deletions: z.number(),
  status: z.enum(["added", "deleted", "modified"]).optional(),
});

export const PermissionRuleSchema = z.object({
  permission: z.string(),
  pattern: z.string(),
  action: z.enum(["allow", "deny", "ask"]),
});

export const PermissionRulesetSchema = z.array(PermissionRuleSchema);

export const SessionStatusIdleSchema = z.object({
  type: z.literal("idle"),
});

export const SessionStatusRetrySchema = z.object({
  type: z.literal("retry"),
  attempt: z.number(),
  message: z.string(),
  next: z.number(),
});

export const SessionStatusBusySchema = z.object({
  type: z.literal("busy"),
});

export const SessionStatusSchema = z.discriminatedUnion("type", [
  SessionStatusIdleSchema,
  SessionStatusRetrySchema,
  SessionStatusBusySchema,
]);

const SessionTimeSchema = z.object({
  created: z.number(),
  updated: z.number(),
  compacting: z.number().nullish(),
  archived: z.number().nullish(),
});

const SessionRevertSchema = z
  .object({
    messageID: z.string(),
    partID: z.string().nullish(),
    snapshot: z.string().nullish(),
    diff: z.string().nullish(),
  })
  .passthrough();

const SessionMetadataModelCostSchema = z
  .object({
    input: z.number(),
    output: z.number(),
    cache: z
      .object({
        read: z.number(),
        write: z.number(),
      })
      .nullish(),
  })
  .passthrough();

const SessionMetadataModelLimitSchema = z
  .object({
    context: z.number(),
    input: z.number().nullish(),
    output: z.number().nullish(),
  })
  .passthrough();

const SessionMetadataModelSchema = z
  .object({
    providerID: z.string(),
    modelID: z.string(),
    cost: SessionMetadataModelCostSchema.nullish(),
    limit: SessionMetadataModelLimitSchema.nullish(),
  })
  .passthrough();

const SessionMetadataSchema = z
  .object({
    agent: z.string().nullish(),
    model: SessionMetadataModelSchema.nullish(),
  })
  .passthrough();

export const SessionSchema = z.object({
  id: z.string(),
  directory: z.string(),
  parentID: z.string().nullish(),
  title: z.string(),
  version: z.string(),
  time: SessionTimeSchema,
  permission: PermissionRulesetSchema.nullish(),
  revert: SessionRevertSchema.nullish(),
  metadata: SessionMetadataSchema.nullish(),
});

export const SessionErrorSchema = z.object({
  name: z.string(),
  data: z.record(z.string(), z.unknown()),
});

export const EventSessionInfoSchema = z.object({
  info: SessionSchema,
});

export const EventSessionStatusSchema = z.object({
  sessionID: z.string(),
  status: SessionStatusSchema,
});

export const EventSessionErrorSchema = z.object({
  sessionID: z.string().nullish(),
  error: SessionErrorSchema.nullish(),
});

export const EventSessionIdleSchema = z.object({
  sessionID: z.string(),
});

export const EventSessionCompactedSchema = z.object({
  sessionID: z.string(),
});

export const EventSessionDiffSchema = z.object({
  sessionID: z.string(),
  diff: z.array(FileDiffSchema),
});

export const EventTodoUpdatedSchema = z.object({
  sessionID: z.string(),
  todos: z.array(TodoSchema),
});
