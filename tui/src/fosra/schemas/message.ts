import { z } from "zod"
import { log } from "@/util/log"

const UserMessageTimeSchema = z.object({
  created: z.number(),
})

const UserMessageModelSchema = z.object({
  providerID: z.string(),
  modelID: z.string(),
})

const OutputFormatSchema = z.object({
  type: z.literal("text"),
})

const FileDiffSchema = z.object({
  file: z.string(),
  before: z.string(),
  after: z.string(),
  additions: z.number(),
  deletions: z.number(),
  status: z.enum(["added", "deleted", "modified"]).optional(),
})

const UserMessageSummarySchema = z.object({
  title: z.string().optional(),
  body: z.string().optional(),
  diffs: z.array(z.any()).default([]),
})

export const UserMessageSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  role: z.literal("user"),
  time: UserMessageTimeSchema,
  format: OutputFormatSchema.optional(),
  summary: UserMessageSummarySchema.optional(),
  agent: z.string(),
  model: UserMessageModelSchema,
  system: z.string().optional(),
  tools: z.record(z.string(), z.boolean()).optional(),
  variant: z.string().optional(),
})

const AssistantMessageTimeSchema = z.object({
  created: z.number(),
  completed: z.number().optional(),
})

const AssistantMessagePathSchema = z.object({
  cwd: z.string(),
  root: z.string(),
})

const AssistantMessageTokensCacheSchema = z.object({
  read: z.number(),
  write: z.number(),
})

const AssistantMessageTokensSchema = z.object({
  total: z.number().optional(),
  input: z.number(),
  output: z.number(),
  reasoning: z.number(),
  cache: AssistantMessageTokensCacheSchema,
})

const AssistantMessageErrorSchema = z.object({
  name: z.string(),
  data: z.record(z.any()),
})

export const AssistantMessageSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  role: z.literal("assistant"),
  time: AssistantMessageTimeSchema,
  error: AssistantMessageErrorSchema.optional(),
  parentID: z.string(),
  modelID: z.string(),
  providerID: z.string(),
  mode: z.string(),
  agent: z.string(),
  path: AssistantMessagePathSchema,
  summary: z.boolean().optional(),
  cost: z.number(),
  tokens: AssistantMessageTokensSchema,
  structured: z.any().optional(),
  variant: z.string().optional(),
  finish: z.string().optional(),
})

export const MessageSchema = z.discriminatedUnion("role", [
  UserMessageSchema,
  AssistantMessageSchema,
])

export type ValidatedUserMessage = z.infer<typeof UserMessageSchema>
export type ValidatedAssistantMessage = z.infer<typeof AssistantMessageSchema>
export type ValidatedMessage = z.infer<typeof MessageSchema>

const TextPartTimeSchema = z.object({
  start: z.number(),
  end: z.number().optional(),
})

export const TextPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("text"),
  text: z.string(),
  synthetic: z.boolean().optional(),
  ignored: z.boolean().optional(),
  time: TextPartTimeSchema.optional(),
  metadata: z.record(z.string(), z.any()).optional(),
})

const SubtaskPartModelSchema = z.object({
  providerID: z.string(),
  modelID: z.string(),
})

export const SubtaskPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("subtask"),
  prompt: z.string(),
  description: z.string(),
  agent: z.string(),
  model: SubtaskPartModelSchema.optional(),
  command: z.string().optional(),
})

const ReasoningPartTimeSchema = z.object({
  start: z.number(),
  end: z.number().optional(),
})

export const ReasoningPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("reasoning"),
  text: z.string(),
  metadata: z.record(z.string(), z.any()).optional(),
  time: ReasoningPartTimeSchema,
})

const FilePartSourceTextSchema = z.object({
  value: z.string(),
  start: z.number(),
  end: z.number(),
})

const FileSourceSchema = z.object({
  text: FilePartSourceTextSchema,
  type: z.literal("file"),
  path: z.string(),
})

export const FilePartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("file"),
  mime: z.string(),
  filename: z.string().optional(),
  url: z.string(),
  source: FileSourceSchema.optional(),
})

export const StepStartPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("step-start"),
  snapshot: z.string().optional(),
})

const StepFinishPartTokensCacheSchema = z.object({
  read: z.number(),
  write: z.number(),
})

const StepFinishPartTokensSchema = z.object({
  total: z.number().optional(),
  input: z.number(),
  output: z.number(),
  reasoning: z.number(),
  cache: StepFinishPartTokensCacheSchema,
})

export const StepFinishPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("step-finish"),
  reason: z.string(),
  snapshot: z.string().optional(),
  cost: z.number(),
  tokens: StepFinishPartTokensSchema,
})

export const SnapshotPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("snapshot"),
  snapshot: z.string(),
})

export const PatchPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("patch"),
  hash: z.string(),
  files: z.array(z.string()),
})

const AgentPartSourceSchema = z.object({
  value: z.string(),
  start: z.number(),
  end: z.number(),
})

export const AgentPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("agent"),
  name: z.string(),
  source: AgentPartSourceSchema.optional(),
})

const RetryPartTimeSchema = z.object({
  created: z.number(),
})

const RetryPartErrorSchema = z.object({
  name: z.string(),
  data: z.record(z.any()),
})

export const RetryPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("retry"),
  attempt: z.number(),
  error: RetryPartErrorSchema.optional(),
  time: RetryPartTimeSchema,
})

export const CompactionPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("compaction"),
  auto: z.boolean(),
  overflow: z.boolean().optional(),
})

const ToolStatePendingSchema = z.object({
  status: z.literal("pending"),
  input: z.record(z.string(), z.any()),
  raw: z.string(),
})

const ToolStateRunningTimeSchema = z.object({
  start: z.number(),
})

const ToolStateRunningSchema = z.object({
  status: z.literal("running"),
  input: z.record(z.string(), z.any()),
  title: z.string().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
  time: ToolStateRunningTimeSchema,
})

const ToolStateCompletedTimeSchema = z.object({
  start: z.number(),
  end: z.number(),
  compacted: z.number().optional(),
})

const ToolStateCompletedSchema = z.object({
  status: z.literal("completed"),
  input: z.record(z.string(), z.any()),
  output: z.string(),
  title: z.string(),
  metadata: z.record(z.string(), z.any()),
  time: ToolStateCompletedTimeSchema,
  attachments: z.array(z.any()).optional(),
})

const ToolStateErrorTimeSchema = z.object({
  start: z.number(),
  end: z.number(),
})

const ToolStateErrorSchema = z.object({
  status: z.literal("error"),
  input: z.record(z.string(), z.any()),
  error: z.string(),
  metadata: z.record(z.string(), z.any()).optional(),
  time: ToolStateErrorTimeSchema,
})

export const ToolStateSchema = z.discriminatedUnion("status", [
  ToolStatePendingSchema,
  ToolStateRunningSchema,
  ToolStateCompletedSchema,
  ToolStateErrorSchema,
])

export const ToolPartSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  messageID: z.string(),
  type: z.literal("tool"),
  callID: z.string(),
  tool: z.string(),
  state: ToolStateSchema,
  metadata: z.record(z.string(), z.any()).optional(),
})

export const PartSchema = z.discriminatedUnion("type", [
  TextPartSchema,
  SubtaskPartSchema,
  ReasoningPartSchema,
  FilePartSchema,
  ToolPartSchema,
  StepStartPartSchema,
  StepFinishPartSchema,
  SnapshotPartSchema,
  PatchPartSchema,
  AgentPartSchema,
  RetryPartSchema,
  CompactionPartSchema,
])

export const MessageWithPartsSchema = z.object({
  info: MessageSchema,
  parts: z.array(PartSchema).default([]),
})

export type ValidatedMessageWithParts = z.infer<typeof MessageWithPartsSchema>
export type ValidatedPart = z.infer<typeof PartSchema>

export function parseMessages(
  raw: unknown[],
): ValidatedMessageWithParts[] {
  const result: ValidatedMessageWithParts[] = []
  for (const item of raw) {
    const parsed = MessageWithPartsSchema.safeParse(item)
    if (parsed.success) {
      result.push(parsed.data)
    } else {
      log.error.warn(
        "MALFORMED_MESSAGE_DROPPED",
        { error: parsed.error.format(), item },
      )
    }
  }
  return result
}
