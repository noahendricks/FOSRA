import { z } from "zod";

export const QuestionOptionSchema = z.object({
  label: z.string(),
  description: z.string(),
});

export const QuestionInfoSchema = z.object({
  question: z.string(),
  header: z.string(),
  options: z.array(QuestionOptionSchema),
  multiple: z.boolean().optional(),
  custom: z.boolean().optional(),
});

export const PermissionRequestSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  permission: z.string(),
  patterns: z.array(z.string()),
  metadata: z.record(z.string(), z.unknown()),
  always: z.array(z.string()),
  tool: z
    .object({
      messageID: z.string(),
      callID: z.string(),
    })
    .nullable(),
});

export const EventPermissionRepliedSchema = z.object({
  sessionID: z.string(),
  requestID: z.string(),
  reply: z.enum(["once", "always", "reject"]),
});

export const QuestionRequestSchema = z.object({
  id: z.string(),
  sessionID: z.string(),
  questions: z.array(QuestionInfoSchema),
  tool: z
    .object({
      messageID: z.string(),
      callID: z.string(),
    })
    .nullable(),
});

export const EventQuestionRepliedSchema = z.object({
  sessionID: z.string(),
  requestID: z.string(),
  answers: z.array(z.array(z.string())),
});

export const EventQuestionRejectedSchema = z.object({
  sessionID: z.string(),
  requestID: z.string(),
});

export const EventInstallationUpdateAvailableSchema = z.object({
  currentVersion: z.string(),
  latestVersion: z.string(),
});

export const EventTuiToastShowSchema = z.object({
  title: z.string().optional(),
  message: z.string(),
  variant: z.enum(["info", "success", "warning", "error"]),
  duration: z.number().optional(),
});

export const EventTuiCommandExecuteSchema = z.object({
  command: z.string(),
});

export const EventTuiPromptAppendSchema = z.object({
  text: z.string(),
});

export const EventTuiSessionSelectSchema = z.object({
  sessionID: z.string(),
});

export const EventServerInstanceDisposedSchema = z.object({
  directory: z.string(),
});

export const EventVcsBranchUpdatedSchema = z.object({
  branch: z.string().nullable(),
});

export const EventMcpBrowserOpenFailedSchema = z.object({
  mcpName: z.string(),
  url: z.string(),
});

export const EventCommandExecutedSchema = z.object({
  name: z.string(),
  sessionID: z.string(),
  arguments: z.record(z.string(), z.unknown()),
  messageID: z.string(),
});

export const EventMessageRemovedSchema = z.object({
  sessionID: z.string(),
  messageID: z.string(),
});

export const EventMessagePartRemovedSchema = z.object({
  sessionID: z.string(),
  messageID: z.string(),
  partID: z.string(),
});
