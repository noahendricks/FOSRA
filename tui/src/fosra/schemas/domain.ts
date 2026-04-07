import { z } from "zod";
import { PermissionRuleSchema } from "./session";

export const PathSchema = z.object({
  home: z.string(),
  state: z.string(),
  config: z.string(),
  worktree: z.string(),
  directory: z.string(),
});

export const VcsInfoSchema = z.object({
  branch: z.string(),
});

export const LspStatusSchema = z.object({
  id: z.string(),
  name: z.string(),
  root: z.string(),
  status: z.enum(["connected", "error"]),
});

export const FormatterStatusSchema = z.object({
  name: z.string(),
  extensions: z.array(z.string()),
  enabled: z.boolean(),
});

export const McpStatusConnectedSchema = z.object({
  status: z.literal("connected"),
});

export const McpStatusDisabledSchema = z.object({
  status: z.literal("disabled"),
});

export const McpStatusFailedSchema = z.object({
  status: z.literal("failed"),
  error: z.string(),
});

export const McpStatusNeedsAuthSchema = z.object({
  status: z.literal("needs_auth"),
});

export const McpStatusNeedsClientRegistrationSchema = z.object({
  status: z.literal("needs_client_registration"),
  error: z.string(),
});

export const McpStatusSchema = z.discriminatedUnion("status", [
  McpStatusConnectedSchema,
  McpStatusDisabledSchema,
  McpStatusFailedSchema,
  McpStatusNeedsAuthSchema,
  McpStatusNeedsClientRegistrationSchema,
]);

export const CommandSchema = z.object({
  name: z.string(),
  description: z.string().nullish(),
  agent: z.string().nullish(),
  model: z.string().nullish(),
  source: z.enum(["command", "mcp", "skill"]).nullish(),
  template: z.string(),
  subtask: z.boolean().nullish(),
  hints: z.array(z.string()),
});

const ModelCapabilitiesSchema = z
  .object({
    temperature: z.boolean(),
    reasoning: z.boolean(),
    attachment: z.boolean(),
    toolcall: z.boolean(),
    input: z.object({
      text: z.boolean(),
      audio: z.boolean(),
      image: z.boolean(),
      video: z.boolean(),
      pdf: z.boolean(),
    }),
    output: z.object({
      text: z.boolean(),
      audio: z.boolean(),
      image: z.boolean(),
      video: z.boolean(),
      pdf: z.boolean(),
    }),
    interleaved: z.union([
      z.boolean(),
      z.object({
        field: z.enum(["reasoning_content", "reasoning_details"]),
      }),
    ]),
  })
  .passthrough();

const ModelCostSchema = z
  .object({
    input: z.number(),
    output: z.number(),
    cache: z.object({
      read: z.number(),
      write: z.number(),
    }),
    experimentalOver200K: z
      .object({
        input: z.number(),
        output: z.number(),
        cache: z.object({
          read: z.number(),
          write: z.number(),
        }),
      })
      .nullish(),
  })
  .passthrough();

const ModelLimitSchema = z.object({
  context: z.number(),
  input: z.number().nullish(),
  output: z.number(),
});

const ModelApiSchema = z.object({
  id: z.string(),
  url: z.string(),
  npm: z.string(),
});

export const ModelSchema = z.object({
  id: z.string(),
  providerID: z.string(),
  api: ModelApiSchema,
  name: z.string(),
  family: z.string().nullish(),
  capabilities: ModelCapabilitiesSchema,
  cost: ModelCostSchema,
  limit: ModelLimitSchema,
  status: z.enum(["alpha", "beta", "deprecated", "active"]),
  options: z.record(z.string(), z.unknown()),
  headers: z.record(z.string(), z.string()),
  release_date: z.string(),
  variants: z.record(z.string(), z.record(z.string(), z.unknown())).nullish(),
});

export const ProviderSchema = z.object({
  id: z.string(),
  name: z.string(),
  source: z.enum(["env", "config", "custom", "api"]),
  env: z.array(z.string()),
  key: z.string().nullish(),
  options: z.record(z.string(), z.unknown()),
  models: z.record(z.string(), ModelSchema),
});

const AgentModelSchema = z.object({
  modelID: z.string(),
  providerID: z.string(),
});

export const AgentSchema = z.object({
  name: z.string(),
  description: z.string().nullish(),
  mode: z.enum(["subagent", "primary", "all"]),
  native: z.boolean().nullish(),
  hidden: z.boolean().nullish(),
  topP: z.number().nullish(),
  temperature: z.number().nullish(),
  color: z.string().nullish(),
  permission: z.array(PermissionRuleSchema),
  model: AgentModelSchema.nullish(),
  variant: z.string().nullish(),
  prompt: z.string().nullish(),
  options: z.record(z.string(), z.unknown()),
  steps: z.array(z.unknown()).nullish(),
});

export const ConfigSchema = z.object({}).passthrough();

export const WorkspaceSchema = z.object({
  id: z.string(),
  type: z.string(),
  branch: z.string().nullish(),
  name: z.string().nullish(),
  directory: z.string().nullish(),
  extra: z.unknown().nullish(),
  projectID: z.string(),
});

export const ProvidersResponseSchema = z.object({
  providers: z.array(ProviderSchema).nullish(),
  default: z.record(z.string(), z.unknown()).nullish(),
  connected: z.array(z.string()).nullish(),
});
