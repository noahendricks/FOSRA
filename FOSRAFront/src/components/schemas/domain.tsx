import * as z from "zod";
import { zMessageResponse, zSourceGroupResponse } from "../../lib/api/zod.gen";

export enum MessageRole {
  USER = "user",
  ASSISTANT = "assistant",
  SYSTEM = "system",
}
export const MessageRoleSchema = z.union([
  z.literal(MessageRole.USER),
  z.literal(MessageRole.ASSISTANT),
  z.literal(MessageRole.SYSTEM),
]);

export type MessageRoleType = z.infer<typeof MessageRole>;

const OriginType = z.enum([
  "CRAWLED_URL",
  "FILESYSTEM",
  "S3_DOCUMENT",
  "YOUTUBE_VIDEO",
  "SLACK_MESSAGE",
  "NOTION_PAGE",
  "GITHUB_CONTENT",
  "LINEAR_ISSUE",
  "JIRA_ISSUE",
  "DISCORD_MESSAGE",
  "GMAIL_MESSAGE",
  "LUMA_EVENT",
  "ELASTICSEARCH_DOCUMENT",
  "EXTENSION_CONTENT",
  "TAVILY_RESULT",
  "SEARXNG_RESULT",
  "BAIDU_RESULT",
  "LINKUP_RESULT",
  "EXA_RESULT",
  "SERPER_RESULT",
]);

// -------------------------------

// # ============================================================================
// # User
// # ============================================================================
//
export const UserCreateRequest = z.object({
  Name: z.string(),
  Enabled: z.boolean().default(true),
});

export type UserCreateRequestType = z.infer<typeof UserCreateRequest>;

export const UserUpdateRequest = z.object({
  Name: z.string(),
  Enabled: z.boolean().default(true),
});

export type UserUpdateRequestType = z.infer<typeof UserUpdateRequest>;

export const UserFromBackend = z.object({
  Name: z.string(),
  Enabled: z.boolean().default(true),
  UserId: z.string(),
  CreatedAt: z.iso.datetime({ offset: true }),
  LastLogin: z.iso.datetime({ offset: true }),
});

export type UserFromBackendType = z.infer<typeof UserFromBackend>;

export const UserDisplay = z.object({
  Name: z.string(),
  Enabled: z.boolean().default(true),
  UserId: z.string(),
  CreatedAt: z.iso.datetime({ offset: true }),
  LastLogin: z.iso.datetime({ offset: true }),
});

export type UserDisplayType = z.infer<typeof UserDisplay>;

// # ============================================================================
// # Workspace
// # ============================================================================
export const NewWorkspaceRequest = z.object({
  Name: z.string().default("New Workspace UI"),
  UserId: z.string(),
  Description: z.string().default("New Place For WORK"),
});

export type NewWorkspaceRequestType = z.infer<typeof NewWorkspaceRequest>;

export const UpdateWorkspaceRequest = z.object({
  Name: z.string().default("New Workspace UI"),
  UserId: z.string(),
  Description: z.string().default("New Place For WORK"),
});

export type UpdateWorkspaceRequestType = z.infer<typeof UpdateWorkspaceRequest>;

export const WorkspaceFromBackend = z.object({
  Name: z.string().default("New Workspace UI"),
  UserId: z.string().nullable(),
  WorkspaceId: z.string(),
  SourcesCount: z.int().default(0),
  ConversationsCount: z.int().default(0),
  Description: z.string().nullable(),
  ArchivedConvos: z.array(z.string()).nullable().default([""]),
});

export type WorkspaceFromBackendType = z.infer<typeof WorkspaceFromBackend>;

export const WorkspaceDisplay = z.object({
  Name: z.string().default("New Workspace UI"),
  UserId: z.string(),
  SourcesCount: z.int().default(0),
  ConversationsCount: z.int().default(0),
  Description: z.string(),
  ArchivedConvos: z.array(z.string()).nullable(),
});

export type WorkspaceDisplayType = z.infer<typeof WorkspaceDisplay>;

// # ============================================================================
// # Chunk
// # ============================================================================

export const ChunkFromBackend = z.object({
  ChunkId: z.string(),
  SourceId: z.string(),
  SourceHash: z.string(),
  StartIndex: z.int(),
  EndIndex: z.int(),
  TokenCount: z.int(),
  Text: z.string(),
});

export type ChunkFromBackendType = z.infer<typeof ChunkFromBackend>;

export const ChunkDisplay = z.object({
  ChunkId: z.string(),
  SourceId: z.string(),
  SourceHash: z.string(),
  StartIndex: z.int(),
  EndIndex: z.int(),
  TokenCount: z.int(),
  Text: z.string(),
});

export type ChunkDisplayType = z.infer<typeof ChunkDisplay>;

// # ============================================================================
// # Source
// # ============================================================================
export const SourceRequest = z.object({
  SourceId: z.string(),
});

export const SourceFromBackendDeep = z.object({
  SourceId: z.string(),
  Hash: z.string().nullable(),
  Name: z.string().default(""),
  OriginPath: z.string().nullable(),
  SourceSummary: z.string().default(""),
  OriginType: OriginType,
  SummaryEmbedding: z.string().default(""),
  UploadedAt: z.iso.datetime({ offset: true }),
  ModifiedTime: z.iso.datetime({ offset: true }),
  Chunks: z.array(ChunkFromBackend).default([]),
  Content: z.string().nullable(),
  Metadata: z.json(),
});

export type SourceFromBackendDeepType = z.infer<typeof SourceFromBackendDeep>;

export const SourceFromBackendShallow = z.object({
  SourceId: z.string(),
  Hash: z.string().nullable(),
  Name: z.string().default(""),
  OriginPath: z.string().nullable(),
  SourceSummary: z.string().default(""),
  OriginType: OriginType,
  SummaryEmbedding: z.string().default(""),
  UploadedAt: z.iso.datetime({ offset: true }),
  ModifiedTime: z.iso.datetime({ offset: true }),
});

export type SourceFromBackendShallowType = z.infer<
  typeof SourceFromBackendShallow
>;

export const SourceDisplayDeep = z.object({
  SourceId: z.string(),
  Hash: z.string().nullable(),
  Name: z.string().default(""),
  OriginPath: z.string().nullable(),
  SourceSummary: z.string().default(""),
  OriginType: OriginType,
  UploadedAt: z.iso.datetime({ offset: true }),
  ModifiedTime: z.iso.datetime({ offset: true }),
  Chunks: z.array(ChunkFromBackend).default([]),
  Content: z.string().nullable(),
  Metadata: z.json(),
});

export type SourceDisplayDeep = z.infer<typeof SourceDisplayDeep>;

export const SourceDisplayShallow = z.object({
  SourceId: z.string(),
  Hash: z.string().nullable(),
  Name: z.string().default(""),
  OriginPath: z.string().nullable(),
  SourceSummary: z.string().default(""),
  OriginType: OriginType,
  UploadedAt: z.iso.datetime({ offset: true }),
  ModifiedTime: z.iso.datetime({ offset: true }),
});
export type SourceDisplayShallowType = z.infer<typeof SourceDisplayShallow>;

// # ============================================================================
// # Retrieved Result
// # ============================================================================

export const RetrievedResultFromBackend = z.object({
  Chunk: z.array(ChunkFromBackend).default([]),
  SourceId: z.string(),
  SourceName: z.string(),
  SourceSummary: z.string(),
  ChunkSnippet: z.string().nullable(),
  SimilarityScore: z.float32(),
  ResultRank: z.int(),
  RerankerScore: z.float32(),
  RetrievedAt: z.iso.datetime({ offset: true }),
  Query: z.string(),
});

export type RetrievedResultFromBackendType = z.infer<
  typeof RetrievedResultDisplay
>;
//
export const RetrievedResultDisplay = z.object({
  Chunk: z.array(ChunkFromBackend).default([]),
  SourceId: z.string(),
  SourceSummary: z.string(),
  ChunkSnippet: z.string().nullable(),
  SimilarityScore: z.float32(),
  ResultRank: z.int(),
  RerankerScore: z.float32(),
  RetrievedAt: z.iso.datetime({ offset: true }),
  Query: z.string(),
});

export type RetrievedResultDisplayType = z.infer<typeof RetrievedResultDisplay>;
// # ============================================================================
// # File
// # ============================================================================
export const FileRequest = z.object({
  OriginPath: z.string(),
});

export type FileRequest = z.infer<typeof FileRequest>;

// export type FileType = z.infer<typeof File>;
export const FileFromBackend = z.object({
  OriginPath: z.string(),
  Url: z.string(),
  UploadedAt: z.iso.datetime({ offset: true }),
  LastModified: z.iso.datetime({ offset: true }),
  Name: z.string().default(""),
  FileType: z.union([z.literal("FILE"), z.literal("IMAGE")]),
  Metadata: z.record(z.string(), z.unknown()).nullable(),
});

export type FileFromBackendType = z.infer<typeof FileFromBackend>;

export const FileDisplay = z.object({
  Id: z.nanoid(),
  FileType: z.string(),
  OriginPath: z.string(),
  UploadedAt: z.iso.datetime({ offset: true }),
  Name: z.string().default(""),
  Metadata: z.record(z.string(), z.unknown()).nullable(),
});
export type FileDisplayType = z.infer<typeof FileDisplay>;

// # ============================================================================
// # Prompt
// # ============================================================================
export const PromptInputSubmitRequest = z.object({
  Text: z.string(),
  Files: z.array(FileRequest),
});

export type PromptInputSubmitRequestType = z.infer<
  typeof PromptInputSubmitRequest
>;

// # ============================================================================
// # Message
// # ============================================================================

export const NewUserMessageRequest = z.object({
  Role: z.string(),
  ConvoId: z.string(),
  Text: z.string(),
  UserId: z.string().nullable(),
  MessageId: z.string() || z.nanoid(),
  Metadata: null,
  AttachedFiles: z.array(z.string()).default([]).nullable(),
});

export type NewMessageRequestType = z.infer<typeof NewMessageRequest>;

export const UpdateMessageRequest = z.object({
  Role: MessageRoleSchema,
  ConvoId: z.string(),
  UserId: z.string().nullable(),
  Text: z.string(),
  MessageId: z.string() || z.nanoid(),
  AttachedFiles: z.array(FileRequest).default([]),
});

export type UpdateMessageRequestType = z.infer<typeof UpdateMessageRequest>;

export const MessageFromBackend = z.object({
  Role: MessageRoleSchema,
  ConvoId: z.string(),
  UserId: z.string().nullable(),
  Text: z.string(),
  MessageId: z.string() || z.nanoid(),
  SourcesCount: z.int().default(0).nullable(),
  Timestamp: z.coerce.date().nullable(),
  AttachedSources: z.array(zSourceGroupResponse),
});

export type MessageFromBackendType = z.infer<typeof MessageFromBackend>;

export const MessageDisplay = z.object({
  MsgId: z.nanoid().nullable(),
  UserId: z.string().nullable(),
  Role: MessageRoleSchema,
  Text: z.string(),
});

export type MessageDisplayType = z.infer<typeof MessageDisplay>;

// # ============================================================================
// # Convo
// # ============================================================================

export const ConvoFromBackend = z.object({
  UserId: z.string(),
  WorkspaceId: z.string(),
  ConvoId: z.string(),
  Title: z.string().nullable().default("New Convo"),
  CreatedAt: z.iso.datetime({ offset: true }),
  UpdatedAt: z.iso.datetime({ offset: true }),
  MessageCount: z.number().int().default(0),
  Messages: z.array(zMessageResponse),
  KnowledgeSources: z
    .array(SourceDisplayDeep || SourceDisplayShallow)
    .default([]),
});

export type ConvoFromBackendType = z.infer<typeof ConvoFromBackend>;

export const ConvoListItemDisplay = z.object({
  UserId: z.string(),
  WorkspaceId: z.string(),
  ConvoId: z.string(),
  Title: z.string().nullable().default("New Convo"),
  CreatedAt: z.iso.datetime({ offset: true }),
  UpdatedAt: z.iso.datetime({ offset: true }),
  MessageCount: z.number().int().default(0),
});

export type ConvoListItemDisplayType = z.infer<typeof ConvoListItemDisplay>;
