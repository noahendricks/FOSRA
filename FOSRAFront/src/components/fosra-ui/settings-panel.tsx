import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ComponentProps, JSX, ReactNode, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Field } from "../ui/field";
import { Textarea } from "../ui/textarea";

export type SettingsPanelProps = ComponentProps<typeof Dialog> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export type SettingsPannelTriggerProps = ComponentProps<typeof DialogTrigger>;

export const SettingsPanelTrigger = (props: SettingsPannelTriggerProps) => (
  <DialogTrigger {...props} />
);

export type SettingsPanelContentProps = ComponentProps<typeof DialogContent> & {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

const UserSettings = {
  llmDefault: {
    configId: 1,
    configName: "Default GPT-4",
    provider: "openrouter",
    customProvider: null,
    model: "openai/gpt-4-turbo",
    apiKey: "sk-or-v1-abc123def456",
    apiBase: "https://openrouter.ai/api/v1",
    language: "English",
    litellmParams: {
      temperature: 0.7,
      maxTokens: 4096,
    },
  },
  llmFast: {
    configId: 2,
    configName: "Fast GPT-3.5",
    provider: "openrouter",
    customProvider: null,
    model: "openai/gpt-3.5-turbo",
    apiKey: "sk-or-v1-abc123def456",
    apiBase: "https://openrouter.ai/api/v1",
    language: "English",
    litellmParams: {
      temperature: 0.5,
      maxTokens: 2048,
    },
  },
  llmLogic: {
    configId: 3,
    configName: "Claude Sonnet Strategic",
    provider: "anthropic",
    customProvider: null,
    model: "claude-3-sonnet-20240229",
    apiKey: "sk-ant-xyz789",
    apiBase: "https://api.anthropic.com",
    language: "English",
    litellmParams: {
      temperature: 0.3,
      maxTokens: 8192,
    },
  },
  llmHeavy: {
    configId: 4,
    configName: "Claude Opus Heavy",
    provider: "anthropic",
    customProvider: null,
    model: "claude-3-opus-20240229",
    apiKey: "sk-ant-xyz789",
    apiBase: "https://api.anthropic.com",
    language: "English",
    litellmParams: {
      temperature: 0.7,
      maxTokens: 16384,
    },
  },
  parser: {
    configId: 1,
    configName: "Default Docling Parser",
    parserType: "DOCLING",
    apiKey: null,
    apiBase: null,
    maxPages: null,
    extractTables: true,
    extractImages: false,
    ocrEnabled: true,
    language: "eng",
    timeoutSeconds: 300,
    fallbackParsers: ["UNSTRUCTURED", "PYPDF"],
    generateSummary: true,
  },
  vectorStore: {
    configId: 1,
    configName: "Default Qdrant",
    apiKey: null,
    apiBase: null,
    collectionName: "documents",
    includeMetadata: true,
    metadata: {},
    storeType: "QDRANT",
    host: "localhost",
    port: 6333,
    location: ":memory:",
    url: null,
    useMemory: true,
    topK: 10,
    minScore: 0.0,
    includeVectors: false,
    filterConditions: {},
  },
  embedder: {
    configId: 1,
    configName: "Default FastEmbed",
    model: "BAAI/bge-small-en-v1.5",
    apiKey: null,
    apiBase: null,
    mode: "DENSE_ONLY",
    batchSize: 32,
    maxConcurrent: 3,
    normalize: true,
    truncate: true,
    maxLength: 512,
    denseModel: null,
    sparseModel: null,
    lateModel: null,
    embedderType: "FASTEMBED",
  },
  reranker: {
    configId: "1",
    configName: "Cohere Reranker",
    rerankerType: "COHERE",
    model: "rerank-english-v3.0",
    apiKey: "co-test-key-123",
    enabled: false,
    params: null,
    topK: 10,
    scoreThreshold: null,
    returnScores: true,
    batchSize: 32,
  },
  storage: {
    configId: 1,
    configName: "Local Filesystem",
    storageType: "FILESYSTEM",
    basePath: "/data/documents",
    apiKey: null,
    bucket: null,
    region: null,
    endpoint: null,
  },
  chunker: {
    chunkSize: 512,
    chunkOverlap: 128,
    minChunkSize: 100,
    maxChunkSize: 2000,
    similarityThreshold: 0.8,
    embeddingModel: "all-MiniLM-L6-v2",
    sentencesPerChunk: 5,
  },
};

// Retrieve User Settings
//// Place in Zustand

// Map Over Types of Setting (LLM, Retriever, Vector, etc) & Set Tab Component
////  Map each Field in the Config & Set Field Name and Type of Input (Radio, Field, Toggle)
///// Fill each Field with Existing Value
//

// for (const [title, field_val] of Object.entries(UserSettings)) {
//   console.log(title, `\n`);
//   for (const [field, val] of Object.entries(field_val)) {
//     console.log(field, val, `  |||${typeof val}`);
//   }
//   console.log(`\n`);
// }

const triggers: JSX.Element[] = [];
const contents: JSX.Element[] = [];

type FieldValue = string | number | boolean | object | null;

const fieldGenerator = (field: string, val: FieldValue): JSX.Element => {
  switch (typeof val) {
    case "string":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Input id={field} defaultValue={val} />
        </div>
      );

    case "boolean":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Switch id={field} checked={val} />
        </div>
      );
    case "number":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Input type="number" id={field} defaultValue={val} />
        </div>
      );

    case "bigint":
      return (
        <div className="space-y-2">
          <Label htmlFor={field}>{field}</Label>
          <Field id={field} defaultValue={"BIGINT NOT SUPPORTED"} />
        </div>
      );
    case "object":
      if (val && !Array.isArray(val)) {
        return (
          <div>
            <Label>{field}</Label>
            <Textarea defaultValue={JSON.stringify(val, null, 2)} />
          </div>
        );
      } else {
        return <></>;
      }
  }
};

export const SettingsPanel = ({ open, onOpenChange }: SettingsPanelProps) => {
  // const settings = useSettingsStore((state) => state.activeUserSettings);
  const settings = UserSettings;

  const { triggers, contents } = useMemo(() => {
    const triggers: JSX.Element[] = [];
    const contents: JSX.Element[] = [];

    for (const [title, field_val] of Object.entries(settings)) {
      triggers.push(
        <TabsTrigger key={title} value={title}>
          {title}
        </TabsTrigger>,
      );

      const tabFields: JSX.Element[] = [];

      for (const [field, val] of Object.entries(field_val)) {
        tabFields.push(fieldGenerator(field, val));
      }
      contents.push(<TabsContent value={title}>{tabFields}</TabsContent>);
    }

    return { triggers, contents };
  }, [settings]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="default" size="default">
          Settings
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-21">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="account" className="w-full">
          <TabsList>{triggers}</TabsList>
          {contents}
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Cancel
          </Button>
          <Button variant="outline" onClick={() => onOpenChange?.(false)}>
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
