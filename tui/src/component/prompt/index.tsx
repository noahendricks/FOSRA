import {
  BoxRenderable,
  TextareaRenderable,
  MouseEvent,
  PasteEvent,
  t,
  dim,
  fg,
} from "@opentui/core";
import {
  createEffect,
  createMemo,
  type JSX,
  onMount,
  createSignal,
  onCleanup,
  on,
  Show,
  Switch,
  Match,
} from "solid-js";
import "opentui-spinner/solid";
import path from "path";
import { log } from "@/util/log";
import { Filesystem } from "@/util/filesystem";
import { useLocal } from "@tui/context/local";
import { useTheme } from "@tui/context/theme";
import { EmptyBorder } from "@tui/component/border";
import { useApi } from "../../context/api";
import { useRoute } from "@tui/context/route";
import { useStore } from "../../context/store";
import { createStore, produce } from "solid-js/store";
import { useKeybind } from "@tui/context/keybind";
import { usePromptHistory, type PromptInfo } from "./history";
import { usePromptStash } from "./stash";
import { DialogStash } from "../dialog-stash";
import { type AutocompleteRef, Autocomplete } from "./autocomplete";
import { useCommandDialog } from "../dialog-command";
import { useRenderer } from "@opentui/solid";
import { Editor } from "@tui/util/editor";
import { useExit } from "../../context/exit";
import { Clipboard } from "../../util/clipboard";
import type { FilePart } from "@fosra/api/v2";
import { TuiEvent } from "../../event";
import { iife } from "@/util/iife";
import { Locale } from "@/util/locale";
import { formatDuration } from "@/util/format";
import { createColors, createFrames } from "../../ui/spinner.ts";
import { useDialog } from "@tui/ui/dialog";
import { DialogProvider as DialogProviderConnect } from "../dialog-provider";
import { DialogAlert } from "../../ui/dialog-alert";
import { useToast } from "../../ui/toast";
import { useKV } from "../../context/kv";
import { useTextareaKeybindings } from "../textarea-keybindings";
import { DialogSkill } from "../dialog-skill";

export type PromptProps = {
  sessionID?: string;
  workspaceID?: string;
  visible?: boolean;
  disabled?: boolean;
  onSubmit?: () => void;
  ref?: (ref: PromptRef) => void;
  hint?: JSX.Element;
  showPlaceholder?: boolean;
};

export type PromptRef = {
  focused: boolean;
  current: PromptInfo;
  set(prompt: PromptInfo): void;
  reset(): void;
  blur(): void;
  focus(): void;
  submit(): void;
};

const PLACEHOLDERS = [
  "Fix a TODO in the codebase",
  "What is the tech stack of this project?",
  "Fix broken tests",
];
const SHELL_PLACEHOLDERS = ["ls -la", "git status", "pwd"];

export function Prompt(props: PromptProps) {
  let input: TextareaRenderable;
  let anchor: BoxRenderable;
  let autocomplete: AutocompleteRef;

  const keybind = useKeybind();
  const local = useLocal();
  const api = useApi();
  const route = useRoute();
  const store = useStore();
  const dialog = useDialog();
  const toast = useToast();
  const status = createMemo(() => {
    const sessionID = props.sessionID;
    if (!sessionID) return { type: "idle" };
    const sessionStatus = store.session.status(sessionID);
    return { type: sessionStatus };
  });
  const history = usePromptHistory();
  const stash = usePromptStash();
  const command = useCommandDialog();
  const renderer = useRenderer();
  const { theme, syntax } = useTheme();
  const kv = useKV();

  function promptModelWarning() {
    toast.show({
      variant: "warning",
      message: "Connect a provider to send prompts",
      duration: 3000,
    });
    if (store.state.providers().length === 0) {
      dialog.replace(() => <DialogProviderConnect />);
    }
  }

  const textareaKeybindings = useTextareaKeybindings();

  const fileStyleId = syntax().getStyleId("extmark.file")!;
  const agentStyleId = syntax().getStyleId("extmark.agent")!;
  const pasteStyleId = syntax().getStyleId("extmark.paste")!;
  let promptPartTypeId = 0;

  createEffect(() => {
    if (props.disabled) input.cursorColor = theme.backgroundElement;
    if (!props.disabled) input.cursorColor = theme.text;
  });

  const lastUserMessage = createMemo(() => {
    if (!props.sessionID) return undefined;
    const messages = store.state.messages.get(props.sessionID) ?? [];
    if (!messages) return undefined;
    return messages.findLast((m) => m.role === "user");
  });

  const [promptUIStore, setPromptUIStore] = createStore<{
    prompt: PromptInfo;
    mode: "normal" | "shell";
    extmarkToPartIndex: Map<number, number>;
    interrupt: number;
    placeholder: number;
  }>({
    placeholder: Math.floor(Math.random() * PLACEHOLDERS.length),
    prompt: {
      input: "",
      parts: [],
    },
    mode: "normal",
    extmarkToPartIndex: new Map(),
    interrupt: 0,
  });

  // reset interrupt counter when session goes idle
  createEffect(() => {
    if (status().type === "idle") {
      setPromptUIStore("interrupt", 0);
    }
  });

  createEffect(
    on(
      () => props.sessionID,
      () => {
        setPromptUIStore(
          "placeholder",
          Math.floor(Math.random() * PLACEHOLDERS.length),
        );
      },
      { defer: true },
    ),
  );

  // Initialize agent/model/variant from last user message when session changes
  let syncedSessionID: string | undefined;
  createEffect(() => {
    const sessionID = props.sessionID;
    const msg = lastUserMessage();

    if (sessionID !== syncedSessionID) {
      if (!sessionID || !msg) return;

      syncedSessionID = sessionID;

      // Only set agent if it's a primary agent (not a subagent)
      const isPrimaryAgent = local.agent
        .list()
        .some((x) => x.name === msg.agent);
      if (msg.agent && isPrimaryAgent) {
        local.agent.set(msg.agent);
        if (msg.model) local.model.set(msg.model);
        if (msg.variant) local.model.variant.set(msg.variant);
      }
    }
  });

  command.register(() => {
    return [
      {
        title: "Clear prompt",
        value: "prompt.clear",
        category: "Prompt",
        hidden: true,
        onSelect: (dialog) => {
          input.extmarks.clear();
          input.clear();
          dialog.clear();
        },
      },
      {
        title: "Submit prompt",
        value: "prompt.submit",
        keybind: "input_submit",
        category: "Prompt",
        hidden: true,
        onSelect: (dialog) => {
          if (!input.focused) return;
          submit();
          dialog.clear();
        },
      },
      {
        title: "Paste",
        value: "prompt.paste",
        keybind: "input_paste",
        category: "Prompt",
        hidden: true,
        onSelect: async () => {
          const content = await Clipboard.read();
          if (content?.mime.startsWith("image/")) {
            await pasteImage({
              filename: "clipboard",
              mime: content.mime,
              content: content.data,
            });
          }
        },
      },
      {
        title: "Interrupt session",
        value: "session.interrupt",
        keybind: "session_interrupt",
        category: "Session",
        hidden: true,
        enabled: status().type !== "idle",
        onSelect: (dialog) => {
          if (autocomplete.visible) return;
          if (!input.focused) return;
          // TODO: this should be its own command
          if (promptUIStore.mode === "shell") {
            setPromptUIStore("mode", "normal");
            return;
          }
          if (!props.sessionID) return;

          setPromptUIStore("interrupt", promptUIStore.interrupt + 1);

          setTimeout(() => {
            setPromptUIStore("interrupt", 0);
          }, 5000);

          if (promptUIStore.interrupt >= 2) {
            api.fosra.session.abort({
              sessionID: props.sessionID,
            });
            setPromptUIStore("interrupt", 0);
          }
          dialog.clear();
        },
      },
      {
        title: "Open editor",
        category: "Session",
        keybind: "editor_open",
        value: "prompt.editor",
        slash: {
          name: "editor",
        },
        onSelect: async (dialog) => {
          dialog.clear();

          // replace summarized text parts with the actual text
          const text = promptUIStore.prompt.parts
            .filter((p) => p.type === "text")
            .reduce((acc, p) => {
              if (!p.source) return acc;
              return acc.replace(p.source.text.value, p.text);
            }, promptUIStore.prompt.input);

          const nonTextParts = promptUIStore.prompt.parts.filter(
            (p) => p.type !== "text",
          );

          const value = text;
          const content = await Editor.open({ value, renderer });
          if (!content) return;

          input.setText(content);

          // Update positions for nonTextParts based on their location in new content
          // Filter out parts whose virtual text was deleted
          // this handles a case where the user edits the text in the editor
          // such that the virtual text moves around or is deleted
          const updatedNonTextParts = nonTextParts
            .map((part) => {
              let virtualText = "";
              if (part.type === "file" && part.source?.text) {
                virtualText = part.source.text.value;
              } else if (part.type === "agent" && part.source) {
                virtualText = part.source.value;
              }

              if (!virtualText) return part;

              const newStart = content.indexOf(virtualText);
              // if the virtual text is deleted, remove the part
              if (newStart === -1) return null;

              const newEnd = newStart + virtualText.length;

              if (part.type === "file" && part.source?.text) {
                return {
                  ...part,
                  source: {
                    ...part.source,
                    text: {
                      ...part.source.text,
                      start: newStart,
                      end: newEnd,
                    },
                  },
                };
              }

              if (part.type === "agent" && part.source) {
                return {
                  ...part,
                  source: {
                    ...part.source,
                    start: newStart,
                    end: newEnd,
                  },
                };
              }

              return part;
            })
            .filter((part) => part !== null);

          setPromptUIStore("prompt", {
            input: content,
            // keep only the non-text parts because the text parts were
            // already expanded inline
            parts: updatedNonTextParts,
          });
          restoreExtmarksFromParts(updatedNonTextParts);
          input.cursorOffset = Bun.stringWidth(content);
        },
      },
      {
        title: "Skills",
        value: "prompt.skills",
        category: "Prompt",
        slash: {
          name: "skills",
        },
        onSelect: () => {
          dialog.replace(() => (
            <DialogSkill
              onSelect={(skill) => {
                input.setText(`/${skill} `);
                setPromptUIStore("prompt", {
                  input: `/${skill} `,
                  parts: [],
                });
                input.gotoBufferEnd();
              }}
            />
          ));
        },
      },
    ];
  });

  const ref: PromptRef = {
    get focused() {
      return input.focused;
    },
    get current() {
      return promptUIStore.prompt;
    },
    focus() {
      input.focus();
    },
    blur() {
      input.blur();
    },
    set(prompt) {
      input.setText(prompt.input);
      setPromptUIStore("prompt", prompt);
      restoreExtmarksFromParts(prompt.parts);
      input.gotoBufferEnd();
    },
    reset() {
      input.clear();
      input.extmarks.clear();
      setPromptUIStore("prompt", {
        input: "",
        parts: [],
      });
      setPromptUIStore("extmarkToPartIndex", new Map());
    },
    submit() {
      submit();
    },
  };

  createEffect(() => {
    if (props.visible !== false) input?.focus();
    if (props.visible === false) input?.blur();
  });

  function restoreExtmarksFromParts(parts: PromptInfo["parts"]) {
    input.extmarks.clear();
    setPromptUIStore("extmarkToPartIndex", new Map());

    parts.forEach((part, partIndex) => {
      let start = 0;
      let end = 0;
      let virtualText = "";
      let styleId: number | undefined;

      if (part.type === "file" && part.source?.text) {
        start = part.source.text.start;
        end = part.source.text.end;
        virtualText = part.source.text.value;
        styleId = fileStyleId;
      } else if (part.type === "agent" && part.source) {
        start = part.source.start;
        end = part.source.end;
        virtualText = part.source.value;
        styleId = agentStyleId;
      } else if (part.type === "text" && part.source?.text) {
        start = part.source.text.start;
        end = part.source.text.end;
        virtualText = part.source.text.value;
        styleId = pasteStyleId;
      }

      if (virtualText) {
        const extmarkId = input.extmarks.create({
          start,
          end,
          virtual: true,
          styleId,
          typeId: promptPartTypeId,
        });
        setPromptUIStore("extmarkToPartIndex", (map: Map<number, number>) => {
          const newMap = new Map(map);
          newMap.set(extmarkId, partIndex);
          return newMap;
        });
      }
    });
  }

  function syncExtmarksWithPromptParts() {
    const allExtmarks = input.extmarks.getAllForTypeId(promptPartTypeId);
    setPromptUIStore(
      produce((draft) => {
        const newMap = new Map<number, number>();
        const newParts: typeof draft.prompt.parts = [];

        for (const extmark of allExtmarks) {
          const partIndex = draft.extmarkToPartIndex.get(extmark.id);
          if (partIndex !== undefined) {
            const part = draft.prompt.parts[partIndex];
            if (part) {
              if (part.type === "agent" && part.source) {
                part.source.start = extmark.start;
                part.source.end = extmark.end;
              } else if (part.type === "file" && part.source?.text) {
                part.source.text.start = extmark.start;
                part.source.text.end = extmark.end;
              } else if (part.type === "text" && part.source?.text) {
                part.source.text.start = extmark.start;
                part.source.text.end = extmark.end;
              }
              newMap.set(extmark.id, newParts.length);
              newParts.push(part);
            }
          }
        }

        draft.extmarkToPartIndex = newMap;
        draft.prompt.parts = newParts;
      }),
    );
  }

  command.register(() => [
    {
      title: "Stash prompt",
      value: "prompt.stash",
      category: "Prompt",
      enabled: !!promptUIStore.prompt.input,
      onSelect: (dialog) => {
        if (!promptUIStore.prompt.input) return;
        stash.push({
          input: promptUIStore.prompt.input,
          parts: promptUIStore.prompt.parts,
        });
        input.extmarks.clear();
        input.clear();
        setPromptUIStore("prompt", { input: "", parts: [] });
        setPromptUIStore("extmarkToPartIndex", new Map());
        dialog.clear();
      },
    },
    {
      title: "Stash pop",
      value: "prompt.stash.pop",
      category: "Prompt",
      enabled: stash.list().length > 0,
      onSelect: (dialog) => {
        const entry = stash.pop();
        if (entry) {
          input.setText(entry.input);
          setPromptUIStore("prompt", {
            input: entry.input,
            parts: entry.parts,
          });
          restoreExtmarksFromParts(entry.parts);
          input.gotoBufferEnd();
        }
        dialog.clear();
      },
    },
    {
      title: "Stash list",
      value: "prompt.stash.list",
      category: "Prompt",
      enabled: stash.list().length > 0,
      onSelect: (dialog) => {
        dialog.replace(() => (
          <DialogStash
            onSelect={(entry) => {
              input.setText(entry.input);
              setPromptUIStore("prompt", {
                input: entry.input,
                parts: entry.parts,
              });
              restoreExtmarksFromParts(entry.parts);
              input.gotoBufferEnd();
            }}
          />
        ));
      },
    },
  ]);

  async function submit() {
    log.prompt.info("SUBMIT_START", {
      disabled: props.disabled,
      sessionID: props.sessionID,
      inputLength: promptUIStore.prompt.input?.length,
    });
    try {
      if (props.disabled) {
        log.prompt.debug("SUBMIT_EARLY_EXIT", { reason: "disabled" });
        return;
      }
      if (autocomplete?.visible) {
        log.prompt.debug("SUBMIT_EARLY_EXIT", { reason: "autocomplete_visible" });
        return;
      }
      if (!promptUIStore.prompt.input) {
        log.prompt.debug("SUBMIT_EARLY_EXIT", { reason: "no_input" });
        return;
      }
      const trimmed = promptUIStore.prompt.input.trim();
      if (trimmed === "exit" || trimmed === "quit" || trimmed === ":q") {
        log.prompt.info("SUBMIT_EXIT_COMMAND", {});
        exit();
        return;
      }
      const selectedModel = local.model.current();
      if (!selectedModel) {
        log.prompt.debug("SUBMIT_EARLY_EXIT", { reason: "no_model_selected" });
        promptModelWarning();
        return;
      }
      log.prompt.info("SUBMIT_MODEL_SELECTED", {
        providerID: selectedModel.providerID,
        modelID: selectedModel.modelID,
      });

      let sessionID = props.sessionID;
      if (sessionID == null) {
        log.prompt.info("SUBMIT_SESSION_CREATE", {
          workspaceID: props.workspaceID,
        });
        const res = await api.fosra.session.create({
          workspaceID: props.workspaceID,
        });

        if (res.error) {
          log.prompt.error("SUBMIT_SESSION_CREATE_ERROR", { error: res.error });
          toast.show({
            message:
              "Creating a session failed. Open console for more details.",
            variant: "error",
          });

          return;
        }

        sessionID = (res.data as { id: string }).id;
        log.prompt.info("SUBMIT_SESSION_CREATED", {
          sessionID,
          route: { type: "session", sessionID },
        });
        route.navigate({ type: "session", sessionID });
      } else {
        log.prompt.debug("SUBMIT_EXISTING_SESSION", { sessionID });
      }

      let inputText = promptUIStore.prompt.input;

      const allExtmarks = input.extmarks.getAllForTypeId(promptPartTypeId);
      const sortedExtmarks = allExtmarks.sort(
        (a: { start: number }, b: { start: number }) => b.start - a.start,
      );

      for (const extmark of sortedExtmarks) {
        const partIndex = promptUIStore.extmarkToPartIndex.get(extmark.id);
        if (partIndex !== undefined) {
          const part = promptUIStore.prompt.parts[partIndex];
          if (part?.type === "text" && part.text) {
            const before = inputText.slice(0, extmark.start);
            const after = inputText.slice(extmark.end);
            inputText = before + part.text + after;
          }
        }
      }

      const nonTextParts = promptUIStore.prompt.parts.filter(
        (part) => part.type !== "text",
      );

      const currentMode = promptUIStore.mode;
      const variant = local.model.variant.current();
      log.prompt.debug("SUBMIT_PARTS", {
        nonTextPartsCount: nonTextParts.length,
        nonTextTypes: nonTextParts.map((p) => p.type),
        mode: currentMode,
        variant,
      });

      if (promptUIStore.mode === "shell") {
        log.prompt.info("SUBMIT_SHELL", {
          sessionID,
          agent: local.agent.current().name,
          model: {
            providerID: selectedModel.providerID,
            modelID: selectedModel.modelID,
          },
          commandLength: inputText.length,
        });
        api.fosra.session.shell({
          sessionID,
          agent: local.agent.current().name,
          model: {
            providerID: selectedModel.providerID,
            modelID: selectedModel.modelID,
          },
          command: inputText,
        });
        setPromptUIStore("mode", "normal");
      } else if (
        inputText.startsWith("/") &&
        iife(() => {
          const firstLine = inputText.split("\n")[0];
          const commandName = firstLine.split(" ")[0].slice(1);
          return command.slashes().some((x: any) => x.name === commandName);
        })
      ) {
        const firstLineEnd = inputText.indexOf("\n");
        const firstLine =
          firstLineEnd === -1 ? inputText : inputText.slice(0, firstLineEnd);
        const [command, ...firstLineArgs] = firstLine.split(" ");
        const restOfInput =
          firstLineEnd === -1 ? "" : inputText.slice(firstLineEnd + 1);
        const args =
          firstLineArgs.join(" ") + (restOfInput ? "\n" + restOfInput : "");
        log.prompt.info("SUBMIT_SLASH_COMMAND", {
          sessionID,
          command,
          argumentsLength: args.length,
          agent: local.agent.current().name,
        });

        api.fosra.session.command({
          sessionID,
          command: command.slice(1),
          arguments: args,
          agent: local.agent.current().name,
          model: `${selectedModel.providerID}/${selectedModel.modelID}`,
          variant,
          parts: nonTextParts.filter((x) => x.type === "file"),
        });
      } else {
        log.prompt.info("SUBMIT_PROMPT", {
          sessionID,
          agent: local.agent.current().name,
          modelID: selectedModel.modelID,
          inputLength: inputText.length,
          nonTextPartsCount: nonTextParts.length,
        });

        api.fosra.session
          .prompt({
            sessionID,
            ...selectedModel,
            agent: local.agent.current().name,
            model: selectedModel,
            variant,
            parts: [
              {
                type: "text",
                text: inputText,
              },
              ...nonTextParts,
            ],
          })
          .catch((err) => {
            log.prompt.error("SUBMIT_PROMPT_ERROR", { error: String(err) });
          });
      }
      log.prompt.debug("SUBMIT_COMPLETE", { sessionID });
      history.append({
        ...promptUIStore.prompt,
        mode: currentMode,
      });
      input.extmarks.clear();
      setPromptUIStore("prompt", {
        input: "",
        parts: [],
      });
      setPromptUIStore("extmarkToPartIndex", new Map());
      props.onSubmit?.();

      input.clear();
    } catch (e) {
      log.prompt.error("SUBMIT_ERROR", {
        error: String(e),
        stack: (e as Error)?.stack?.split("\n").slice(0, 3).join("\n"),
      });
    }
  }
  const exit = useExit();

  function pasteText(text: string, virtualText: string) {
    const currentOffset = input.visualCursor.offset;
    const extmarkStart = currentOffset;
    const extmarkEnd = extmarkStart + virtualText.length;

    input.insertText(virtualText + " ");

    const extmarkId = input.extmarks.create({
      start: extmarkStart,
      end: extmarkEnd,
      virtual: true,
      styleId: pasteStyleId,
      typeId: promptPartTypeId,
    });

    setPromptUIStore(
      produce((draft) => {
        const partIndex = draft.prompt.parts.length;
        draft.prompt.parts.push({
          type: "text" as const,
          text,
          source: {
            text: {
              start: extmarkStart,
              end: extmarkEnd,
              value: virtualText,
            },
          },
        });
        draft.extmarkToPartIndex.set(extmarkId, partIndex);
      }),
    );
  }

  async function pasteImage(file: {
    filename?: string;
    content: string;
    mime: string;
  }) {
    const currentOffset = input.visualCursor.offset;
    const extmarkStart = currentOffset;
    const count = promptUIStore.prompt.parts.filter(
      (x) => x.type === "file" && x.mime.startsWith("image/"),
    ).length;
    const virtualText = `[Image ${count + 1}]`;
    const extmarkEnd = extmarkStart + virtualText.length;
    const textToInsert = virtualText + " ";

    input.insertText(textToInsert);

    const extmarkId = input.extmarks.create({
      start: extmarkStart,
      end: extmarkEnd,
      virtual: true,
      styleId: pasteStyleId,
      typeId: promptPartTypeId,
    });

    const part: Omit<FilePart, "id" | "messageID" | "sessionID"> = {
      type: "file" as const,
      mime: file.mime,
      filename: file.filename,
      url: `data:${file.mime};base64,${file.content}`,
      source: {
        type: "file",
        path: file.filename ?? "",
        text: {
          start: extmarkStart,
          end: extmarkEnd,
          value: virtualText,
        },
      },
    };
    setPromptUIStore(
      produce((draft) => {
        const partIndex = draft.prompt.parts.length;
        draft.prompt.parts.push(part);
        draft.extmarkToPartIndex.set(extmarkId, partIndex);
      }),
    );
    return;
  }

  const highlight = createMemo(() => {
    if (keybind.leader) return theme.border;
    if (promptUIStore.mode === "shell") return theme.primary;
    return local.agent.color(local.agent.current().name);
  });

  const showVariant = createMemo(() => {
    const variants = local.model.variant.list();
    if (variants.length === 0) return false;
    const current = local.model.variant.current();
    return !!current;
  });

  const placeholderText = createMemo(() => {
    if (props.sessionID) return undefined;
    if (promptUIStore.mode === "shell") {
      const example =
        SHELL_PLACEHOLDERS[
          promptUIStore.placeholder % SHELL_PLACEHOLDERS.length
        ];
      return `Run a command... "${example}"`;
    }
    return `Ask anything... "${PLACEHOLDERS[promptUIStore.placeholder % PLACEHOLDERS.length]}"`;
  });

  const spinnerDef = createMemo(() => {
    const color = local.agent.color(local.agent.current().name);
    return {
      frames: createFrames({
        color,
        style: "blocks",
        inactiveFactor: 0.6,
        // enableFading: false,
        minAlpha: 0.3,
      }),
      color: createColors({
        color,
        style: "diamonds",
        inactiveFactor: 0.6,
        // enableFading: false,
        minAlpha: 0.3,
      }),
    };
  });

  return (
    <>
      <Autocomplete
        sessionID={props.sessionID}
        ref={(r) => (autocomplete = r)}
        anchor={() => anchor}
        input={() => input}
        setPrompt={(cb) => {
          setPromptUIStore("prompt", produce(cb));
        }}
        setExtmark={(partIndex, extmarkId) => {
          setPromptUIStore("extmarkToPartIndex", (map: Map<number, number>) => {
            const newMap = new Map(map);
            newMap.set(extmarkId, partIndex);
            return newMap;
          });
        }}
        value={promptUIStore.prompt.input}
        fileStyleId={fileStyleId}
        agentStyleId={agentStyleId}
        promptPartTypeId={() => promptPartTypeId}
      />
      <box ref={(r) => (anchor = r)} visible={props.visible !== false}>
        <box
          border={["left"]}
          borderColor={highlight()}
          customBorderChars={{
            ...EmptyBorder,
            vertical: "┃",
            bottomLeft: "╹",
          }}
        >
          <box
            paddingLeft={2}
            paddingRight={2}
            paddingTop={1}
            flexShrink={0}
            backgroundColor={theme.backgroundElement}
            flexGrow={1}
          >
            <textarea
              placeholder={placeholderText()}
              textColor={keybind.leader ? theme.textMuted : theme.text}
              focusedTextColor={keybind.leader ? theme.textMuted : theme.text}
              minHeight={1}
              maxHeight={6}
              onContentChange={() => {
                const value = input.plainText;
                setPromptUIStore("prompt", "input", value);
                autocomplete.onInput(value);
                syncExtmarksWithPromptParts();
              }}
              keyBindings={textareaKeybindings()}
              onKeyDown={async (e) => {
                if (props.disabled) {
                  e.preventDefault();
                  return;
                }
                if (e.name === "return" && !e.ctrl && !e.meta && !e.shift) {
                  if (!autocomplete?.visible && promptUIStore.prompt.input) {
                    e.preventDefault();
                    submit();
                    return;
                  }
                }
                // Handle clipboard paste (Ctrl+V) - check for images first on Windows
                // This is needed because Windows terminal doesn't properly send image data
                // through bracketed paste, so we need to intercept the keypress and
                // directly read from clipboard before the terminal handles it
                if (keybind.match("input_paste", e)) {
                  const content = await Clipboard.read();
                  if (content?.mime.startsWith("image/")) {
                    e.preventDefault();
                    await pasteImage({
                      filename: "clipboard",
                      mime: content.mime,
                      content: content.data,
                    });
                    return;
                  }
                  // If no image, let the default paste behavior continue
                }
                if (
                  keybind.match("input_clear", e) &&
                  promptUIStore.prompt.input !== ""
                ) {
                  input.clear();
                  input.extmarks.clear();
                  setPromptUIStore("prompt", {
                    input: "",
                    parts: [],
                  });
                  setPromptUIStore("extmarkToPartIndex", new Map());
                  return;
                }
                if (keybind.match("app_exit", e)) {
                  if (promptUIStore.prompt.input === "") {
                    await exit();
                    // Don't preventDefault - let textarea potentially handle the event
                    e.preventDefault();
                    return;
                  }
                }
                if (e.name === "!" && input.visualCursor.offset === 0) {
                  setPromptUIStore(
                    "placeholder",
                    Math.floor(Math.random() * SHELL_PLACEHOLDERS.length),
                  );
                  setPromptUIStore("mode", "shell");
                  e.preventDefault();
                  return;
                }
                if (promptUIStore.mode === "shell") {
                  if (
                    (e.name === "backspace" &&
                      input.visualCursor.offset === 0) ||
                    e.name === "escape"
                  ) {
                    setPromptUIStore("mode", "normal");
                    e.preventDefault();
                    return;
                  }
                }
                if (promptUIStore.mode === "normal") autocomplete.onKeyDown(e);
                if (!autocomplete.visible) {
                  if (
                    (keybind.match("history_previous", e) &&
                      input.cursorOffset === 0) ||
                    (keybind.match("history_next", e) &&
                      input.cursorOffset === input.plainText.length)
                  ) {
                    const direction = keybind.match("history_previous", e)
                      ? -1
                      : 1;
                    const item = history.move(direction, input.plainText);

                    if (item) {
                      input.setText(item.input);
                      setPromptUIStore("prompt", item);
                      setPromptUIStore("mode", item.mode ?? "normal");
                      restoreExtmarksFromParts(item.parts);
                      e.preventDefault();
                      if (direction === -1) input.cursorOffset = 0;
                      if (direction === 1)
                        input.cursorOffset = input.plainText.length;
                    }
                    return;
                  }

                  if (
                    keybind.match("history_previous", e) &&
                    input.visualCursor.visualRow === 0
                  )
                    input.cursorOffset = 0;
                  if (
                    keybind.match("history_next", e) &&
                    input.visualCursor.visualRow === input.height - 1
                  )
                    input.cursorOffset = input.plainText.length;
                }
              }}
              onPaste={async (event: PasteEvent) => {
                if (props.disabled) {
                  event.preventDefault();
                  return;
                }

                // normalize line endings at the boundary
                // windows ConPTY/Terminal often sends CR-only newlines in bracketed paste
                // replace CRLF first, then any remaining CR
                const normalizedText = event.text
                  .replace(/\r\n/g, "\n")
                  .replace(/\r/g, "\n");
                const pastedContent = normalizedText.trim();
                if (!pastedContent) {
                  command.trigger("prompt.paste");
                  return;
                }

                // trim ' from the beginning and end of the pasted content. just
                // ' and nothing else
                const filepath = pastedContent
                  .replace(/^'+|'+$/g, "")
                  .replace(/\\ /g, " ");
                const isUrl = /^(https?):\/\//.test(filepath);
                if (!isUrl) {
                  try {
                    const mime = Filesystem.mimeType(filepath);
                    const filename = path.basename(filepath);
                    // Handle SVG as raw text content, not as base64 image
                    if (mime === "image/svg+xml") {
                      event.preventDefault();
                      const content = await Filesystem.readText(filepath).catch(
                        () => {},
                      );
                      if (content) {
                        pasteText(content, `[SVG: ${filename ?? "image"}]`);
                        return;
                      }
                    }
                    if (mime.startsWith("image/")) {
                      event.preventDefault();
                      const content = await Filesystem.readArrayBuffer(filepath)
                        .then((buffer) =>
                          Buffer.from(buffer).toString("base64"),
                        )
                        .catch(() => {});
                      if (content) {
                        await pasteImage({
                          filename,
                          mime,
                          content,
                        });
                        return;
                      }
                    }
                  } catch {}
                }

                const lineCount = (pastedContent.match(/\n/g)?.length ?? 0) + 1;
                if (
                  (lineCount >= 3 || pastedContent.length > 150) &&
                  !store.state.config().experimental?.disable_paste_summary
                ) {
                  event.preventDefault();
                  pasteText(pastedContent, `[Pasted ~${lineCount} lines]`);
                  return;
                }

                // Force layout update and render for the pasted content
                setTimeout(() => {
                  // setTimeout is a workaround and needs to be addressed properly
                  if (!input || input.isDestroyed) return;
                  input.getLayoutNode().markDirty();
                  renderer.requestRender();
                }, 0);
              }}
              ref={(r: TextareaRenderable) => {
                input = r;
                if (promptPartTypeId === 0) {
                  promptPartTypeId = input.extmarks.registerType("prompt-part");
                }
                props.ref?.(ref);
                setTimeout(() => {
                  // setTimeout is a workaround and needs to be addressed properly
                  if (!input || input.isDestroyed) return;
                  input.cursorColor = theme.text;
                }, 0);
              }}
              onMouseDown={(r: MouseEvent) => r.target?.focus()}
              focusedBackgroundColor={theme.backgroundElement}
              cursorColor={theme.text}
              syntaxStyle={syntax()}
            />
            <box flexDirection="row" flexShrink={0} paddingTop={1} gap={1}>
              <text fg={highlight()}>
                {promptUIStore.mode === "shell"
                  ? "Shell"
                  : Locale.titlecase(local.agent.current().name)}{" "}
              </text>
              <Show when={promptUIStore.mode === "normal"}>
                <box flexDirection="row" gap={1}>
                  <text
                    flexShrink={0}
                    fg={keybind.leader ? theme.textMuted : theme.text}
                  >
                    {local.model.parsed().model}
                  </text>
                  <text fg={theme.textMuted}>
                    {local.model.parsed().provider}
                  </text>
                  <Show when={showVariant()}>
                    <text fg={theme.textMuted}>·</text>
                    <text>
                      <span style={{ fg: theme.warning, bold: true }}>
                        {local.model.variant.current()}
                      </span>
                    </text>
                  </Show>
                </box>
              </Show>
            </box>
          </box>
        </box>
        <box
          height={1}
          border={["left"]}
          borderColor={highlight()}
          customBorderChars={{
            ...EmptyBorder,
            vertical: theme.backgroundElement.a !== 0 ? "╹" : " ",
          }}
        >
          <box
            height={1}
            border={["bottom"]}
            borderColor={theme.backgroundElement}
            customBorderChars={
              theme.backgroundElement.a !== 0
                ? {
                    ...EmptyBorder,
                    horizontal: "▀",
                  }
                : {
                    ...EmptyBorder,
                    horizontal: " ",
                  }
            }
          />
        </box>
        <box flexDirection="row" justifyContent="space-between">
          <Show when={status().type !== "idle"} fallback={<text />}>
            <box
              flexDirection="row"
              gap={1}
              flexGrow={1}
              justifyContent={
                status().type === "retry" ? "space-between" : "flex-start"
              }
            >
              <box flexShrink={0} flexDirection="row" gap={1}>
                <box marginLeft={1}>
                  <Show
                    when={kv.get("animations_enabled", true)}
                    fallback={<text fg={theme.textMuted}>[⋯]</text>}
                  >
                    <spinner
                      color={spinnerDef().color}
                      frames={spinnerDef().frames}
                      interval={41}
                    />
                  </Show>
                </box>
                <box flexDirection="row" gap={1} flexShrink={0}>
                  {(() => {
                    const retry = createMemo(() => {
                      const s = status();
                      if (s.type !== "retry") return;
                      return s;
                    });
                    const message = createMemo(() => {
                      const r = retry();
                      if (!r) return;
                      if (
                        (r as any).message.includes(
                          "exceeded your current quota",
                        ) &&
                        (r as any).message.includes("gemini")
                      )
                        return "gemini is way too hot right now";
                      if ((r as any).message.length > 80)
                        return (r as any).message.slice(0, 80) + "...";
                      return (r as any).message;
                    });
                    const isTruncated = createMemo(() => {
                      const r = retry();
                      if (!r) return false;
                      return (r as any).message.length > 120;
                    });
                    const [seconds, setSeconds] = createSignal(0);
                    onMount(() => {
                      const timer = setInterval(() => {
                        const next = (retry() as any)?.next;
                        if (next)
                          setSeconds(Math.round((next - Date.now()) / 1000));
                      }, 1000);

                      onCleanup(() => {
                        clearInterval(timer);
                      });
                    });
                    const handleMessageClick = () => {
                      const r = retry();
                      if (!r) return;
                      if (isTruncated()) {
                        DialogAlert.show(
                          dialog,
                          "Retry Error",
                          (r as any).message,
                        );
                      }
                    };

                    const retryText = () => {
                      const r = retry();
                      if (!r) return "";
                      const baseMessage = message();
                      const truncatedHint = isTruncated()
                        ? " (click to expand)"
                        : "";
                      const duration = formatDuration(seconds());
                      const retryInfo = ` [retrying ${duration ? `in ${duration} ` : ""}attempt #${(r as any).attempt}]`;
                      return baseMessage + truncatedHint + retryInfo;
                    };

                    return (
                      <Show when={retry()}>
                        <box onMouseUp={handleMessageClick}>
                          <text fg={theme.error}>{retryText()}</text>
                        </box>
                      </Show>
                    );
                  })()}
                </box>
              </box>
              <text
                fg={promptUIStore.interrupt > 0 ? theme.primary : theme.text}
              >
                esc{" "}
                <span
                  style={{
                    fg:
                      promptUIStore.interrupt > 0
                        ? theme.primary
                        : theme.textMuted,
                  }}
                >
                  {promptUIStore.interrupt > 0
                    ? "again to interrupt"
                    : "interrupt"}
                </span>
              </text>
            </box>
          </Show>
          <Show when={status().type !== "retry"}>
            <box gap={2} flexDirection="row">
              <Switch>
                <Match when={promptUIStore.mode === "normal"}>
                  <Show when={local.model.variant.list().length > 0}>
                    <text fg={theme.text}>
                      {keybind.print("variant_cycle")}{" "}
                      <span style={{ fg: theme.textMuted }}>variants</span>
                    </text>
                  </Show>
                  <text fg={theme.text}>
                    {keybind.print("agent_cycle")}{" "}
                    <span style={{ fg: theme.textMuted }}>agents</span>
                  </text>
                  <text fg={theme.text}>
                    {keybind.print("command_list")}{" "}
                    <span style={{ fg: theme.textMuted }}>commands</span>
                  </text>
                </Match>
                <Match when={promptUIStore.mode === "shell"}>
                  <text fg={theme.text}>
                    esc{" "}
                    <span style={{ fg: theme.textMuted }}>exit shell mode</span>
                  </text>
                </Match>
              </Switch>
            </box>
          </Show>
        </box>
      </box>
    </>
  );
}
