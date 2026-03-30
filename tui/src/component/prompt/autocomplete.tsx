import type { BoxRenderable, TextareaRenderable, KeyEvent, ScrollBoxRenderable } from "@opentui/core"
import { pathToFileURL } from "bun"
import fuzzysort from "fuzzysort"
import { firstBy } from "remeda"
import { createMemo, createResource, createEffect, onMount, onCleanup, Index, Show, createSignal } from "solid-js"
import { createStore } from "solid-js/store"
import { useApi } from "@tui/context/api"
import { useStore } from "@tui/context/store"
import { useTheme, selectedForeground } from "@tui/context/theme"
import { SplitBorder } from "@tui/component/border"
import { useCommandDialog } from "@tui/component/dialog-command"
import { useTerminalDimensions } from "@opentui/solid"
import { Locale } from "@/util/locale"
import type { PromptInfo } from "./history"
import { useFrecency } from "./frecency"

function removeLineRange(input: string) {
  const hashIndex = input.lastIndexOf("#")
  return hashIndex !== -1 ? input.substring(0, hashIndex) : input
}

function extractLineRange(input: string) {
  const hashIndex = input.lastIndexOf("#")
  if (hashIndex === -1) {
    return { baseQuery: input }
  }

  const baseName = input.substring(0, hashIndex)
  const linePart = input.substring(hashIndex + 1)
  const lineMatch = linePart.match(/^(\d+)(?:-(\d*))?$/)

  if (!lineMatch) {
    return { baseQuery: baseName }
  }

  const startLine = Number(lineMatch[1])
  const endLine = lineMatch[2] && startLine < Number(lineMatch[2]) ? Number(lineMatch[2]) : undefined

  return {
    lineRange: {
      baseName,
      startLine,
      endLine,
    },
    baseQuery: baseName,
  }
}

export type AutocompleteRef = {
  onInput: (value: string) => void
  onKeyDown: (e: KeyEvent) => void
  visible: false | "@" | "/"
}

export type AutocompleteOption = {
  display: string
  value?: string
  aliases?: string[]
  disabled?: boolean
  description?: string
  isDirectory?: boolean
  onSelect?: () => void
  path?: string
}

export function Autocomplete(props: {
  value: string
  sessionID?: string
  setPrompt: (input: (prompt: PromptInfo) => void) => void
  setExtmark: (partIndex: number, extmarkId: number) => void
  anchor: () => BoxRenderable
  input: () => TextareaRenderable
  ref: (ref: AutocompleteRef) => void
  fileStyleId: number
  agentStyleId: number
  promptPartTypeId: () => number
}) {
  const api = useApi()
  const store = useStore()
  const command = useCommandDialog()
  const { theme } = useTheme()
  const dimensions = useTerminalDimensions()
  const frecency = useFrecency()

  const [autocompleteStore, setAutocompleteStore] = createStore({
    index: 0,
    selected: 0,
    visible: false as AutocompleteRef["visible"],
    input: "keyboard" as "keyboard" | "mouse",
  })

  const [positionTick, setPositionTick] = createSignal(0)

  createEffect(() => {
    if (autocompleteStore.visible) {
      let lastPos = { x: 0, y: 0, width: 0 }
      const interval = setInterval(() => {
        const anchor = props.anchor()
        if (anchor.x !== lastPos.x || anchor.y !== lastPos.y || anchor.width !== lastPos.width) {
          lastPos = { x: anchor.x, y: anchor.y, width: anchor.width }
          setPositionTick((t) => t + 1)
        }
      }, 50)

      onCleanup(() => clearInterval(interval))
    }
  })

  const position = createMemo(() => {
    if (!autocompleteStore.visible) return { x: 0, y: 0, width: 0 }
    const dims = dimensions()
    positionTick()
    const anchor = props.anchor()
    const parent = anchor.parent
    const parentX = parent?.x ?? 0
    const parentY = parent?.y ?? 0

    return {
      x: anchor.x - parentX,
      y: anchor.y - parentY,
      width: anchor.width,
    }
  })

  const filter = createMemo(() => {
    if (!autocompleteStore.visible) return
    // Track props.value to make memo reactive to text changes
    props.value // <- there surely is a better way to do this, like making .input() reactive

    return props.input().getTextRange(autocompleteStore.index + 1, props.input().cursorOffset)
  })

  // filter() reads reactive props.value plus non-reactive cursor/text state.
  // On keypress those can be briefly out of sync, so filter() may return an empty/partial string.
  // Copy it into search in an effect because effects run after reactive updates have been rendered and painted
  // so the input has settled and all consumers read the same stable value.
  const [search, setSearch] = createSignal("")
  createEffect(() => {
    const next = filter()
    setSearch(next ? next : "")
  })

  // When the filter changes due to how TUI works, the mousemove might still be triggered
  // via a synthetic event as the layout moves underneath the cursor. This is a workaround to make sure the input mode remains keyboard so
  // that the mouseover event doesn't trigger when filtering.
  createEffect(() => {
    filter()
    setAutocompleteStore("input", "keyboard")
  })

  function insertPart(text: string, part: PromptInfo["parts"][number]) {
    const input = props.input()
    const currentCursorOffset = input.cursorOffset

    const charAfterCursor = props.value.at(currentCursorOffset)
    const needsSpace = charAfterCursor !== " "
    const append = "@" + text + (needsSpace ? " " : "")

    input.cursorOffset = autocompleteStore.index
    const startCursor = input.logicalCursor
    input.cursorOffset = currentCursorOffset
    const endCursor = input.logicalCursor

    input.deleteRange(startCursor.row, startCursor.col, endCursor.row, endCursor.col)
    input.insertText(append)

    const virtualText = "@" + text
    const extmarkStart = autocompleteStore.index
    const extmarkEnd = extmarkStart + Bun.stringWidth(virtualText)

    const styleId = part.type === "file" ? props.fileStyleId : part.type === "agent" ? props.agentStyleId : undefined

    const extmarkId = input.extmarks.create({
      start: extmarkStart,
      end: extmarkEnd,
      virtual: true,
      styleId,
      typeId: props.promptPartTypeId(),
    })

    props.setPrompt((draft) => {
      if (part.type === "file") {
        const existingIndex = draft.parts.findIndex((p) => p.type === "file" && "url" in p && p.url === part.url)
        if (existingIndex !== -1) {
          const existing = draft.parts[existingIndex]
          if (
            part.source?.text &&
            existing &&
            "source" in existing &&
            existing.source &&
            "text" in existing.source &&
            existing.source.text
          ) {
            existing.source.text.start = extmarkStart
            existing.source.text.end = extmarkEnd
            existing.source.text.value = virtualText
          }
          return
        }
      }

      if (part.type === "file" && part.source?.text) {
        part.source.text.start = extmarkStart
        part.source.text.end = extmarkEnd
        part.source.text.value = virtualText
      } else if (part.type === "agent" && part.source) {
        part.source.start = extmarkStart
        part.source.end = extmarkEnd
        part.source.value = virtualText
      }
      const partIndex = draft.parts.length
      draft.parts.push(part)
      props.setExtmark(partIndex, extmarkId)
    })

    if (part.type === "file" && part.source && part.source.type === "file") {
      frecency.updateFrecency(part.source.path)
    }
  }

  const [files] = createResource(
    () => search(),
    async (query) => {
      if (!autocompleteStore.visible || autocompleteStore.visible === "/") return []

      const { lineRange, baseQuery } = extractLineRange(query ?? "")

      // Get files from SDK
      const result = await api.fosra.find.files({
        query: baseQuery,
      })

      const options: AutocompleteOption[] = []

      // Add file options
      if (!result.error && result.data) {
        const sortedFiles = result.data.sort((a, b) => {
          const aScore = frecency.getFrecency(a)
          const bScore = frecency.getFrecency(b)
          if (aScore !== bScore) return bScore - aScore
          const aDepth = a.split("/").length
          const bDepth = b.split("/").length
          if (aDepth !== bDepth) return aDepth - bDepth
          return a.localeCompare(b)
        })

        const width = props.anchor().width - 4
        options.push(
          ...sortedFiles.map((item): AutocompleteOption => {
            const baseDir = process.cwd().replace(/\/+$/, "")
            const fullPath = `${baseDir}/${item}`
            const urlObj = pathToFileURL(fullPath)
            let filename = item
            if (lineRange && !item.endsWith("/")) {
              filename = `${item}#${lineRange.startLine}${lineRange.endLine ? `-${lineRange.endLine}` : ""}`
              urlObj.searchParams.set("start", String(lineRange.startLine))
              if (lineRange.endLine !== undefined) {
                urlObj.searchParams.set("end", String(lineRange.endLine))
              }
            }
            const url = urlObj.href

            const isDir = item.endsWith("/")
            return {
              display: Locale.truncateMiddle(filename, width),
              value: filename,
              isDirectory: isDir,
              path: item,
              onSelect: () => {
                insertPart(filename, {
                  type: "file",
                  mime: "text/plain",
                  filename,
                  url,
                  source: {
                    type: "file",
                    text: {
                      start: 0,
                      end: 0,
                      value: "",
                    },
                    path: item,
                  },
                })
              },
            }
          }),
        )
      }

      return options
    },
    {
      initialValue: [],
    },
  )

  const mcpResources = createMemo(() => {
    if (!autocompleteStore.visible || autocompleteStore.visible === "/") return []

    const options: AutocompleteOption[] = []
    const width = props.anchor().width - 4

    return options
  })

  const agents = createMemo(() => {
    const agents = store.state.agents()
    return agents
      .filter((agent) => !agent.hidden && agent.mode !== "primary")
      .map(
        (agent): AutocompleteOption => ({
          display: "@" + agent.name,
          onSelect: () => {
            insertPart(agent.name, {
              type: "agent",
              name: agent.name,
              source: {
                start: 0,
                end: 0,
                value: "",
              },
            })
          },
        }),
      )
  })

  const commands = createMemo((): AutocompleteOption[] => {
    const results: AutocompleteOption[] = [...command.slashes()]

    results.sort((a, b) => a.display.localeCompare(b.display))

    const max = firstBy(results, [(x) => x.display.length, "desc"])?.display.length
    if (!max) return results
    return results.map((item) => ({
      ...item,
      display: item.display.padEnd(max + 2),
    }))
  })

  const options = createMemo((prev: AutocompleteOption[] | undefined) => {
    const filesValue = files()
    const agentsValue = agents()
    const commandsValue = commands()

    const mixed: AutocompleteOption[] =
      autocompleteStore.visible === "@" ? [...agentsValue, ...(filesValue || []), ...mcpResources()] : [...commandsValue]

    const searchValue = search()

    if (!searchValue) {
      return mixed
    }

    if (files.loading && prev && prev.length > 0) {
      return prev
    }

    const result = fuzzysort.go(removeLineRange(searchValue), mixed, {
      keys: [
        (obj) => removeLineRange((obj.value ?? obj.display).trimEnd()),
        "description",
        (obj) => obj.aliases?.join(" ") ?? "",
      ],
      limit: 10,
      scoreFn: (objResults) => {
        const displayResult = objResults[0]
        let score = objResults.score
        if (displayResult && displayResult.target.startsWith(autocompleteStore.visible + searchValue)) {
          score *= 2
        }
        const frecencyScore = objResults.obj.path ? frecency.getFrecency(objResults.obj.path) : 0
        return score * (1 + frecencyScore)
      },
    })

    return result.map((arr) => arr.obj)
  })

  createEffect(() => {
    filter()
    setAutocompleteStore("selected", 0)
  })

  function move(direction: -1 | 1) {
    if (!autocompleteStore.visible) return
    if (!options().length) return
    let next = autocompleteStore.selected + direction
    if (next < 0) next = options().length - 1
    if (next >= options().length) next = 0
    moveTo(next)
  }

  function moveTo(next: number) {
    setAutocompleteStore("selected", next)
    if (!scroll) return
    const viewportHeight = Math.min(height(), options().length)
    const scrollBottom = scroll.scrollTop + viewportHeight
    if (next < scroll.scrollTop) {
      scroll.scrollBy(next - scroll.scrollTop)
    } else if (next + 1 > scrollBottom) {
      scroll.scrollBy(next + 1 - scrollBottom)
    }
  }

  function select() {
    const selected = options()[autocompleteStore.selected]
    if (!selected) return
    hide()
    selected.onSelect?.()
  }

  function expandDirectory() {
    const selected = options()[autocompleteStore.selected]
    if (!selected) return

    const input = props.input()
    const currentCursorOffset = input.cursorOffset

    const displayText = selected.display.trimEnd()
    const path = displayText.startsWith("@") ? displayText.slice(1) : displayText

    input.cursorOffset = autocompleteStore.index
    const startCursor = input.logicalCursor
    input.cursorOffset = currentCursorOffset
    const endCursor = input.logicalCursor

    input.deleteRange(startCursor.row, startCursor.col, endCursor.row, endCursor.col)
    input.insertText("@" + path)

    setAutocompleteStore("selected", 0)
  }

  function show(mode: "@" | "/") {
    command.keybinds(false)
    setAutocompleteStore({
      visible: mode,
      index: props.input().cursorOffset,
    })
  }

  function hide() {
    const text = props.input().plainText
    if (autocompleteStore.visible === "/" && !text.endsWith(" ") && text.startsWith("/")) {
      const cursor = props.input().logicalCursor
      props.input().deleteRange(0, 0, cursor.row, cursor.col)
      // Sync the prompt store immediately since onContentChange is async
      props.setPrompt((draft) => {
        draft.input = props.input().plainText
      })
    }
    command.keybinds(true)
    setAutocompleteStore("visible", false)
  }

  onMount(() => {
    props.ref({
      get visible() {
        return autocompleteStore.visible
      },
      onInput(value) {
        if (autocompleteStore.visible) {
          if (
            // Typed text before the trigger
            props.input().cursorOffset <= autocompleteStore.index ||
            // There is a space between the trigger and the cursor
            props.input().getTextRange(autocompleteStore.index, props.input().cursorOffset).match(/\s/) ||
            // "/<command>" is not the sole content
            (autocompleteStore.visible === "/" && value.match(/^\S+\s+\S+\s*$/))
          ) {
            hide()
          }
          return
        }

        // Check if autocomplete should reopen (e.g., after backspace deleted a space)
        const offset = props.input().cursorOffset
        if (offset === 0) return

        // Check for "/" at position 0 - reopen slash commands
        if (value.startsWith("/") && !value.slice(0, offset).match(/\s/)) {
          show("/")
          setAutocompleteStore("index", 0)
          return
        }

        // Check for "@" trigger - find the nearest "@" before cursor with no whitespace between
        const text = value.slice(0, offset)
        const idx = text.lastIndexOf("@")
        if (idx === -1) return

        const between = text.slice(idx)
        const before = idx === 0 ? undefined : value[idx - 1]
        if ((before === undefined || /\s/.test(before)) && !between.match(/\s/)) {
          show("@")
          setAutocompleteStore("index", idx)
        }
      },
      onKeyDown(e: KeyEvent) {
        if (autocompleteStore.visible) {
          const name = e.name?.toLowerCase()
          const ctrlOnly = e.ctrl && !e.meta && !e.shift
          const isNavUp = name === "up" || (ctrlOnly && name === "p")
          const isNavDown = name === "down" || (ctrlOnly && name === "n")

          if (isNavUp) {
            setAutocompleteStore("input", "keyboard")
            move(-1)
            e.preventDefault()
            return
          }
          if (isNavDown) {
            setAutocompleteStore("input", "keyboard")
            move(1)
            e.preventDefault()
            return
          }
          if (name === "escape") {
            hide()
            e.preventDefault()
            return
          }
          if (name === "return") {
            select()
            e.preventDefault()
            return
          }
          if (name === "tab") {
            const selected = options()[autocompleteStore.selected]
            if (selected?.isDirectory) {
              expandDirectory()
            } else {
              select()
            }
            e.preventDefault()
            return
          }
        }
        if (!autocompleteStore.visible) {
          if (e.name === "@") {
            const cursorOffset = props.input().cursorOffset
            const charBeforeCursor =
              cursorOffset === 0 ? undefined : props.input().getTextRange(cursorOffset - 1, cursorOffset)
            const canTrigger = charBeforeCursor === undefined || charBeforeCursor === "" || /\s/.test(charBeforeCursor)
            if (canTrigger) show("@")
          }

          if (e.name === "/") {
            if (props.input().cursorOffset === 0) show("/")
          }
        }
      },
    })
  })

  const height = createMemo(() => {
    const count = options().length || 1
    if (!autocompleteStore.visible) return Math.min(10, count)
    positionTick()
    return Math.min(10, count, Math.max(1, props.anchor().y))
  })

  let scroll: ScrollBoxRenderable

  return (
    <box
      visible={autocompleteStore.visible !== false}
      position="absolute"
      top={position().y - height()}
      left={position().x}
      width={position().width}
      zIndex={100}
      {...SplitBorder}
      borderColor={theme.border}
    >
      <scrollbox
        ref={(r: ScrollBoxRenderable) => (scroll = r)}
        backgroundColor={theme.backgroundMenu}
        height={height()}
        scrollbarOptions={{ visible: false }}
      >
        <Index
          each={options()}
          fallback={
            <box paddingLeft={1} paddingRight={1}>
              <text fg={theme.textMuted}>No matching items</text>
            </box>
          }
        >
          {(option, index) => (
            <box
              paddingLeft={1}
              paddingRight={1}
              backgroundColor={index === autocompleteStore.selected ? theme.primary : undefined}
              flexDirection="row"
              onMouseMove={() => {
                setAutocompleteStore("input", "mouse")
              }}
              onMouseOver={() => {
                if (autocompleteStore.input !== "mouse") return
                moveTo(index)
              }}
              onMouseDown={() => {
                setAutocompleteStore("input", "mouse")
                moveTo(index)
              }}
              onMouseUp={() => select()}
            >
              <text fg={index === autocompleteStore.selected ? selectedForeground(theme) : theme.text} flexShrink={0}>
                {option().display}
              </text>
              <Show when={option().description}>
                <text fg={index === autocompleteStore.selected ? selectedForeground(theme) : theme.textMuted} wrapMode="none">
                  {option().description}
                </text>
              </Show>
            </box>
          )}
        </Index>
      </scrollbox>
    </box>
  )
}
