import { createMemo } from "solid-js"
import { useStore } from "@tui/context/store"
import { DialogSelect } from "@tui/components/dialogs/dialog-select"
import { useApi } from "@tui/context/api"
import { useRoute } from "@tui/context/route"
import { Clipboard } from "@tui/util/clipboard"
import type { PromptInfo } from "@tui/components/prompt/history"
import { strip } from "@tui/components/prompt/part"

export function DialogMessage(props: {
  messageID: string
  sessionID: string
  setPrompt?: (prompt: PromptInfo) => void
}) {
  const store = useStore()
  const api = useApi()
  const message = createMemo(() => store.state.messages.get(props.sessionID)?.find((x) => x.id === props.messageID))
  const route = useRoute()

  return (
    <DialogSelect
      title="Message Actions"
      options={[
        {
          title: "Revert",
          value: "session.revert",
          description: "undo messages and file changes",
          onSelect: (dialog) => {
            const msg = message()
            if (!msg) return

            api.fosra.session.revert({
              sessionID: props.sessionID,
              messageID: msg.id,
            })

            if (props.setPrompt) {
              const parts = store.state.parts.get(msg.id) ?? []
              const promptInfo = parts.reduce(
                (agg, part) => {
                  if (part.type === "text") {
                    if (!part.synthetic) agg.input += part.text
                  }
                  if (part.type === "file") agg.parts.push(strip(part))
                  return agg
                },
                { input: "", parts: [] as PromptInfo["parts"] },
              )
              props.setPrompt(promptInfo)
            }

            dialog.clear()
          },
        },
        {
          title: "Copy",
          value: "message.copy",
          description: "message text to clipboard",
          onSelect: async (dialog) => {
            const msg = message()
            if (!msg) return

            const parts = store.state.parts.get(msg.id) ?? []
            const text = parts.reduce((agg, part) => {
              if (part.type === "text" && !part.synthetic) {
                agg += part.text
              }
              return agg
            }, "")

            await Clipboard.copy(text)
            dialog.clear()
          },
        },
        {
          title: "Fork",
          value: "session.fork",
          description: "create a new session",
          onSelect: async (dialog) => {
            const result = await api.fosra.session.fork({
              sessionID: props.sessionID,
              messageID: props.messageID,
            })
            const initialPrompt = (() => {
              const msg = message()
              if (!msg) return undefined
              const parts = store.state.parts.get(msg.id) ?? []
              return parts.reduce(
                (agg, part) => {
                  if (part.type === "text") {
                    if (!part.synthetic) agg.input += part.text
                  }
                  if (part.type === "file") agg.parts.push(part)
                  return agg
                },
                { input: "", parts: [] as PromptInfo["parts"] },
              )
            })()
            route.navigate({
              sessionID: result.data!.id,
              type: "session",
              initialPrompt,
            })
            dialog.clear()
          },
        },
      ]}
    />
  )
}
