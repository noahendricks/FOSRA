import { createMemo, onMount } from "solid-js"
import { useStore } from "@tui/context/store"
import { DialogSelect, type DialogSelectOption } from "@tui/components/dialogs/dialog-select"
import type { TextPart } from "@fosra/api/v2"
import { Locale } from "@/util/locale"
import { useApi } from "@tui/context/api"
import { useRoute } from "@tui/context/route"
import { useDialog } from "@tui/components/dialogs/dialog"
import type { PromptInfo } from "@tui/components/prompt/history"
import { strip } from "@tui/components/prompt/part"

export function DialogForkFromTimeline(props: { sessionID: string; onMove: (messageID: string) => void }) {
  const store = useStore()
  const dialog = useDialog()
  const api = useApi()
  const route = useRoute()

  onMount(() => {
    dialog.setSize("large")
  })

  const options = createMemo((): DialogSelectOption<string>[] => {
    const messages = store.state.messages.get(props.sessionID) ?? []
    const result = [] as DialogSelectOption<string>[]
    for (const message of messages) {
      if (message.role !== "user") continue
      const part = (store.state.parts.get(message.id) ?? []).find(
        (x) => x.type === "text" && !x.synthetic && !x.ignored,
      ) as TextPart
      if (!part) continue
      result.push({
        title: part.text.replace(/\n/g, " "),
        value: message.id,
        footer: Locale.time(message.time.created),
        onSelect: async (dialog) => {
          const forked = await api.fosra.session.fork({
            sessionID: props.sessionID,
            messageID: message.id,
          })
          const parts = store.state.parts.get(message.id) ?? []
          const initialPrompt = parts.reduce(
            (agg, part) => {
              if (part.type === "text") {
                if (!part.synthetic) agg.input += part.text
              }
              if (part.type === "file") agg.parts.push(strip(part))
              return agg
            },
            { input: "", parts: [] as PromptInfo["parts"] },
          )
          route.navigate({
            sessionID: forked.data!.id,
            type: "session",
            initialPrompt,
          })
          dialog.clear()
        },
      })
    }
    result.reverse()
    return result
  })

  return <DialogSelect onMove={(option) => props.onMove(option.value)} title="Fork from message" options={options()} />
}
