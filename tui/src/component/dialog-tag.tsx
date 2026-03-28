import { createMemo, createResource } from "solid-js"
import { DialogSelect } from "@tui/ui/dialog-select"
import { useDialog } from "@tui/ui/dialog"
import { useApi } from "../context/api"
import { createStore } from "solid-js/store"

export function DialogTag(props: { onSelect?: (value: string) => void }) {
  const api = useApi()
  const dialog = useDialog()

  const [store] = createStore({
    filter: "",
  })

  const [files] = createResource<string[], string[]>(
    () => [store.filter],
    async () => {
      const result = await api.fosra.find.files({
        query: store.filter,
      })
      if (result.error) return []
      const sliced = (result.data ?? []).slice(0, 5)
      return sliced as string[]
    },
  )

  const options = createMemo(() =>
    (files() ?? []).map((file) => ({
      value: file,
      title: file,
    })),
  )

  return (
    <DialogSelect
      title="Autocomplete"
      options={options()}
      onSelect={(option) => {
        props.onSelect?.(option.value)
        dialog.clear()
      }}
    />
  )
}
