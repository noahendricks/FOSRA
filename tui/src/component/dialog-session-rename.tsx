import { DialogPrompt } from "@tui/ui/dialog-prompt"
import { useDialog } from "@tui/ui/dialog"
import { useStore } from "../context/store"
import { createMemo } from "solid-js"
import { useApi } from "../context/api"

interface DialogSessionRenameProps {
  session: string
}

export function DialogSessionRename(props: DialogSessionRenameProps) {
  const dialog = useDialog()
  const store = useStore()
  const api = useApi()
  const session = createMemo(() => store.state.sessions.get(props.session))

  return (
    <DialogPrompt
      title="Rename Session"
      value={session()?.title}
      onConfirm={(value) => {
        api.fosra.session.update({
          sessionID: props.session,
          title: value,
        })
        dialog.clear()
      }}
      onCancel={() => dialog.clear()}
    />
  )
}
