import { createMemo, Match, onCleanup, onMount, Show, Switch } from "solid-js"
import { useTheme } from "../../context/theme"
import { useStore } from "../../context/store"
import { useDirectory } from "../../context/directory"
import { useConnected } from "@tui/components/dialogs/dialog-model"
import { createStore } from "solid-js/store"
import { useRoute } from "../../context/route"

export function Footer() {
  const { theme } = useTheme()
  const appStore = useStore()
  const route = useRoute()
  const mcpEntries = createMemo(() => Object.entries(Object.fromEntries([...appStore.state.mcp.entries()])))
  const mcp = createMemo(() => mcpEntries().filter(([_, x]) => x.status === "connected").length)
  const mcpError = createMemo(() => mcpEntries().some(([_, x]) => x.status === "failed"))
  const lsp = createMemo(() => [])
  const permissions = createMemo(() => {
    if (route.data.type !== "session") return []
    return []
  })
  const directory = useDirectory()
  const connected = useConnected()

  const [footerStore, setFooterStore] = createStore({
    welcome: false,
  })

  onMount(() => {
    // Track all timeouts to ensure proper cleanup
    const timeouts: ReturnType<typeof setTimeout>[] = []

    function tick() {
      if (connected()) return
      if (!footerStore.welcome) {
        setFooterStore("welcome", true)
        timeouts.push(setTimeout(() => tick(), 5000))
        return
      }

      if (footerStore.welcome) {
        setFooterStore("welcome", false)
        timeouts.push(setTimeout(() => tick(), 10_000))
        return
      }
    }
    timeouts.push(setTimeout(() => tick(), 10_000))

    onCleanup(() => {
      timeouts.forEach(clearTimeout)
    })
  })

  return (
    <box flexDirection="row" justifyContent="space-between" gap={1} flexShrink={0}>
      <text fg={theme.textMuted}>{directory()}</text>
      <box gap={2} flexDirection="row" flexShrink={0}>
        <Switch>
          <Match when={footerStore.welcome}>
            <text fg={theme.text}>
              Get started <span style={{ fg: theme.textMuted }}>/connect</span>
            </text>
          </Match>
          <Match when={connected()}>
            <Show when={permissions().length > 0}>
              <text fg={theme.warning}>
                <span style={{ fg: theme.warning }}>△</span> {permissions().length} Permission
                {permissions().length > 1 ? "s" : ""}
              </text>
            </Show>
            <text fg={theme.text}>
              <span style={{ fg: lsp().length > 0 ? theme.success : theme.textMuted }}>•</span> {lsp().length} LSP
            </text>
            <Show when={mcp()}>
              <text fg={theme.text}>
                <Switch>
                  <Match when={mcpError()}>
                    <span style={{ fg: theme.error }}>⊙ </span>
                  </Match>
                  <Match when={true}>
                    <span style={{ fg: theme.success }}>⊙ </span>
                  </Match>
                </Switch>
                {mcp()} MCP
              </text>
            </Show>
            <text fg={theme.textMuted}>/status</text>
          </Match>
        </Switch>
      </box>
    </box>
  )
}
