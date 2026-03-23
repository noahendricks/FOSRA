// Minimal test TUI entry point - verifies OpenTUI setup works
import { render, useKeyboard } from "@opentui/solid"
import { createSignal, onCleanup } from "solid-js"

function App() {
  const [count, setCount] = createSignal(0)
  
  useKeyboard((key) => {
    if (key.name === "q" || (key.name === "c" && key.ctrl)) {
      process.exit(0)
    }
    if (key.name === "+") {
      setCount(c => c + 1)
    }
  })

  return (
    <box flexDirection="column" padding={2} gap={1}>
      <text fg="#7aa2f7"><strong>FOSRA TUI</strong></text>
      <text fg="#c0caf5">OpenTUI setup verified!</text>
      <text fg="#9ece6a">Count: {count()}</text>
      <box flexDirection="row" gap={2}>
        <text fg="#ff9e64">Press + to increment</text>
        <text fg="#565f89">Press q to quit</text>
      </box>
    </box>
  )
}

render(() => <App />)