package ui

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ChatPane renders the scrollable message area using a viewport bubble.
// It supports user, assistant, system messages plus tool calls, thinking
// indicators, todo lists, code blocks, and streaming tokens.
type ChatPane struct {
	styles   Styles
	viewport viewport.Model
	width    int
	height   int
	spinner  Spinner

	// track content so we know when to re-render
	lastMsgCount int
	wasStreaming bool
}

func NewChatPane(styles Styles) ChatPane {
	vp := viewport.New(
		viewport.WithWidth(80),
		viewport.WithHeight(20),
	)
	vp.MouseWheelEnabled = true
	vp.SoftWrap = true

	return ChatPane{
		styles:   styles,
		viewport: vp,
		spinner:  NewSpinner(styles.Spinner),
	}
}

func (c *ChatPane) SetSize(w, h int) {
	c.width = w
	c.height = h
	c.viewport.SetWidth(w - 2) // padding
	c.viewport.SetHeight(h)
}

// update forwards messages to the viewport for scroll/mouse handling.
func (c *ChatPane) Update(msg tea.Msg) tea.Cmd {
	var cmd tea.Cmd
	c.viewport, cmd = c.viewport.Update(msg)
	return cmd
}

// SetMessages renders all messages into the viewport content.
// call this whenever messages change (new message, streaming token, etc.).
func (c *ChatPane) SetMessages(messages []session.Message) {
	content := c.renderAll(messages)
	c.viewport.SetContent(content)

	// auto-scroll to bottom on new messages or while streaming
	isStreaming := len(messages) > 0 && messages[len(messages)-1].IsStreaming
	newMessage := len(messages) != c.lastMsgCount

	if newMessage || isStreaming {
		c.viewport.GotoBottom()
	}

	c.lastMsgCount = len(messages)
	c.wasStreaming = isStreaming
}

func (c *ChatPane) View() string {
	return c.styles.ChatPane.
		Width(c.width).
		Height(c.height).
		Render(c.viewport.View())
}

// ViewEmpty renders the empty state placeholder.
func (c *ChatPane) ViewEmpty() string {
	placeholder := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true).
		Width(c.width - 4).
		Align(lipgloss.Center).
		AlignVertical(lipgloss.Center).
		Height(c.height).
		Render("No messages yet. Start a conversation below.")

	return c.styles.ChatPane.
		Width(c.width).
		Height(c.height).
		Render(placeholder)
}

// ── Rendering pipeline ────────────────────────────────────────────────

func (c *ChatPane) renderAll(messages []session.Message) string {
	if len(messages) == 0 {
		return ""
	}

	contentW := c.width - 4
	var blocks []string

	for _, msg := range messages {
		rendered := c.renderMessage(msg, contentW)
		blocks = append(blocks, rendered)
	}

	return strings.Join(blocks, "\n\n")
}

func (c *ChatPane) renderMessage(msg session.Message, contentW int) string {
	switch msg.Role {
	case session.RoleUser:
		return c.renderUser(msg, contentW)
	case session.RoleAssistant:
		return c.renderAssistant(msg, contentW)
	case session.RoleSystem:
		return c.renderSystem(msg, contentW)
	default:
		return msg.Content
	}
}

// ── User message ──────────────────────────────────────────────────────

func (c *ChatPane) renderUser(msg session.Message, w int) string {
	label := c.styles.MessageUser.Render("you")
	body := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Width(w).
		Render(msg.Content)
	return label + "\n" + body
}

// ── ASSISTANT MESSAGE ─────────────────────────────────────────────────

func (c *ChatPane) renderAssistant(msg session.Message, w int) string {
	var parts []string

	// error takes priority
	if msg.Error != "" {
		parts = append(parts, c.styles.MessageErr.Render("Error: "+msg.Error))
		return strings.Join(parts, "\n")
	}

	// label
	parts = append(parts, c.styles.MessageMeta.Render("assistant"))

	// thinking indicator (shown before content when model is reasoning)
	if msg.ThinkingMs > 0 {
		parts = append(parts, c.renderThinking(msg.ThinkingMs))
	}

	// tool calls (rendered before the main content body)
	for _, tc := range msg.ToolCalls {
		parts = append(parts, c.renderToolCall(tc, w))
	}

	// main content body
	if msg.Content != "" {
		body := c.styles.MessageAI.Width(w).Render(msg.Content)
		parts = append(parts, body)
	}

	// streaming cursor
	if msg.IsStreaming {
		indicator := c.spinner.View() + " " + c.styles.Streaming.Render("generating...")
		parts = append(parts, indicator)
	}

	// todo list
	if len(msg.Todos) > 0 {
		parts = append(parts, c.renderTodos(msg.Todos, w))
	}

	// source citations
	if len(msg.Sources) > 0 {
		parts = append(parts, c.renderSources(msg.Sources))
	}

	return strings.Join(parts, "\n")
}

// ── SYSTEM MESSAGE ────────────────────────────────────────────────────

func (c *ChatPane) renderSystem(msg session.Message, w int) string {
	return c.styles.MessageMeta.Width(w).Render("system: " + msg.Content)
}

// ── TOOL CALL BLOCK ───────────────────────────────────────────────────
// Renders like:
//   ✓ Tool  search_codebase
//   │ Searching for: "auth middleware"
//   │ Found 3 results

func (c *ChatPane) renderToolCall(tc session.ToolCall, w int) string {
	// Status icon
	var icon string
	switch tc.Status {
	case "done":
		icon = c.styles.ToolCallCheck.Render("✓")
	case "error":
		icon = c.styles.MessageErr.Render("✗")
	default:
		icon = c.spinner.View()
	}

	// header line: icon + "Tool" label + name + args
	header := icon + " " + c.styles.ToolCallHeader.Render(tc.Name)
	if tc.Args != "" {
		header += " " + c.styles.ToolCallName.Render(tc.Args)
	}

	if tc.Output == "" {
		return header
	}

	// Output block with left border
	outputW := w - 6
	if outputW < 20 {
		outputW = 20
	}

	output := c.styles.ToolCallOutput.
		Width(outputW).
		Render(tc.Output)

	return header + "\n" + output
}

// ── THINKING INDICATOR ────────────────────────────────────────────────

func (c *ChatPane) renderThinking(ms int) string {
	seconds := float64(ms) / 1000.0

	label := fmt.Sprintf("Thought for %.0fs", seconds)
	return "  " + c.styles.ThinkingLabel.Render(label)
}

// ── TODO / TASK LIST ──────────────────────────────────────────────────
// renders:
//   ╭──────────────────────────────╮
//   │ ✓ Search codebase            │
//   │ ● Analyze middleware chain   │
//   │ ○ Summarize findings         │
//   ╰──────────────────────────────╯

func (c *ChatPane) renderTodos(todos []session.TodoItem, w int) string {
	var lines []string

	for _, t := range todos {
		var line string
		switch t.Status {
		case session.TodoStatusDone:
			line = c.styles.TodoDone.Render("✓") + " " + c.styles.TodoDone.Render(t.Text)
		case session.TodoStatusInProgress:
			line = c.styles.TodoActive.Render("●") + " " + c.styles.TodoActive.Render(t.Text)
		default:
			line = c.styles.TodoPending.Render("○") + " " + c.styles.TodoPending.Render(t.Text)
		}
		lines = append(lines, line)
	}

	content := strings.Join(lines, "\n")

	todoW := w - 6
	if todoW < 20 {
		todoW = 20
	}

	return c.styles.TodoBlock.
		Width(todoW).
		Render(content)
}

// ── SOURCE CITATIONS ──────────────────────────────────────────────────

func (c *ChatPane) renderSources(sources []session.Source) string {
	var chips []string
	for _, src := range sources {
		score := fmt.Sprintf("%.0f%%", src.Score*100)
		label := src.DocName + " " + c.styles.SourceScore.Render(score)
		chips = append(chips, c.styles.SourceChip.Render(label))
	}
	return "  " + strings.Join(chips, " ")
}

// TickSpinner advances the spinner frame. Call from App on AnimTickMsg.
func (c *ChatPane) TickSpinner() {
	c.spinner.Tick()
}
