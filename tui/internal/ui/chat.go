package ui

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

const maxToolOutputLines = 10

// ChatPane renders the scrollable message area using a viewport bubble.
type ChatPane struct {
	styles   Styles
	viewport viewport.Model
	width    int
	height   int
	spinner  Spinner

	lastMsgCount int
	wasStreaming bool
}

func NewChatPane(styles Styles) ChatPane {
	vp := viewport.New(
		viewport.WithWidth(80),
		viewport.WithHeight(20),
	)
	vp.MouseWheelEnabled = true
	vp.SoftWrap = false

	return ChatPane{
		styles:   styles,
		viewport: vp,
		spinner:  NewSpinner(styles.Spinner),
	}
}

func (c *ChatPane) SetSize(w, h int) {
	c.width = w
	c.height = h
	c.viewport.SetWidth(w - ChatPadding*2)
	c.viewport.SetHeight(h)
}

func (c *ChatPane) Update(msg tea.Msg) tea.Cmd {
	var cmd tea.Cmd
	c.viewport, cmd = c.viewport.Update(msg)
	return cmd
}

// SetMessages renders all messages into the viewport content.
func (c *ChatPane) SetMessages(messages []session.Message) {
	wasAtBottom := c.viewport.AtBottom() || c.viewport.TotalLineCount() <= c.viewport.VisibleLineCount()

	// manage spinner state based on streaming
	isStreaming := len(messages) > 0 && messages[len(messages)-1].IsStreaming
	if isStreaming && !c.spinner.Running() {
		c.spinner.Start()
	} else if !isStreaming && c.spinner.Running() {
		c.spinner.Stop()
	}

	content := c.renderAll(messages)
	c.viewport.SetContent(content)

	// auto-scroll on new messages or while streaming
	newMessage := len(messages) != c.lastMsgCount
	if (newMessage || isStreaming) && wasAtBottom {
		c.viewport.GotoBottom()
	}

	c.lastMsgCount = len(messages)
	c.wasStreaming = isStreaming
}

func (c *ChatPane) View() string {
	content := lipgloss.NewStyle().
		PaddingLeft(ChatPadding).
		Render(c.viewport.View())

	return c.styles.ChatPane.
		Width(c.width).
		Height(c.height).
		Render(content)
}

// ViewEmpty renders the empty state placeholder.
func (c *ChatPane) ViewEmpty() string {
	placeholder := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true).
		Width(c.width - ChatPadding*2).
		Align(lipgloss.Center).
		AlignVertical(lipgloss.Center).
		Height(c.height).
		Render("No messages yet. Start a conversation below.")

	content := lipgloss.NewStyle().
		PaddingLeft(ChatPadding).
		Render(placeholder)

	return c.styles.ChatPane.
		Width(c.width).
		Height(c.height).
		Render(content)
}

// ── Rendering pipeline ────────────────────────────────────────────────

func (c *ChatPane) renderAll(messages []session.Message) string {
	if len(messages) == 0 {
		return ""
	}

	// content width with left/right padding (gutter)
	contentW := c.width - ChatPadding*2 - 2
	if contentW > 92 {
		contentW = 92
	}
	if contentW < 30 {
		contentW = 30
	}

	var blocks []string
	for _, msg := range messages {
		rendered := c.renderMessage(msg, contentW)
		blocks = append(blocks, rendered)
	}

	return strings.Join(blocks, "\n\n")
}

func (c *ChatPane) renderMessage(msg session.Message, w int) string {
	switch msg.Role {
	case session.RoleUser:
		return c.renderUser(msg, w)
	case session.RoleAssistant:
		return c.renderAssistant(msg, w)
	case session.RoleSystem:
		return c.renderSystem(msg, w)
	default:
		return msg.Content
	}
}

func (c *ChatPane) renderUser(msg session.Message, w int) string {
	// thick purple left border on the body
	bodyW := w - 3
	if bodyW < 20 {
		bodyW = 20
	}

	body := c.styles.UserBlock.
		Width(bodyW).
		Render(msg.Content)

	return body
}

func (c *ChatPane) renderAssistant(msg session.Message, w int) string {
	var parts []string
	blockW := messageBlockWidth(c.styles.AssistantBlock, w)

	// error takes priority
	if msg.Error != "" {
		parts = append(parts, c.styles.MessageErr.Width(blockW).Render("Error: "+msg.Error))
		return c.styles.AssistantBlock.Width(w).Render(strings.Join(parts, "\n"))
	}

	// thinking indicator
	if msg.ThinkingMs > 0 {
		parts = append(parts, c.renderThinking(msg.ThinkingMs))
	}

	// tool calls (rendered before the main content body)
	for _, tc := range msg.ToolCalls {
		parts = append(parts, c.renderToolCall(tc, w))
	}

	// todo list (rendered before or after content depending on pipeline stage)
	if len(msg.Todos) > 0 {
		parts = append(parts, c.renderTodos(msg.Todos))
	}

	// main content body
	if msg.Content != "" {
		body := c.renderAssistantBody(msg, w)
		parts = append(parts, body)
	}

	// streaming indicator with spinner + blinking cursor
	if msg.IsStreaming {
		spinnerView := c.spinner.View()
		if spinnerView == "" {
			spinnerView = "⠋" // fallback
		}
		indicator := spinnerView + " " + c.styles.Streaming.Render("generating...")
		parts = append(parts, indicator)
	}

	// source citations at end
	if len(msg.Sources) > 0 {
		parts = append(parts, c.renderSources(msg.Sources))
	}

	return c.styles.AssistantBlock.Width(w).Render(strings.Join(parts, "\n"))
}

func (c *ChatPane) renderAssistantBody(msg session.Message, w int) string {
	bodyW := w
	if bodyW > 88 {
		bodyW = 88
	}
	if bodyW < 20 {
		bodyW = 20
	}

	if msg.IsStreaming {
		return c.styles.MessageAI.Width(bodyW).Render(msg.Content)
	}

	rendered := renderMarkdown(msg.Content, bodyW)
	if rendered == "" {
		return c.styles.MessageAI.Width(bodyW).Render(msg.Content)
	}

	return c.styles.MessageAI.Width(bodyW).Render(rendered)
}

// ── System message ────────────────────────────────────────────────────

func (c *ChatPane) renderSystem(msg session.Message, w int) string {
	return c.styles.MessageMeta.Width(w).Render("system: " + msg.Content)
}

// ── Tool call block ──────────────────────────────────
//
// Renders:
//   ✱ Grep "pattern" in . (9 matches)
//   │ matched output lines...
//
//   ✓ Read internal/ui/chat.go
//   │ file content...
//
//   $ go build ./...

func (c *ChatPane) renderToolCall(tc session.ToolCall, w int) string {
	// status icon
	var icon string
	switch tc.Status {
	case "done":
		icon = c.styles.ToolCallCheck.Render("✓")
	case "error":
		icon = c.styles.MessageErr.Render("✗")
	default: // running
		sv := c.spinner.View()
		if sv == "" {
			sv = c.styles.ToolCallIcon.Render("✱")
		}
		icon = sv
	}

	// header: icon + tool name + args
	header := icon + " " + c.styles.ToolCallHeader.Render(tc.Name)
	if tc.Args != "" {
		header += " " + c.styles.ToolCallArgs.Render(tc.Args)
	}

	if tc.Output == "" {
		return header
	}

	// output with left border
	outputW := w - 6
	if outputW < 20 {
		outputW = 20
	}

	toolOutput := c.renderToolOutput(tc, outputW)
	output := c.styles.ToolCallOutput.
		Width(outputW).
		Render(toolOutput)

	return header + "\n" + output
}

func (c *ChatPane) renderToolOutput(tc session.ToolCall, w int) string {
	content := strings.TrimSpace(tc.Output)
	if content == "" {
		return ""
	}

	if tc.Status == "error" {
		return c.styles.MessageErr.Render(truncateTextLines(content, maxToolOutputLines))
	}

	markdown := c.toolOutputMarkdown(tc, content)
	rendered := renderMarkdownWithBackground(markdown, w, colorBg)
	if rendered == "" {
		rendered = content
	}

	return truncateRenderedOutput(rendered, maxToolOutputLines)
}

func (c *ChatPane) toolOutputMarkdown(tc session.ToolCall, content string) string {
	name := strings.ToLower(strings.TrimSpace(tc.Name))
	trimmedArgs := strings.TrimSpace(tc.Args)

	switch name {
	case "bash", "run", "shell", "$":
		return fencedBlock("bash", content)
	case "read", "view":
		return fencedBlock(languageFromPath(trimmedArgs), content)
	case "fetch":
		return fencedBlock("markdown", content)
	case "grep", "search", "sources":
		return fencedBlock("text", content)
	default:
		if looksLikeCode(content) {
			return fencedBlock(languageFromPath(trimmedArgs), content)
		}
		return content
	}
}

func fencedBlock(language, content string) string {
	content = strings.TrimSpace(content)
	if content == "" {
		return ""
	}
	return "```" + language + "\n" + content + "\n```"
}

func languageFromPath(path string) string {
	path = strings.Trim(path, `"'`)
	if path == "" {
		return "text"
	}

	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(path), "."))
	switch ext {
	case "go", "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "md", "html", "css", "sh", "bash", "sql", "xml", "java", "rb", "rs", "c", "cpp", "h":
		return ext
	case "":
		return "text"
	default:
		return ext
	}
}

func looksLikeCode(content string) bool {
	codeSignals := []string{
		"func ",
		"package ",
		"import ",
		"class ",
		"const ",
		"let ",
		"var ",
		"return ",
		"{",
		"}",
		"=>",
	}

	for _, signal := range codeSignals {
		if strings.Contains(content, signal) {
			return true
		}
	}

	return strings.Contains(content, "\n\t")
}

// ── Thinking indicator ────────────────────────────────────────────────

func (c *ChatPane) renderThinking(ms int) string {
	seconds := float64(ms) / 1000.0
	label := fmt.Sprintf("thinking · %.1fs", seconds)
	return c.styles.ThinkingLabel.Render(label)
}

// ── Todo / task list  ────────────────────
//
// Renders:
//   # Todos
//   [✓] Search codebase
//   [●] Analyze middleware chain
//   [ ] Summarize findings

func (c *ChatPane) renderTodos(todos []session.TodoItem) string {
	var lines []string

	// header
	lines = append(lines, c.styles.TodoHeader.Render("# Todos"))
	lines = append(lines, "")

	for _, t := range todos {
		var line string
		switch t.Status {
		case session.TodoStatusDone:
			marker := c.styles.TodoDone.Render("[✓]")
			text := c.styles.TodoDone.Render(t.Text)
			line = marker + " " + text
		case session.TodoStatusInProgress:
			marker := c.styles.TodoActive.Render("[●]")
			text := c.styles.TodoActive.Render(t.Text)
			line = marker + " " + text
		default:
			marker := c.styles.TodoPending.Render("[ ]")
			text := c.styles.TodoPending.Render(t.Text)
			line = marker + " " + text
		}
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

// ── Source citations ──────────────────────────────────────────────────

func (c *ChatPane) renderSources(sources []session.Source) string {
	parts := make([]string, 0, len(sources)+1)
	parts = append(parts, c.styles.SidebarDim.Render("Sources:"))
	for _, src := range sources {
		score := fmt.Sprintf("%.0f%%", src.Score*100)
		label := src.DocName + " " + c.styles.SourceScore.Render(score)
		parts = append(parts, c.styles.SourceChip.Render(label))
	}
	return strings.Join(parts, " ")
}

func messageBlockWidth(style lipgloss.Style, width int) int {
	blockW := width
	if blockW > 92 {
		blockW = 92
	}
	if blockW < 20 {
		blockW = 20
	}

	innerW := blockW - style.GetHorizontalFrameSize()
	if innerW >= 20 {
		return innerW
	}

	return blockW
}

func truncateRenderedOutput(rendered string, maxLines int) string {
	lines := strings.Split(rendered, "\n")
	if len(lines) <= maxLines {
		return rendered
	}

	hidden := len(lines) - maxLines
	visible := append([]string{}, lines[:maxLines]...)
	visible = append(visible, lipgloss.NewStyle().Foreground(lipgloss.Color(colorComment)).Render("... +"+strconv.Itoa(hidden)+" more lines"))
	return strings.Join(visible, "\n")
}

func truncateTextLines(content string, maxLines int) string {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) <= maxLines {
		return strings.Join(lines, "\n")
	}

	hidden := len(lines) - maxLines
	visible := append([]string{}, lines[:maxLines]...)
	visible = append(visible, "... +"+strconv.Itoa(hidden)+" more lines")
	return strings.Join(visible, "\n")
}

// TickSpinner advances the spinner frame. Call from App on AnimTickMsg.
func (c *ChatPane) TickSpinner() {
	c.spinner.Tick()
}
