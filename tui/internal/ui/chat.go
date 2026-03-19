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
		Padding(0, ChatPadding).
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
		Width(c.viewport.Width()).
		Align(lipgloss.Center).
		AlignVertical(lipgloss.Center).
		Height(c.height).
		Render("No messages yet. Start a conversation below.")

	content := lipgloss.NewStyle().
		Padding(0, ChatPadding).
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

	contentW := c.viewport.Width()
	if contentW < 30 {
		contentW = 30
	}

	var blocks []string
	for _, msg := range messages {
		blocks = append(blocks, c.renderMessageBlocks(msg, contentW)...)
	}

	return strings.Join(blocks, "\n\n")
}

func (c *ChatPane) renderMessageBlocks(msg session.Message, w int) []string {
	switch msg.Role {
	case session.RoleUser:
		return []string{c.renderUser(msg, w)}
	case session.RoleAssistant:
		return c.renderAssistantBlocks(msg, w)
	case session.RoleSystem:
		return []string{c.renderSystem(msg, w)}
	default:
		return []string{msg.Content}
	}
}

func (c *ChatPane) renderUser(msg session.Message, w int) string {
	return c.styles.UserBlock.Width(w).Render(msg.Content)
}

func (c *ChatPane) renderAssistantBlocks(msg session.Message, w int) []string {
	var blocks []string

	if primary := c.renderAssistantPrimaryBlock(msg, w); primary != "" {
		blocks = append(blocks, primary)
	}

	if len(msg.Todos) > 0 {
		blocks = append(blocks, c.renderTodoBlock(msg.Todos, w))
	}

	if len(msg.Sources) > 0 {
		blocks = append(blocks, c.renderSourcesBlock(msg.Sources, w))
	}

	for _, tc := range msg.ToolCalls {
		blocks = append(blocks, c.renderToolCallBlock(tc, w))
	}

	return blocks
}

func (c *ChatPane) renderAssistantPrimaryBlock(msg session.Message, w int) string {
	var parts []string
	blockW := messageBlockWidth(c.styles.AssistantBlock, w)

	if msg.ThinkingMs > 0 {
		parts = append(parts, c.renderThinking(msg.ThinkingMs))
	}

	if msg.Error != "" {
		parts = append(parts, c.styles.MessageErr.Width(blockW).Render("Error: "+msg.Error))
	}

	if msg.Content != "" {
		parts = append(parts, c.renderAssistantBody(msg, blockW))
	}

	if msg.IsStreaming {
		spinnerView := c.spinner.View()
		if spinnerView == "" {
			spinnerView = "⠋"
		}
		parts = append(parts, c.styles.Streaming.Width(blockW).Render(spinnerView+" generating..."))
	}

	if len(parts) == 0 {
		return ""
	}

	return c.styles.AssistantBlock.Width(w).Render(lipgloss.JoinVertical(lipgloss.Left, parts...))
}

func (c *ChatPane) renderAssistantBody(msg session.Message, bodyW int) string {
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

func (c *ChatPane) renderToolCallBlock(tc session.ToolCall, w int) string {
	blockW := messageBlockWidth(c.styles.ToolCallBlock, w)

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
	parts := []string{lipgloss.NewStyle().Width(blockW).Render(header)}

	if tc.Output != "" {
		toolOutput := c.renderToolOutput(tc, blockW)
		parts = append(parts, c.styles.ToolCallOutput.Width(blockW).Render(toolOutput))
	}

	if tc.Output == "" && tc.Status != "done" {
		parts = append(parts, c.styles.ToolCallStatus.Width(blockW).Render(tc.Status))
	}

	return c.styles.ToolCallBlock.Width(w).Render(lipgloss.JoinVertical(lipgloss.Left, parts...))
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

func (c *ChatPane) renderTodoBlock(todos []session.TodoItem, w int) string {
	blockW := messageBlockWidth(c.styles.ToolCallBlock, w)
	content := lipgloss.NewStyle().Width(blockW).Render(c.renderTodos(todos))
	return c.styles.ToolCallBlock.Width(w).Render(content)
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

func (c *ChatPane) renderSourcesBlock(sources []session.Source, w int) string {
	blockW := messageBlockWidth(c.styles.ToolCallBlock, w)
	content := lipgloss.NewStyle().Width(blockW).Render(c.renderSources(sources))
	return c.styles.ToolCallBlock.Width(w).Render(content)
}

func messageBlockWidth(style lipgloss.Style, width int) int {
	blockW := width
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
