package ui

import (
	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ChatInput is the multi-line input area at the bottom of the chat.
// Uses textarea for shift+enter newlines, enter to send.
type ChatInput struct {
	Area   textarea.Model
	styles Styles
	width  int
}

func NewChatInput(styles Styles) ChatInput {
	ta := textarea.New()
	ta.Placeholder = "Ask anything..."
	ta.Prompt = " "
	ta.ShowLineNumbers = false
	ta.SetHeight(InputMinHeight)
	ta.CharLimit = -1
	ta.Focus()

	// Style the textarea to blend with the main background
	s := ta.Styles()
	s.Focused.Base = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))
	s.Blurred.Base = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFgDim))
	s.Focused.Text = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))
	s.Blurred.Text = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim))
	s.Focused.Placeholder = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))
	s.Blurred.Placeholder = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))
	s.Focused.CursorLine = lipgloss.NewStyle()
	s.Blurred.CursorLine = lipgloss.NewStyle()
	s.Cursor.Color = lipgloss.Color(colorCyan)
	s.Cursor.Blink = true
	ta.SetStyles(s)

	return ChatInput{
		Area:   ta,
		styles: styles,
		width:  80,
	}
}

func (i *ChatInput) SetWidth(w int) {
	i.width = w
	taW := w - 2
	if taW < 1 {
		taW = 1
	}
	i.Area.SetWidth(taW)
}

// Update forwards messages to the textarea.
func (i *ChatInput) Update(msg tea.Msg) tea.Cmd {
	var cmd tea.Cmd
	i.Area, cmd = i.Area.Update(msg)
	return cmd
}

// value returns the current input text.
func (i *ChatInput) Value() string {
	return i.Area.Value()
}

// reset clears the input.
func (i *ChatInput) Reset() {
	i.Area.Reset()
}

func (i *ChatInput) Focus() tea.Cmd {
	return i.Area.Focus()
}

func (i *ChatInput) Blur() {
	i.Area.Blur()
}

func (i *ChatInput) Focused() bool {
	return i.Area.Focused()
}

func (i *ChatInput) View(sess *session.Session) string {
	prompt := i.styles.InputPrompt.Render(">")
	prefix := i.renderPrefix(sess, prompt)

	taW := i.textareaWidth(prefix)
	i.Area.SetWidth(taW)

	inner := lipgloss.JoinHorizontal(lipgloss.Top, prefix, " ", i.Area.View())

	// pick style based on focus
	paneStyle := i.styles.InputPane
	if i.Area.Focused() {
		paneStyle = i.styles.InputFocused
	}

	return paneStyle.
		Width(i.width).
		Render(inner)
}

func (i *ChatInput) renderPrefix(_ *session.Session, prompt string) string {
	// Clean input: just the ">" prompt. Model/context info lives in the status bar.
	return prompt
}

func (i *ChatInput) textareaWidth(prefix string) int {
	taW := i.width - lipgloss.Width(prefix) - 3
	if taW < 1 {
		return 1
	}
	return taW
}

func truncateInputLabel(s string, limit int) string {
	if limit <= 1 {
		return "…"
	}
	if len(s) <= limit {
		return s
	}
	return s[:limit-1] + "…"
}
