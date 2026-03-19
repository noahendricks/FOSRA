package ui

import (
	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

// ChatInput is the multi-line input area at the bottom of the chat.
// It renders just the prompt + textarea content; the surrounding border
// is drawn by the inputContainer in app.go.
type ChatInput struct {
	Area   textarea.Model
	styles Styles
	width  int
	height int
}

func NewChatInput(styles Styles) ChatInput {
	ta := createTextArea(nil)
	return ChatInput{
		Area:   ta,
		styles: styles,
		width:  80,
		height: InputMinHeight,
	}
}

func createTextArea(existing *textarea.Model) textarea.Model {
	ta := textarea.New()
	ta.Placeholder = "Ask anything..."
	ta.Prompt = " "
	ta.ShowLineNumbers = false
	ta.CharLimit = -1
	ta.SetHeight(InputMinHeight)

	s := ta.Styles()
	s.Focused.Base = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorFg))
	s.Blurred.Base = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorFg))
	s.Focused.CursorLine = lipgloss.NewStyle().Background(lipgloss.Color(colorBg))
	s.Blurred.CursorLine = lipgloss.NewStyle().Background(lipgloss.Color(colorBg))
	s.Focused.Placeholder = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorComment))
	s.Blurred.Placeholder = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorComment))
	s.Focused.Text = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorFg))
	s.Blurred.Text = lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Foreground(lipgloss.Color(colorFg))
	s.Cursor.Color = lipgloss.Color(colorCyan)
	s.Cursor.Blink = true
	ta.SetStyles(s)

	if existing != nil {
		ta.SetValue(existing.Value())
		ta.SetWidth(existing.Width())
		ta.SetHeight(existing.Height())
	}

	ta.Focus()
	return ta
}

// SetSize sets both the outer width and height for the input.
// The textarea width is reduced by 3 to account for the ">" prompt
// (1 char left-pad + 1 char ">" + 1 char space).
func (i *ChatInput) SetSize(w, h int) {
	i.width = w
	i.height = max(InputMinHeight, h)
	i.Area.SetWidth(max(1, w-3))
	i.Area.SetHeight(i.height)
}

// Update forwards messages to the textarea.
func (i *ChatInput) Update(msg tea.Msg) tea.Cmd {
	var cmd tea.Cmd
	i.Area, cmd = i.Area.Update(msg)
	return cmd
}

func (i *ChatInput) Value() string {
	return i.Area.Value()
}

func (i *ChatInput) Reset() {
	i.Area.Reset()
	i.Area.SetHeight(i.height)
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

// View renders the prompt + textarea content. No border — the container
// in app.go handles that.
func (i *ChatInput) View() string {
	prompt := i.styles.InputPrompt.Render(">")
	return lipgloss.JoinHorizontal(lipgloss.Top, prompt, i.Area.View())
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
