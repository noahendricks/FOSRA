package ui

import (
	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
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
	ta.ShowLineNumbers = false
	ta.SetHeight(InputMinHeight)
	ta.CharLimit = 0 // no limit
	ta.Focus()

	// Style the textarea to blend with the input pane
	s := ta.Styles()
	s.Focused.Base = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFg))
	s.Blurred.Base = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
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
	// textarea width minus padding and prompt
	taW := w - 6
	if taW < 20 {
		taW = 20
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

func (i *ChatInput) View(ragEnabled bool, isStreaming bool) string {
	// Prompt indicator
	indicator := "›"
	if isStreaming {
		indicator = "…"
	}

	prompt := i.styles.InputPrompt.Render(indicator)

	// RAG badge
	var ragBadge string
	if ragEnabled {
		ragBadge = i.styles.InputRAG.Render("RAG") + " "
	}

	// compose: RAG badge + prompt + textarea
	inner := lipgloss.JoinHorizontal(lipgloss.Top,
		ragBadge,
		prompt,
		" ",
		i.Area.View(),
	)

	// pick style based on focus
	paneStyle := i.styles.InputPane
	if i.Area.Focused() {
		paneStyle = i.styles.InputFocused
	}

	return paneStyle.
		Width(i.width).
		Render(inner)
}
