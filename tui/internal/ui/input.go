package ui

import (
	"strings"

	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

// ChatInput is the multi-line input area at the bottom of the chat.
// It renders the textarea + footer; the surrounding border
// is drawn by the inputContainer in app.go.
type ChatInput struct {
	Area     textarea.Model
	styles   Styles
	width    int
	height   int
	agent    string
	model    string
	provider string
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
	ta.Prompt = ""
	ta.ShowLineNumbers = false
	ta.CharLimit = -1
	ta.SetHeight(InputMinHeight)

	s := ta.Styles()
	s.Focused.Base = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorFg))
	s.Blurred.Base = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorFg))
	s.Focused.CursorLine = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight))
	s.Blurred.CursorLine = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight))
	s.Focused.Placeholder = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorComment))
	s.Blurred.Placeholder = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorComment))
	s.Focused.Text = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorFg))
	s.Blurred.Text = lipgloss.NewStyle().Background(lipgloss.Color(colorBgHighlight)).Foreground(lipgloss.Color(colorFg))
	s.Cursor.Color = lipgloss.Color(colorInfo)
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

// SetModelInfo sets the agent, model, and provider for the footer display.
func (i *ChatInput) SetModelInfo(agent, model, provider string) {
	i.agent = agent
	i.model = model
	i.provider = provider
}

// SetSize sets both the outer width and height for the input.
func (i *ChatInput) SetSize(w, h int) {
	i.width = w
	i.height = max(InputMinHeight, h)
	i.Area.SetWidth(max(1, w-4))
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

// View renders the textarea + footer. No border — the container
// in app.go handles that.
func (i *ChatInput) View() string {
	textareaView := i.Area.View()

	footer := i.renderFooter()

	return lipgloss.JoinVertical(lipgloss.Left,
		textareaView,
		footer,
	)
}

func (i *ChatInput) renderFooter() string {
	agentPart := i.styles.InputFooterAgent.Render(i.agent)
	modelPart := i.styles.InputFooterModel.Render(i.model)
	providerPart := i.styles.InputFooterProvider.Render(i.provider)

	sep := i.styles.InputFooterProvider.Render("·")

	parts := []string{agentPart, modelPart}
	if i.provider != "" {
		parts = append(parts, providerPart)
	}

	footerContent := strings.Join(parts, " "+sep+" ")

	return i.styles.InputFooter.Render(footerContent)
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
