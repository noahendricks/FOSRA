package ui

import (
	"charm.land/bubbles/v2/textinput"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/exp/charmtone"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type TextInput struct {
	input       textinput.Model
	session     *session.Session
	width       int
	ragEnabled  bool
	isStreaming bool
}

func NewTextInput(ragEnabled bool) TextInput {
	m := textinput.New()
	m.Placeholder = "Ask anything..."
	m.Focus()

	return TextInput{
		input:      m,
		width:      80,
		ragEnabled: ragEnabled,
	}
}

func (i *TextInput) SetWidth(w int) { i.width = w }

func (i *TextInput) ToggleFocus() {
	if i.input.Focused() {
		i.input.Blur()
	} else {
		i.input.Focus()
	}
}

func (i TextInput) View(ragEnabled bool, isStreaming bool) string {
	indicator := "›"
	indicatorColor := charmtone.Malibu

	if isStreaming {
		indicator = "…"
		indicatorColor = charmtone.Sardine
	}

	var ragBadge string
	if ragEnabled {
		ragBadge = lipgloss.NewStyle().
			Foreground(charmtone.Damson).
			Background(charmtone.Malibu).
			Padding(0, 1).
			MarginRight(1).
			Render("RAG")
	}

	prompt := lipgloss.NewStyle().
		Foreground(indicatorColor).
		MarginRight(1).
		Render(indicator)

	inputContent := i.input.View()

	inner := lipgloss.JoinHorizontal(lipgloss.Center,
		ragBadge,
		prompt,
		inputContent,
	)

	outerWidth := i.width - 2

	bar := lipgloss.NewStyle().
		Background(charmtone.Guppy).
		Foreground(charmtone.Bengal).
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		BorderForeground(charmtone.Thunder).
		Padding(0, 1).
		Width(outerWidth).
		Render(inner)

	return bar
}
