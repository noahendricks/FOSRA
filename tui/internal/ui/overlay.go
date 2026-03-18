package ui

import (
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/exp/charmtone"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type SessionOverlay struct {
	styles Styles
	cursor int
	width  int
	height int
	active bool
}

func NewSessionOverlay(styles Styles) SessionOverlay {
	return SessionOverlay{
		styles: styles,
	}
}

func (o *SessionOverlay) SetSize(w, h int) {
	o.width = w
	o.height = h
}

func (o *SessionOverlay) Open()   { o.active = true }
func (o *SessionOverlay) Close()  { o.active = false }
func (o *SessionOverlay) Toggle() { o.active = !o.active }

func (o *SessionOverlay) CursorUp(n int) {
	o.cursor--
	if o.cursor < 0 {
		o.cursor = n - 1
	}
}

func (o *SessionOverlay) CursorDown(n int) {
	o.cursor = (o.cursor + 1) % n
}

func (o *SessionOverlay) SelectedID(sessions []*session.Session) string {
	if o.cursor < 0 || o.cursor >= len(sessions) {
		return ""
	}
	return sessions[o.cursor].ID
}

func (o *SessionOverlay) View() string {
	box := o.styles.Overlay.
		Width(o.width / 3).
		Height(o.height / 2).
		Align(lipgloss.Center).
		BorderForeground(charmtone.Pickle).
		Render()

	return lipgloss.NewStyle().Render(box)
}

func truncate(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
