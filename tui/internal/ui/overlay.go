package ui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// OverlayView tracks which screen the command palette is showing.
type OverlayView int

const (
	OverlayCommands OverlayView = iota
	OverlaySessions
)

// Command is a single entry in the command palette.
type Command struct {
	Name string
	Key  string
	ID   string
}

// DefaultCommands returns the built-in command list with updated key hints.
func DefaultCommands() []Command {
	return []Command{
		{Name: "Sessions", Key: "ctrl+s", ID: "sessions"},
		{Name: "New Session", Key: "ctrl+n", ID: "new_session"},
		{Name: "Toggle Sidebar", Key: "ctrl+b", ID: "toggle_sidebar"},
		{Name: "Toggle RAG", Key: "ctrl+r", ID: "toggle_rag"},
		{Name: "Attach Document", Key: "ctrl+a", ID: "attach_doc"},
		{Name: "Quit", Key: "ctrl+c", ID: "quit"},
	}
}

// CommandPalette is a centered overlay with two views:
// commands list and sessions list.
type CommandPalette struct {
	styles   Styles
	view     OverlayView
	cursor   int
	filter   string
	width    int
	height   int
	commands []Command
}

func NewCommandPalette(styles Styles) CommandPalette {
	return CommandPalette{
		styles:   styles,
		commands: DefaultCommands(),
	}
}

func (p *CommandPalette) SetSize(w, h int) {
	p.width = w
	p.height = h
}

func (p *CommandPalette) Reset() {
	p.view = OverlayCommands
	p.cursor = 0
	p.filter = ""
}

func (p *CommandPalette) ShowSessions() {
	p.view = OverlaySessions
	p.cursor = 0
}

func (p *CommandPalette) ShowCommands() {
	p.view = OverlayCommands
	p.cursor = 0
}

func (p *CommandPalette) CurrentView() OverlayView { return p.view }

func (p *CommandPalette) CursorUp(n int) {
	if n == 0 {
		return
	}
	p.cursor--
	if p.cursor < 0 {
		p.cursor = n - 1
	}
}

func (p *CommandPalette) CursorDown(n int) {
	if n == 0 {
		return
	}
	p.cursor = (p.cursor + 1) % n
}

func (p *CommandPalette) Cursor() int { return p.cursor }

func (p *CommandPalette) SelectedCommand() Command {
	if p.cursor < 0 || p.cursor >= len(p.commands) {
		return Command{}
	}
	return p.commands[p.cursor]
}

func (p *CommandPalette) SelectedSessionID(sessions []*session.Session) string {
	if p.cursor < 0 || p.cursor >= len(sessions) {
		return ""
	}
	return sessions[p.cursor].ID
}

// View renders the overlay content (positioning is handled by app.go).
func (p *CommandPalette) View(sessions []*session.Session, activeID string) string {
	boxW := p.width / 3
	if boxW < 40 {
		boxW = 40
	}
	// innerW accounts for overlay Padding(1,2) = 4 horizontal + border = 2
	innerW := boxW - 6

	switch p.view {
	case OverlaySessions:
		return p.renderSessions(sessions, activeID, boxW, innerW)
	default:
		return p.renderCommands(boxW, innerW)
	}
}

func (p *CommandPalette) renderCommands(boxW, innerW int) string {
	title := p.styles.OverlayTitle.Render("Commands")

	var rows []string
	rows = append(rows, title)

	for i, cmd := range p.commands {
		keyHint := p.styles.CommandKey.Render(cmd.Key)
		keyW := lipgloss.Width(cmd.Key)

		// available width for the name: innerW minus key hint minus gap
		nameW := innerW - keyW - 3
		if nameW < 8 {
			nameW = 8
		}

		name := cmd.Name
		if len(name) > nameW {
			name = name[:nameW-1] + "…"
		}
		padded := fmt.Sprintf("%-*s", nameW, name)

		line := " " + padded + " " + keyHint

		if i == p.cursor {
			rows = append(rows, p.styles.SessionActive.Render(line))
		} else {
			rows = append(rows, p.styles.CommandItem.Render(line))
		}
	}

	hint := p.styles.HelpDesc.Render("  ↑↓ navigate · enter select · esc close")
	rows = append(rows, "")
	rows = append(rows, hint)

	content := strings.Join(rows, "\n")
	return p.styles.Overlay.Width(boxW).Render(content)
}

func (p *CommandPalette) renderSessions(sessions []*session.Session, activeID string, boxW, innerW int) string {
	title := p.styles.OverlayTitle.Render("Sessions")

	var rows []string
	rows = append(rows, title)

	if len(sessions) == 0 {
		empty := lipgloss.NewStyle().
			Foreground(lipgloss.Color(colorComment)).
			Italic(true).
			Render("  No sessions")
		rows = append(rows, empty)
	} else {
		for i, sess := range sessions {
			dot := session.InactiveDot
			if sess.ID == activeID {
				dot = session.ActiveDot
			}

			name := truncate(sess.Title, innerW-6)
			line := " " + dot + " " + name

			if i == p.cursor {
				rows = append(rows, p.styles.SessionActive.Render(line))
			} else {
				rows = append(rows, p.styles.SessionItem.Render(line))
			}
		}
	}

	hint := p.styles.HelpDesc.Render("  ↑↓ navigate · enter select · backspace back · esc close")
	rows = append(rows, "")
	rows = append(rows, hint)

	content := strings.Join(rows, "\n")
	return p.styles.Overlay.Width(boxW).Render(content)
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
