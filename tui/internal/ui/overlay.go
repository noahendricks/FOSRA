package ui

import (
	"fmt"
	"strings"

	"github.com/roccoluxe/fosra-tui/tui/internal/session"

	"github.com/charmbracelet/lipgloss"
)

// SessionOverlay is the slide-in panel listing conversations.
type SessionOverlay struct {
	styles Styles
	anim   OverlayAnim
	cursor int // currently highlighted session index
	width  int
	height int
}

func NewSessionOverlay(styles Styles) SessionOverlay {
	return SessionOverlay{
		styles: styles,
		anim:   NewOverlayAnim(),
	}
}

func (o *SessionOverlay) SetSize(w, h int) {
	o.width = w
	o.height = h
}

func (o *SessionOverlay) Toggle()      { o.anim.Toggle() }
func (o *SessionOverlay) Open()        { o.anim.Open() }
func (o *SessionOverlay) Close()       { o.anim.Close() }
func (o *SessionOverlay) IsOpen() bool { return o.anim.IsOpen() }
func (o *SessionOverlay) AtRest() bool { return o.anim.AtRest() }

// Step advances the spring animation.
func (o *SessionOverlay) Step() { o.anim.Step() }

func (o *SessionOverlay) CursorUp(n int) {
	o.cursor--
	if o.cursor < 0 {
		o.cursor = n - 1
	}
}

func (o *SessionOverlay) CursorDown(n int) {
	o.cursor = (o.cursor + 1) % n
}

// SelectedID returns the session ID at the cursor.
func (o *SessionOverlay) SelectedID(sessions []*session.Session) string {
	if o.cursor < 0 || o.cursor >= len(sessions) {
		return ""
	}
	return sessions[o.cursor].ID
}

// View renders the overlay, offset by spring progress.
// Returns an empty string when fully closed and at rest.
func (o *SessionOverlay) View(mgr *session.Manager) string {
	progress := o.anim.Progress()
	if progress < 0.01 {
		return ""
	}

	overlayW := o.width / 3
	if overlayW < 30 {
		overlayW = 30
	}
	overlayH := o.height - 4

	// Vertical offset for slide-in (slides down from top)
	totalShift := int(float64(overlayH) * (1.0 - progress))

	var rows []string

	// Title
	rows = append(rows, o.styles.OverlayTitle.Render("Sessions"))
	rows = append(rows, o.styles.HelpSep.Render(strings.Repeat("─", overlayW-6)))

	// Session list
	for i, s := range mgr.Sessions {
		label := truncate(s.Title, overlayW-10)
		date := s.UpdatedAt.Format("Jan 02 15:04")

		var dot string
		if s.ID == mgr.ActiveID {
			dot = session.ActiveDot
		} else {
			dot = session.InactiveDot
		}

		line := fmt.Sprintf("%s  %-*s  %s", dot, overlayW-20, label, date)

		if i == o.cursor {
			rows = append(rows, o.styles.SessionActive.Width(overlayW-6).Render(line))
		} else {
			rows = append(rows, o.styles.SessionItem.Width(overlayW-6).Render(line))
		}
	}

	// Footer hints
	rows = append(rows, "")
	rows = append(rows, o.styles.HelpSep.Render(strings.Repeat("─", overlayW-6)))
	rows = append(rows, o.styles.MessageMeta.Render(" ↑↓ navigate  enter select  ctrl+n new  esc close"))

	content := strings.Join(rows, "\n")
	box := o.styles.OverlayBg.
		Width(overlayW - 4).
		Height(overlayH - 2).
		Render(content)

	// Apply vertical shift (clip top rows to simulate slide)
	boxLines := strings.Split(box, "\n")
	if totalShift > 0 && totalShift < len(boxLines) {
		boxLines = boxLines[totalShift:]
	} else if totalShift >= len(boxLines) {
		return ""
	}

	// Blend transparency by adding horizontal padding to simulate placement
	rightPad := strings.Repeat(" ", o.width-overlayW)
	var placed []string
	for _, line := range boxLines {
		placed = append(placed, rightPad+line)
	}

	// Dim overlay: add a semi-transparent backdrop hint via a left column
	backdrop := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorBg))
	_ = backdrop // use when compositing full-screen overlays

	return strings.Join(placed, "\n")
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
