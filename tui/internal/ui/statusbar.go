package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// StatusBar renders the bottom status line.
type StatusBar struct {
	styles Styles
	width  int
}

func NewStatusBar(styles Styles) StatusBar {
	return StatusBar{styles: styles}
}

func (sb *StatusBar) SetWidth(w int) { sb.width = w }

func (sb StatusBar) View(sess *session.Session, ragEnabled bool, isStreaming bool) string {
	if sess == nil {
		return sb.styles.StatusBar.Width(sb.width).Render(" No active session")
	}

	// Left section: session title + model
	title := truncate(sess.Title, 30)
	model := sess.ModelName
	left := sb.styles.StatusItem.Render(fmt.Sprintf(" %s  %s", title, model))

	// Center: streaming indicator
	center := ""
	if isStreaming {
		center = sb.styles.Streaming.Render(" generating…")
	}

	// Right section: RAG toggle + message count
	var ragBadge string
	if ragEnabled {
		ragBadge = sb.styles.StatusRAGOn.Render("RAG ✓")
	} else {
		ragBadge = sb.styles.StatusRAGOff.Render("RAG ✗")
	}
	msgCount := fmt.Sprintf("%d msgs", len(sess.Messages))
	right := sb.styles.StatusItem.Render(ragBadge + "  " + msgCount + " ")

	// Layout: left + spacer + center + spacer + right
	leftW := lipgloss.Width(left)
	rightW := lipgloss.Width(right)
	centerW := lipgloss.Width(center)
	spacerTotal := sb.width - leftW - rightW - centerW
	if spacerTotal < 0 {
		spacerTotal = 0
	}
	spacerLeft := spacerTotal / 2
	spacerRight := spacerTotal - spacerLeft

	bar := left +
		strings.Repeat(" ", spacerLeft) +
		center +
		strings.Repeat(" ", spacerRight) +
		right

	return lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Width(sb.width).
		Render(bar)
}
