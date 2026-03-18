package ui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type Sidebar struct {
	styles  Styles
	width   int
	height  int
	sources []session.Source
}

func NewSidebar(styles Styles) Sidebar {
	return Sidebar{
		styles: styles,
	}
}

func (s *Sidebar) SetSize(w, h int) {
	s.width = w
	s.height = h
}

func (s *Sidebar) SetSources(sources []session.Source) {
	s.sources = sources
}

func (s *Sidebar) View() string {
	innerW := s.width - 2

	title := s.styles.SidebarTitle.
		Width(innerW).
		Render("Context")

	separator := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder)).
		Width(innerW).
		Render(strings.Repeat("─", innerW))

	var rows []string
	rows = append(rows, title)
	rows = append(rows, separator)

	if len(s.sources) == 0 {
		empty := lipgloss.NewStyle().
			Foreground(lipgloss.Color(colorComment)).
			Italic(true).
			Width(innerW).
			Padding(1, 1).
			Render("No sources loaded.\nToggle RAG with Ctrl+R")
		rows = append(rows, empty)
	} else {
		for i, src := range s.sources {
			score := fmt.Sprintf("%.0f%%", src.Score*100)
			scoreBadge := s.styles.SourceScore.Render(score)

			name := truncateSidebar(src.DocName, innerW-8)
			line := fmt.Sprintf(" %s  %s", scoreBadge, name)

			item := s.styles.SidebarItem.
				Width(innerW).
				Render(line)

			rows = append(rows, item)

			// Only show up to what fits
			if i >= s.height-6 {
				more := s.styles.SidebarItem.
					Foreground(lipgloss.Color(colorComment)).
					Render(fmt.Sprintf(" +%d more", len(s.sources)-i-1))
				rows = append(rows, more)
				break
			}
		}
	}

	content := strings.Join(rows, "\n")

	return s.styles.Sidebar.
		Width(innerW).
		Height(s.height - 2).
		Render(content)
}

func truncateSidebar(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
