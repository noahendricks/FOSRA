package ui

import (
	"fmt"
	"sort"
	"strings"

	"charm.land/lipgloss/v2"
	ansi "github.com/charmbracelet/x/ansi"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

const appVersion = "0.1.0"

type Sidebar struct {
	styles Styles
	width  int
	height int
}

func NewSidebar(styles Styles) Sidebar {
	return Sidebar{styles: styles}
}

func (s *Sidebar) SetSize(w, h int) {
	s.width = w
	s.height = h
}

// View renders a denser OpenCode-like sidebar adapted for RAG metadata.
func (s *Sidebar) View(sess *session.Session) string {
	if s.width < 4 || s.height < 3 {
		return ""
	}

	contentW := s.width - s.styles.Sidebar.GetHorizontalFrameSize()
	if contentW < 1 {
		contentW = 1
	}

	var sections []string
	sections = append(sections, s.renderHeader(contentW))

	if sess == nil {
		sections = append(sections, s.styles.SidebarDim.Width(contentW).Render("No active session"))
		return s.render(sections)
	}

	sections = append(sections,
		s.renderHeadline(sess, contentW),
		"",
		s.renderRAGSection(sess, contentW, s.height-countSections(sections)-4),
		"",
		s.styles.SidebarSection.Width(contentW).Render("Context"),
		s.renderContextInfo(sess, contentW),
		s.renderProgressBar(sess.ContextUsed, contentW),
	)

	return s.render(sections)
}

func (s *Sidebar) render(sections []string) string {
	content := lipgloss.JoinVertical(lipgloss.Left, sections...)
	return s.styles.Sidebar.
		Width(s.width).
		Height(s.height).
		Background(lipgloss.Color(colorBgAlt)).
		Render(content)
}

func (s *Sidebar) renderHeader(w int) string {
	icon := lipgloss.NewStyle().Foreground(lipgloss.Color(colorSecondary)).Bold(true).Render("⌬")
	name := lipgloss.NewStyle().Foreground(lipgloss.Color(colorFg)).Bold(true).Render(" FOSRA")
	ver := s.styles.SidebarDim.Render(" v" + appVersion)
	return lipgloss.NewStyle().Background(lipgloss.Color(colorBgAlt)).Width(w).Render(icon + name + ver)
}

func (s *Sidebar) renderHeadline(sess *session.Session, w int) string {
	headline := sess.Title
	if query := lastUserQuery(sess); query != "" {
		headline = query
	}

	return s.styles.SidebarValue.
		Bold(true).
		Width(w).
		Render(truncSidebar(headline, w))
}

func (s *Sidebar) renderContextInfo(sess *session.Session, w int) string {
	pct := int(sess.ContextUsed * 100)
	info := fmt.Sprintf("%s tokens  %d%% used", formatCompactTokens(sess.ContextTotal), pct)
	if sess.Cost > 0 {
		info += fmt.Sprintf("  $%.2f", sess.Cost)
	}
	return s.styles.SidebarDim.Width(w).Render(truncSidebar(info, w))
}

func (s *Sidebar) renderProgressBar(used float64, w int) string {
	barW := w
	if barW < 8 {
		barW = 8
	}
	filled := int(used * float64(barW))
	if filled > barW {
		filled = barW
	}
	if filled < 0 {
		filled = 0
	}
	empty := barW - filled

	bar := s.styles.SidebarProgressFull.Render(strings.Repeat("█", filled)) +
		s.styles.SidebarProgressEmpty.Render(strings.Repeat("░", empty))
	return lipgloss.NewStyle().Background(lipgloss.Color(colorBgAlt)).Width(w).Render(bar)
}

func (s *Sidebar) renderRAGSection(sess *session.Session, w int, availableH int) string {
	var rows []string

	dot := s.styles.SidebarDim.Render("○")
	if sess.RAG.Active {
		dot = s.styles.SidebarDot.Render("●")
	}

	header := s.styles.SidebarSection.Render("RAG") + " " + dot
	rows = append(rows, lipgloss.NewStyle().Background(lipgloss.Color(colorBgAlt)).Width(w).Render(header))

	indexLine := "disabled"
	if sess.RAG.Active {
		indexLine = sess.RAG.IndexName
		if indexLine == "" {
			indexLine = "no index selected"
		}
		if sess.RAG.Latency > 0 {
			indexLine += fmt.Sprintf("  %dms", sess.RAG.Latency.Milliseconds())
		}
	}
	rows = append(rows, s.styles.SidebarValue.Width(w).Render(truncSidebar(indexLine, w)))

	sources := sortedSources(s.lastSources(sess))
	if len(sources) == 0 {
		if sess.RAG.Active {
			rows = append(rows, s.styles.SidebarDim.Width(w).Render("no sources yet"))
		}
		return strings.Join(rows, "\n")
	}

	rows = append(rows, s.styles.SidebarSection.Width(w).Render("Sources"))

	maxItems := availableH - countSections(rows)
	if maxItems < 2 {
		maxItems = 2
	}
	if maxItems > len(sources) {
		maxItems = len(sources)
	}

	for i := 0; i < maxItems; i++ {
		src := sources[i]
		score := fmt.Sprintf("%.0f%%", src.Score*100)
		scorePart := s.styles.SidebarDim.Render(score)
		nameW := w - lipgloss.Width(scorePart) - 1
		if nameW < 8 {
			nameW = 8
		}
		namePart := s.styles.SidebarValue.Render(truncSidebar(src.DocName, nameW))
		rows = append(rows, lipgloss.NewStyle().Background(lipgloss.Color(colorBgAlt)).Width(w).Render(scorePart+" "+namePart))
	}

	if maxItems < len(sources) {
		rows = append(rows, s.styles.SidebarDim.Width(w).Render(fmt.Sprintf("+%d more", len(sources)-maxItems)))
	}

	return strings.Join(rows, "\n")
}

func (s *Sidebar) lastSources(sess *session.Session) []session.Source {
	for i := len(sess.Messages) - 1; i >= 0; i-- {
		if len(sess.Messages[i].Sources) > 0 {
			return sess.Messages[i].Sources
		}
	}
	return nil
}

func lastUserQuery(sess *session.Session) string {
	if sess == nil {
		return ""
	}

	for i := len(sess.Messages) - 1; i >= 0; i-- {
		msg := sess.Messages[i]
		if msg.Role != session.RoleUser || strings.TrimSpace(msg.Content) == "" {
			continue
		}
		return strings.Join(strings.Fields(msg.Content), " ")
	}

	return ""
}

// countSections counts how many lines are in the rows so far.
func countSections(rows []string) int {
	n := 0
	for _, r := range rows {
		n += strings.Count(r, "\n") + 1
	}
	return n
}

func truncSidebar(value string, width int) string {
	if width <= 0 {
		return ""
	}
	return ansi.Truncate(strings.Join(strings.Fields(value), " "), width, "…")
}

func sortedSources(sources []session.Source) []session.Source {
	if len(sources) == 0 {
		return nil
	}

	items := append([]session.Source(nil), sources...)
	sort.SliceStable(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
	return items
}
