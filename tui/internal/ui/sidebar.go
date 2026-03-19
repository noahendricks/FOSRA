package ui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
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

// View renders the sidebar with a compact app header, session headline, context, and RAG sources.
func (s *Sidebar) View(sess *session.Session) string {
	if s.width < 4 || s.height < 3 {
		return ""
	}

	contentW := s.width - 3
	if contentW < 1 {
		contentW = 1
	}

	var sections []string

	// ── Section 1: App header ──
	icon := lipgloss.NewStyle().Foreground(lipgloss.Color(colorBlue)).Bold(true).Render("⌬")
	name := lipgloss.NewStyle().Foreground(lipgloss.Color(colorFg)).Bold(true).Render(" FOSRA")
	ver := s.styles.SidebarDim.Render(" v" + appVersion)
	sections = append(sections, icon+name+ver)

	if sess == nil {
		sections = append(sections, s.styles.SidebarDim.Render("No active session"))
		return s.render(sections)
	}

	sections = append(sections, s.renderHeadline(sess, contentW))

	sections = append(sections, s.styles.SidebarSection.Render("Context"))
	sections = append(sections, s.renderContextInfo(sess, contentW))
	sections = append(sections, s.renderProgressBar(sess.ContextUsed, contentW))

	remainingH := s.height - countSections(sections)
	if remainingH < 3 {
		remainingH = 3
	}
	sections = append(sections, s.renderRAGSection(sess, contentW, remainingH))

	return s.render(sections)
}

func (s *Sidebar) render(sections []string) string {
	content := lipgloss.JoinVertical(lipgloss.Left, sections...)
	return s.styles.Sidebar.
		Width(s.width).
		Height(s.height).
		Render(content)
}

func (s *Sidebar) renderContextInfo(sess *session.Session, w int) string {
	pct := int(sess.ContextUsed * 100)
	tokens := sess.ContextTotal
	var tokenStr string
	if tokens >= 1000 {
		tokenStr = fmt.Sprintf("%dk", tokens/1000)
	} else {
		tokenStr = fmt.Sprintf("%d", tokens)
	}

	info := fmt.Sprintf("%s tokens (%d%%)", tokenStr, pct)
	if sess.Cost > 0 {
		info += fmt.Sprintf("  $%.4f", sess.Cost)
	}
	return s.styles.SidebarDim.Render(truncSidebar(info, w))
}

func (s *Sidebar) renderHeadline(sess *session.Session, w int) string {
	headline := sess.Title
	if query := lastUserQuery(sess); query != "" {
		headline = query
	}
	return s.styles.SidebarValue.
		Bold(true).
		Render(truncSidebar(headline, w))
}

func (s *Sidebar) renderProgressBar(used float64, w int) string {
	barW := w
	if barW < 4 {
		barW = 4
	}
	filled := int(used * float64(barW))
	if filled > barW {
		filled = barW
	}
	empty := barW - filled

	bar := s.styles.SidebarProgressFull.Render(strings.Repeat("█", filled)) +
		s.styles.SidebarProgressEmpty.Render(strings.Repeat("░", empty))
	return bar
}

func (s *Sidebar) renderRAGSection(sess *session.Session, w int, availableH int) string {
	var rows []string

	// Header with status dot
	var dot string
	if sess.RAG.Active {
		dot = s.styles.SidebarDot.Render("●")
	} else {
		dot = s.styles.SidebarDim.Render("○")
	}
	label := s.styles.SidebarSection.Render("RAG")
	rows = append(rows, label+" "+dot)

	// Index name + latency
	if sess.RAG.IndexName != "" {
		idx := s.styles.SidebarValue.Render(sess.RAG.IndexName)
		if sess.RAG.Latency > 0 {
			lat := s.styles.SidebarDim.Render(fmt.Sprintf("  %dms", sess.RAG.Latency.Milliseconds()))
			idx += lat
		}
		rows = append(rows, truncSidebar(idx, w))
	}

	// Sources from last assistant message
	sources := s.lastSources(sess)
	if len(sources) == 0 {
		if !sess.RAG.Active {
			rows = append(rows, s.styles.SidebarDim.Render("disabled"))
		} else {
			rows = append(rows, s.styles.SidebarDim.Render("no sources yet"))
		}
		return strings.Join(rows, "\n")
	}

	rows = append(rows, s.styles.SidebarSection.Render("Sources"))

	maxItems := availableH - countSections(rows)
	if maxItems < 1 {
		maxItems = 1
	}
	if maxItems > len(sources) {
		maxItems = len(sources)
	}

	for i := 0; i < maxItems; i++ {
		src := sources[i]
		score := fmt.Sprintf("%.0f%%", src.Score*100)
		scorePart := s.styles.SidebarDim.Render(score)
		nameW := w - lipgloss.Width(score) - 1
		if nameW < 8 {
			nameW = 8
		}
		namePart := s.styles.SidebarValue.Render(truncSidebar(src.DocName, nameW))
		rows = append(rows, scorePart+" "+namePart)
	}

	if maxItems < len(sources) {
		more := s.styles.SidebarDim.Render(fmt.Sprintf("+%d more", len(sources)-maxItems))
		rows = append(rows, more)
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

// countSections counts how many lines are in the rows so far (rough estimate).
func countSections(rows []string) int {
	n := 0
	for _, r := range rows {
		n += strings.Count(r, "\n") + 1
	}
	return n
}

func truncSidebar(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return "…"
	}
	return s[:n-1] + "…"
}
