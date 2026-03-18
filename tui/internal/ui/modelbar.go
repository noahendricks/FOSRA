package ui

import (
	"fmt"

	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ModelBar renders the centered model info bar at the top of the chat area.
// Shows: diamond icon, provider, model name, context %, and cost.
type ModelBar struct {
	styles Styles
	width  int
}

func NewModelBar(styles Styles) ModelBar {
	return ModelBar{styles: styles}
}

func (m *ModelBar) SetWidth(w int) { m.width = w }

func (m *ModelBar) View(sess *session.Session) string {
	if sess == nil {
		return m.renderEmpty()
	}

	diamond := m.styles.ModelDiamond.Render("◆")
	sep := m.styles.ModelSep.Render(" · ")

	provider := m.styles.ModelProvider.Render(sess.Provider)
	name := m.styles.ModelName.Render(sess.ModelName)

	contextStr := formatContext(sess.ContextUsed, sess.ContextTotal)
	context := m.styles.ModelContext.Render(contextStr)

	costStr := fmt.Sprintf("$%.2f", sess.Cost)
	cost := m.styles.ModelCost.Render(costStr)

	inner := diamond + " " + provider + " " + name + sep + context + sep + cost

	bar := m.styles.ModelBar.Render(inner)

	// center the bar within the available width
	return lipgloss.Place(m.width, ModelBarHeight, lipgloss.Center, lipgloss.Center, bar,
		lipgloss.WithWhitespaceStyle(lipgloss.NewStyle().Background(lipgloss.Color(colorBg))),
	)
}

func (m *ModelBar) renderEmpty() string {
	diamond := m.styles.ModelDiamond.Render("◆")
	inner := diamond + " " + m.styles.ModelName.Render("No session")
	bar := m.styles.ModelBar.Render(inner)
	return lipgloss.Place(m.width, ModelBarHeight, lipgloss.Center, lipgloss.Center, bar,
		lipgloss.WithWhitespaceStyle(lipgloss.NewStyle().Background(lipgloss.Color(colorBg))),
	)
}

func formatContext(used float64, total int) string {
	pct := int(used * 100)
	if total > 0 {
		return fmt.Sprintf("%d%% (%s)", pct, formatTokens(total))
	}
	return fmt.Sprintf("%d%%", pct)
}

func formatTokens(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	}
	if n >= 1000 {
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}
