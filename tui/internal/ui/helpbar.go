package ui

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/key"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// HelpBar renders a compact status bar with key hints.
type HelpBar struct {
	styles Styles
	width  int
}

func NewHelpBar(styles Styles) HelpBar {
	return HelpBar{styles: styles}
}

func (h *HelpBar) SetWidth(w int) { h.width = w }

func (h *HelpBar) View(sess *session.Session) string {
	if h.width <= 0 {
		return ""
	}

	innerW := h.width - h.styles.HelpBar.GetHorizontalFrameSize()
	if innerW < 0 {
		innerW = 0
	}

	help := h.styles.HelpBarRight.Render("ctrl+p help")
	model := h.styles.InputModel.Render(h.modelLabel(sess, 24))
	info := h.renderStatusInfo(sess)

	leftParts := []string{help}
	if info != "" {
		leftParts = append(leftParts, " ", info)
	}
	left := lipgloss.JoinHorizontal(lipgloss.Center, leftParts...)
	leftW := lipgloss.Width(left)
	rightW := lipgloss.Width(model)
	available := innerW - leftW - rightW

	if available < 6 && info != "" {
		left = help
		leftW = lipgloss.Width(left)
		available = innerW - leftW - rightW
	}

	if available < 0 {
		model = h.styles.InputModel.Render(h.modelLabel(sess, 14))
		rightW = lipgloss.Width(model)
		available = innerW - leftW - rightW
	}

	if available < 0 {
		available = 0
	}

	middle := h.renderShortcuts(available)
	row := left + middle + model

	return h.styles.HelpBar.
		Width(h.width).
		Render(row)
}

func (h *HelpBar) renderStatusInfo(sess *session.Session) string {
	if sess == nil {
		return ""
	}

	pct := int(sess.ContextUsed * 100)
	info := fmt.Sprintf("Ctx %s (%d%%)", formatCompactTokens(sess.ContextTotal), pct)
	if sess.Cost > 0 {
		info += fmt.Sprintf("  $%.2f", sess.Cost)
	}
	return h.styles.HelpBarInfo.Render(info)
}

func (h *HelpBar) renderShortcuts(width int) string {
	if width <= 0 {
		return ""
	}

	bindings := []key.Binding{
		keys.DefaultKeyMap.FocusInput,
		keys.DefaultKeyMap.ToggleSidebar,
		keys.DefaultKeyMap.SessionsDirect,
		keys.DefaultKeyMap.ToggleRAG,
		keys.DefaultKeyMap.Quit,
	}
	sep := h.styles.HelpSep.Render(" · ")

	var parts []string
	used := 0
	for _, binding := range bindings {
		help := binding.Help()
		if help.Key == "" || help.Desc == "" {
			continue
		}

		entry := h.styles.HelpKey.Render(help.Key) + " " + h.styles.HelpDesc.Render(help.Desc)
		entryW := lipgloss.Width(entry)
		if len(parts) > 0 {
			entryW += lipgloss.Width(sep)
		}
		if used+entryW > width {
			break
		}
		parts = append(parts, entry)
		used += entryW
	}

	strip := strings.Join(parts, sep)
	return lipgloss.NewStyle().
		Width(width).
		Foreground(lipgloss.Color(colorComment)).
		Render(strip)
}

func (h *HelpBar) modelLabel(sess *session.Session, limit int) string {
	if sess == nil {
		return "No model"
	}

	label := sess.Provider + " / " + sess.ModelName
	return truncateInputLabel(label, limit)
}

func formatCompactTokens(tokens int) string {
	switch {
	case tokens >= 1_000_000:
		value := float64(tokens) / 1_000_000
		if value == float64(int(value)) {
			return fmt.Sprintf("%dM", int(value))
		}
		return fmt.Sprintf("%.1fM", value)
	case tokens >= 1_000:
		value := float64(tokens) / 1_000
		if value == float64(int(value)) {
			return fmt.Sprintf("%dK", int(value))
		}
		return fmt.Sprintf("%.1fK", value)
	default:
		return fmt.Sprintf("%d", tokens)
	}
}
