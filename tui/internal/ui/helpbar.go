package ui

import (
	"fmt"
	"strings"

	"charm.land/bubbles/v2/key"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// HelpBar renders a continuous segmented status bar.
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

	help := h.styles.HelpBarRight.Render("ctrl+p help")
	info := h.renderStatusInfo(sess)
	state := h.renderState(sess)
	model := h.styles.InputModel.Render(h.modelLabel(sess, 18))

	left := help + info
	right := state + model
	fillW := h.width - lipgloss.Width(left) - lipgloss.Width(right)
	if fillW < 0 {
		fillW = 0
	}

	middle := h.renderShortcuts(fillW)
	if middle == "" {
		middle = h.styles.HelpBarFill.Width(fillW).Render("")
	}

	return h.styles.HelpBar.Width(h.width).Render(left + middle + right)
}

func (h *HelpBar) renderStatusInfo(sess *session.Session) string {
	if sess == nil {
		return ""
	}

	pct := int(sess.ContextUsed * 100)
	info := fmt.Sprintf("Context: %s", formatCompactTokens(sess.ContextTotal))
	if pct > 0 {
		info += fmt.Sprintf(", %d%%", pct)
	}
	if sess.Cost > 0 {
		info += fmt.Sprintf(", $%.2f", sess.Cost)
	}
	return h.styles.HelpBarInfo.Render(info)
}

func (h *HelpBar) renderState(sess *session.Session) string {
	if sess == nil {
		return h.styles.HelpBarState.Render("No session")
	}

	label := "RAG off"
	if sess.RAG.Active {
		label = "RAG on"
		if sess.RAG.SourceCount > 0 {
			label = fmt.Sprintf("%d sources", sess.RAG.SourceCount)
		}
	}
	return h.styles.HelpBarState.Render(label)
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
	return h.styles.HelpBarFill.Width(width).Render(strip)
}

func (h *HelpBar) modelLabel(sess *session.Session, limit int) string {
	if sess == nil {
		return "No model"
	}
	return truncateInputLabel(sess.Provider+" / "+sess.ModelName, limit)
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
