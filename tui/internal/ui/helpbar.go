package ui

import (
	"strings"

	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
)

// HelpBar renders a persistent bottom bar with keyboard shortcut hints.
type HelpBar struct {
	styles Styles
	width  int
}

func NewHelpBar(styles Styles) HelpBar {
	return HelpBar{styles: styles}
}

func (h *HelpBar) SetWidth(w int) { h.width = w }

func (h *HelpBar) View() string {
	bindings := keys.DefaultKeyMap.ShortHelp()

	var parts []string
	for _, b := range bindings {
		help := b.Help()
		entry := h.styles.HelpKey.Render(help.Key) + " " + h.styles.HelpDesc.Render(help.Desc)
		parts = append(parts, entry)
	}

	sep := h.styles.HelpSep.Render(" · ")
	inner := strings.Join(parts, sep)

	return h.styles.HelpBar.
		Width(h.width).
		Render(inner)
}
