package ui

import (
	"fmt"

	"charm.land/lipgloss/v2"
)

// top bar with badge and model info.
type Header struct {
	styles     Styles
	width      int
	modelName  string
	tokenCount int
}

func NewHeader(styles Styles) Header {
	return Header{
		styles:    styles,
		modelName: "gpt-4o",
	}
}

func (h *Header) SetWidth(w int) { h.width = w }

func (h *Header) SetModelInfo(name string, tokens int) {
	h.modelName = name
	h.tokenCount = tokens
}

func (h *Header) View() string {
	// badge on the left
	badge := h.styles.Badge.Render("FOSRA")

	// model info bar
	var info string
	if h.tokenCount > 0 {
		info = fmt.Sprintf("%s | %s tokens", h.modelName, formatTokens(h.tokenCount))
	} else {
		info = h.modelName
	}

	infoBarW := h.width*6/10 - lipgloss.Width(badge) - 3
	if infoBarW < 20 {
		infoBarW = 20
	}

	modelBar := h.styles.ModelInfo.
		Width(infoBarW).
		Render(info)

	// compose: badge + model info, left-aligned within header width
	inner := lipgloss.JoinHorizontal(lipgloss.Center, badge, modelBar)

	return h.styles.Header.
		Width(h.width).
		Render(inner)
}

func formatTokens(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}
