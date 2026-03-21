package ui

import (
	"strings"

	"charm.land/lipgloss/v2"
	ansi "github.com/charmbracelet/x/ansi"
)

func placeOverlay(x, y int, fg, bg string, shadow bool) string {
	if shadow {
		fg = withOverlayShadow(fg)
	}

	fgLines := strings.Split(fg, "\n")
	bgLines := strings.Split(bg, "\n")
	fgWidth := lipgloss.Width(fg)
	fgHeight := len(fgLines)
	bgWidth := lipgloss.Width(bg)
	bgHeight := len(bgLines)

	if fgWidth >= bgWidth && fgHeight >= bgHeight {
		return fg
	}

	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x > bgWidth-fgWidth {
		x = bgWidth - fgWidth
	}
	if y > bgHeight-fgHeight {
		y = bgHeight - fgHeight
	}

	var out strings.Builder
	for row, bgLine := range bgLines {
		if row > 0 {
			out.WriteByte('\n')
		}

		if row < y || row >= y+fgHeight {
			out.WriteString(bgLine)
			continue
		}

		fgLine := fgLines[row-y]
		left := ansi.Cut(bgLine, 0, x)
		right := ansi.Cut(bgLine, x+lipgloss.Width(fgLine), lipgloss.Width(bgLine))

		out.WriteString(left)
		if lipgloss.Width(left) < x {
			out.WriteString(strings.Repeat(" ", x-lipgloss.Width(left)))
		}
		out.WriteString(fgLine)
		out.WriteString(right)
	}

	return out.String()
}

func withOverlayShadow(content string) string {
	lines := strings.Split(content, "\n")
	width := lipgloss.Width(content)
	shadowChar := lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorBorderSubtle)).
		Render("░")

	shadowStyle := lipgloss.NewStyle().Background(lipgloss.Color(colorBg)).Render(" ")

	shadowLines := make([]string, 0, len(lines)+1)
	shadowLines = append(shadowLines, shadowStyle+strings.Repeat(shadowStyle, width))
	for range lines {
		shadowLines = append(shadowLines, shadowStyle+strings.Repeat(shadowChar, width))
	}

	shadow := strings.Join(shadowLines, "\n")
	return placeOverlay(0, 0, content, shadow, false)
}
