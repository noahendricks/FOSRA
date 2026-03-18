package ui

import (
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type ChatPane struct {
	styles    Styles
	width     int
	height    int
	scrollPos int

	reveals []MessageReveal
	cursor  BlinkCursor
	spinner Spinner
}

func NewChatPane(styles Styles) ChatPane {
	return ChatPane{
		styles:  styles,
		cursor:  NewBlinkCursor(20),
		spinner: NewSpinner(styles.Spinner),
		width:   0,
		height:  0,
	}
}

func (c *ChatPane) SetSize(w, h int) {
	c.width = w
	c.height = h
}

func (c *ChatPane) ScrollUp() { c.scrollPos++ }

func (c *ChatPane) ScrollDown() {
	if c.scrollPos > 0 {
		c.scrollPos--
	}
}

func (c *ChatPane) ScrollToBottom() { c.scrollPos = 0 }

func (c *ChatPane) View(messages []session.Message) string {
	if len(messages) == 0 {
		placeholder := lipgloss.NewStyle().
			Foreground(lipgloss.Color(colorComment)).
			Italic(true).
			Width(c.width - 4).
			Align(lipgloss.Center).
			MarginTop(c.height / 3).
			Render("No messages yet. Start a conversation ↓")
		return c.styles.ChatPane.Width(c.width - 2).Height(c.height).Render(placeholder)
	}

	var lines []string

	for i, msg := range messages {
		rendered := c.renderMessage(msg, i)
		lines = append(lines, rendered)
		lines = append(lines, "")
	}

	content := strings.Join(lines, "\n")

	allLines := strings.Split(content, "\n")
	visible := c.height - 4
	total := len(allLines)

	start := total - visible - c.scrollPos
	if start < 0 {
		start = 0
	}
	end := start + visible
	if end > total {
		end = total
	}

	clipped := strings.Join(allLines[start:end], "\n")

	return c.styles.ChatPane.
		Width(c.width - 2).
		Height(c.height - 2).
		Render(clipped)
}

func (c *ChatPane) renderMessage(msg session.Message, idx int) string {
	contentW := c.width - 6

	switch msg.Role {
	case session.RoleUser:
		return c.renderUserBubble(msg.Content, contentW)

	case session.RoleAssistant:
		var parts []string

		if msg.Error != "" {
			errLine := c.styles.MessageErr.Render("Error: " + msg.Error)
			parts = append(parts, errLine)
			return strings.Join(parts, "\n")
		}

		label := c.styles.MessageMeta.Render("assistant")
		parts = append(parts, label)

		body := c.styles.MessageAI.
			Width(contentW).
			Render(msg.Content)
		parts = append(parts, body)

		if msg.IsStreaming {
			indicator := c.styles.Streaming.Render("generating...")
			parts = append(parts, indicator)
		}

		if len(msg.Sources) > 0 {
			parts = append(parts, c.renderSources(msg.Sources))
		}

		return strings.Join(parts, "\n")

	case session.RoleSystem:
		return c.styles.MessageMeta.
			Width(contentW).
			Render("system: " + msg.Content)

	default:
		return msg.Content
	}
}

func (c *ChatPane) renderUserBubble(content string, width int) string {
	label := c.styles.MessageUser.Render("you")

	body := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Width(width).
		Render(content)

	return label + "\n" + body
}

func (c *ChatPane) renderSources(sources []session.Source) string {
	var chips []string
	for _, src := range sources {
		label := fmt.Sprintf("📄 %s  %.0f%%", src.DocName, src.Score*100)
		chips = append(chips, c.styles.SourceChip.Render(label))
	}
	return "  " + strings.Join(chips, " ")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
