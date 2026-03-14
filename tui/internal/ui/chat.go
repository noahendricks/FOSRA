package ui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ChatPane renders the scrollable message history.
type ChatPane struct {
	styles    Styles
	width     int
	height    int
	scrollPos int // number of lines scrolled from bottom

	// Animation state per message (indexed by message slice position)
	reveals []MessageReveal
	cursor  BlinkCursor
	spinner Spinner
}

func NewChatPane(styles Styles) ChatPane {
	return ChatPane{
		styles:  styles,
		cursor:  NewBlinkCursor(20),
		spinner: NewSpinner(styles.Spinner),
	}
}

func (c *ChatPane) SetSize(w, h int) {
	c.width = w
	c.height = h
}

// OnNewMessage prepares a reveal animation for the newest message.
func (c *ChatPane) OnNewMessage(count int) {
	for len(c.reveals) < count {
		rev := NewMessageReveal()
		rev.Trigger()
		c.reveals = append(c.reveals, rev)
	}
}

// Step advances all animations one frame.
func (c *ChatPane) Step() {
	for i := range c.reveals {
		c.reveals[i].Step()
	}
	c.cursor.Tick()
	c.spinner.Tick()
}

func (c *ChatPane) ScrollUp() { c.scrollPos++ }
func (c *ChatPane) ScrollDown() {
	if c.scrollPos > 0 {
		c.scrollPos--
	}
}
func (c *ChatPane) ScrollToBottom() { c.scrollPos = 0 }

// View renders all messages.
func (c *ChatPane) View(messages []session.Message) string {
	if len(messages) == 0 {
		placeholder := lipgloss.NewStyle().
			Foreground(lipgloss.Color(colorComment)).
			Italic(true).
			Width(c.width - 4).
			Align(lipgloss.Center).
			MarginTop(c.height / 3).
			Render("No messages yet. Start a conversation ↓")
		return c.styles.ChatPane.Width(c.width - 2).Height(c.height - 2).Render(placeholder)
	}

	var lines []string
	for i, msg := range messages {
		rendered := c.renderMessage(msg, i)
		lines = append(lines, rendered)
		lines = append(lines, "") // spacer
	}

	content := strings.Join(lines, "\n")

	// Clip to height with scroll offset
	allLines := strings.Split(content, "\n")
	visible := c.height - 4 // account for border + padding
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
	innerWidth := c.width - 6

	var b strings.Builder

	switch msg.Role {
	case session.RoleUser:
		b.WriteString(c.styles.MessageUser.Render("  you"))
	case session.RoleAssistant:
		b.WriteString(c.styles.MessageAI.Render("  ai"))
		if c.spinner.running && idx == len(c.reveals)-1 {
			b.WriteString(" " + c.spinner.View())
		}
	case session.RoleSystem:
		b.WriteString(c.styles.MessageMeta.Render("  sys"))
	}

	// Timestamp
	b.WriteString("  ")
	b.WriteString(c.styles.MessageMeta.Render(msg.Timestamp.Format(time.Kitchen)))
	b.WriteString("\n")

	// Content
	if msg.Error != "" {
		b.WriteString(c.styles.MessageErr.Width(innerWidth).Render("⚠ " + msg.Error))
	} else {
		content := msg.Content
		if msg.IsStreaming {
			content += c.cursor.View()
		}
		switch msg.Role {
		case session.RoleUser:
			b.WriteString(c.renderUserBubble(content, innerWidth))
		default:
			b.WriteString(c.styles.MessageAI.Width(innerWidth).Render(content))
		}
	}

	// RAG sources
	if len(msg.Sources) > 0 {
		b.WriteString("\n")
		b.WriteString(c.renderSources(msg.Sources))
	}

	rendered := b.String()

	// Apply reveal animation (indent slides in from left)
	if idx < len(c.reveals) && c.reveals[idx].active {
		offset := c.reveals[idx].Offset(innerWidth)
		rendered = indentString(rendered, offset)
	}

	return rendered
}

func (c *ChatPane) renderUserBubble(content string, width int) string {
	bubbleWidth := min(width*3/4, len(content)+4)
	bubble := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1).
		Width(bubbleWidth).
		Align(lipgloss.Right).
		Render(content)

	// Right-align by padding left
	pad := width - lipgloss.Width(bubble)
	if pad < 0 {
		pad = 0
	}
	return strings.Repeat(" ", pad) + bubble
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
