package ui

import (
	"math"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/harmonica"
)

const animFPS = 60

type AnimTickMsg time.Time

func AnimTick() tea.Cmd {
	return tea.Tick(time.Second/animFPS, func(t time.Time) tea.Msg {
		return AnimTickMsg(t)
	})
}

type SpringAnim struct {
	spring   harmonica.Spring
	pos      float64
	velocity float64
	target   float64
}

func NewSpring(stiffness, damping float64) SpringAnim {
	return SpringAnim{
		spring: harmonica.NewSpring(harmonica.FPS(animFPS), stiffness, damping),
	}
}

func (a *SpringAnim) SetTarget(t float64) { a.target = t }

func (a *SpringAnim) Step() {
	a.pos, a.velocity = a.spring.Update(a.pos, a.velocity, a.target)
}

func (a *SpringAnim) Value() float64 { return a.pos }

func (a *SpringAnim) AtRest() bool {
	return math.Abs(a.pos-a.target) < 0.01 && math.Abs(a.velocity) < 0.01
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type Spinner struct {
	frame   int
	running bool
	style   lipgloss.Style
}

func NewSpinner(style lipgloss.Style) Spinner {
	return Spinner{style: style}
}

func (s *Spinner) Start()        { s.running = true }
func (s *Spinner) Stop()         { s.running = false; s.frame = 0 }
func (s *Spinner) Running() bool { return s.running }

func (s *Spinner) Tick() {
	if s.running {
		s.frame = (s.frame + 1) % len(spinnerFrames)
	}
}

func (s Spinner) View() string {
	if !s.running {
		return ""
	}
	return s.style.Render(spinnerFrames[s.frame])
}

// ── Blink cursor (used for streaming text cursor) ────────────────────

type BlinkCursor struct {
	visible bool
	ticks   int
	rate    int
}

func NewBlinkCursor(rateFrames int) BlinkCursor {
	return BlinkCursor{visible: true, rate: rateFrames}
}

func (c *BlinkCursor) Tick() {
	c.ticks++
	if c.ticks >= c.rate {
		c.visible = !c.visible
		c.ticks = 0
	}
}

func (c BlinkCursor) View() string {
	if c.visible {
		return "▋"
	}
	return " "
}

// ── Message reveal (spring-based slide-in for new messages) ──────────

type MessageReveal struct {
	spring SpringAnim
	active bool
}

func NewMessageReveal() MessageReveal {
	return MessageReveal{spring: NewSpring(120, 14)}
}

func (m *MessageReveal) Trigger() {
	m.spring.pos = 0
	m.spring.velocity = 0
	m.spring.SetTarget(1.0)
	m.active = true
}

func (m *MessageReveal) Step() {
	m.spring.Step()
	if m.spring.AtRest() {
		m.active = false
	}
}

func (m *MessageReveal) Active() bool { return m.active }

func (m *MessageReveal) Offset(width int) int {
	return int(float64(width) * (1.0 - m.spring.Value()))
}

// ── Sidebar toggle (instant, no animation) ───────────────────────────

type SidebarToggle struct {
	open  bool
	baseW int
}

func NewSidebarToggle(maxWidth int) SidebarToggle {
	return SidebarToggle{open: true, baseW: maxWidth}
}

func (s *SidebarToggle) Toggle()      { s.open = !s.open }
func (s *SidebarToggle) IsOpen() bool { return s.open }

func (s *SidebarToggle) Width(totalWidth int) int {
	if s.open {
		width := s.baseW
		if totalWidth >= MinWidthForSidebar {
			width = int(float64(totalWidth) * 0.28)
		}
		if width < SidebarMinWidth {
			width = SidebarMinWidth
		}
		if width > SidebarMaxWidth {
			width = SidebarMaxWidth
		}
		if width >= totalWidth-20 {
			width = max(0, totalWidth-20)
		}
		return width
	}
	return 0
}

// ── Overlay toggle (instant, no animation) ───────────────────────────

type OverlayToggle struct {
	open bool
}

func NewOverlayToggle() OverlayToggle {
	return OverlayToggle{}
}

func (o *OverlayToggle) Open()        { o.open = true }
func (o *OverlayToggle) Close()       { o.open = false }
func (o *OverlayToggle) IsOpen() bool { return o.open }

// ── Helpers ──────────────────────────────────────────────────────────

func indentString(s string, n int) string {
	if n <= 0 {
		return s
	}
	pad := strings.Repeat(" ", n)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = pad + l
	}
	return strings.Join(lines, "\n")
}
