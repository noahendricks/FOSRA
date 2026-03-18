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

func (s *Spinner) Start() { s.running = true }
func (s *Spinner) Stop()  { s.running = false; s.frame = 0 }

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

type OverlayAnim struct {
	spring SpringAnim
	open   bool
}

func NewOverlayAnim() OverlayAnim {
	return OverlayAnim{
		spring: NewSpring(180, 18),
	}
}

func (o *OverlayAnim) Open() {
	o.open = true
	o.spring.SetTarget(1.0)
}

func (o *OverlayAnim) Close() {
	o.open = false
	o.spring.SetTarget(0.0)
}

func (o *OverlayAnim) Toggle() {
	if o.open {
		o.Close()
	} else {
		o.Open()
	}
}

func (o *OverlayAnim) Step() { o.spring.Step() }

func (o *OverlayAnim) Progress() float64 { return o.spring.Value() }
func (o *OverlayAnim) IsOpen() bool      { return o.open }
func (o *OverlayAnim) AtRest() bool      { return o.spring.AtRest() }

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

func (m *MessageReveal) Offset(width int) int {
	shifted := int(float64(width) * (1.0 - m.spring.Value()))
	return shifted
}

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

// SidebarAnim drives the collapsible sidebar width via spring physics.
type SidebarAnim struct {
	spring SpringAnim
	open   bool
	maxW   int // target width when fully open
}

func NewSidebarAnim(maxWidth int) SidebarAnim {
	sa := SidebarAnim{
		spring: NewSpring(170, 16),
		open:   true,
		maxW:   maxWidth,
	}
	sa.spring.pos = float64(maxWidth)
	sa.spring.SetTarget(float64(maxWidth))
	return sa
}

func (s *SidebarAnim) Toggle() {
	s.open = !s.open
	if s.open {
		s.spring.SetTarget(float64(s.maxW))
	} else {
		s.spring.SetTarget(0)
	}
}

func (s *SidebarAnim) IsOpen() bool { return s.open }
func (s *SidebarAnim) Step()        { s.spring.Step() }
func (s *SidebarAnim) AtRest() bool { return s.spring.AtRest() }

func (s *SidebarAnim) Width() int {
	w := int(s.spring.Value() + 0.5)
	if w < 0 {
		return 0
	}
	if w > s.maxW {
		return s.maxW
	}
	return w
}
