package ui

import (
	"math"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/lipgloss"
)

// ─── Tick message ─────────────────────────────────────────────────────────────

const animFPS = 60

// AnimTickMsg is sent on every animation frame.
type AnimTickMsg time.Time

func AnimTick() tea.Cmd {
	return tea.Tick(time.Second/animFPS, func(t time.Time) tea.Msg {
		return AnimTickMsg(t)
	})
}

// ─── Spring animator ──────────────────────────────────────────────────────────

// SpringAnim tracks a single spring-animated float value.
// Use it for smooth slide-in/out, opacity fades, scale effects, etc.
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

// SetTarget moves the spring toward a new value.
func (a *SpringAnim) SetTarget(t float64) { a.target = t }

// Step advances the spring one frame.
func (a *SpringAnim) Step() {
	a.pos, a.velocity = a.spring.Update(a.pos, a.velocity, a.target)
}

// Value returns the current animated position.
func (a *SpringAnim) Value() float64 { return a.pos }

// AtRest returns true when the spring has settled.
func (a *SpringAnim) AtRest() bool {
	return math.Abs(a.pos-a.target) < 0.01 && math.Abs(a.velocity) < 0.01
}

// ─── Spinner ──────────────────────────────────────────────────────────────────

// Frames for the thinking spinner.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spinner tracks a braille animation for streaming / loading states.
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

// View renders the spinner, or empty string when stopped.
func (s Spinner) View() string {
	if !s.running {
		return ""
	}
	return s.style.Render(spinnerFrames[s.frame])
}

// ─── Typing cursor ────────────────────────────────────────────────────────────

// BlinkCursor renders a blinking block cursor appended to streaming text.
type BlinkCursor struct {
	visible bool
	ticks   int
	rate    int // blink every N ticks
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

// ─── Overlay slide spring ─────────────────────────────────────────────────────

// OverlayAnim drives the vertical slide-in of the session overlay.
type OverlayAnim struct {
	spring SpringAnim
	open   bool
}

func NewOverlayAnim() OverlayAnim {
	return OverlayAnim{
		// stiff spring with moderate damping → snappy slide
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

// Progress returns 0→1 slide progress.
func (o *OverlayAnim) Progress() float64 { return o.spring.Value() }
func (o *OverlayAnim) IsOpen() bool      { return o.open }
func (o *OverlayAnim) AtRest() bool      { return o.spring.AtRest() }

// ─── Message reveal animation ─────────────────────────────────────────────────

// MessageReveal animates a new message sliding in via spring.
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

// Offset returns how many characters to shift/clip for the reveal effect.
// Use this as a left-margin or opacity approximation in terminal rendering.
func (m *MessageReveal) Offset(width int) int {
	shifted := int(float64(width) * (1.0 - m.spring.Value()))
	return shifted
}

// ─── Utility: indent string by N spaces ──────────────────────────────────────

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
