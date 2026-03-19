package ui

import (
	"image/color"

	"charm.land/lipgloss/v2"
)

// ── Container ─────────────────────────────────────────────────────────
//
// Container is a layout primitive that wraps rendered content with
// configurable padding and borders. It tracks outer dimensions and
// exposes ContentWidth / ContentHeight so the wrapped component can
// size itself to the available inner space.
//
// Modelled after opencode's layout.Container, adapted for struct-based
// (non-tea.Model) components.

type Container struct {
	width  int
	height int

	// padding
	paddingTop    int
	paddingRight  int
	paddingBottom int
	paddingLeft   int

	// border flags
	borderTop    bool
	borderRight  bool
	borderBottom bool
	borderLeft   bool
	borderStyle  lipgloss.Border
}

type ContainerOption func(*Container)

// NewContainer creates a Container with the given options.
func NewContainer(opts ...ContainerOption) Container {
	c := Container{
		borderStyle: lipgloss.NormalBorder(),
	}
	for _, opt := range opts {
		opt(&c)
	}
	return c
}

// ── Dimension methods ─────────────────────────────────────────────────

// SetSize stores the outer dimensions of the container.
func (c *Container) SetSize(width, height int) {
	c.width = width
	c.height = height
}

// GetSize returns the outer dimensions.
func (c *Container) GetSize() (int, int) {
	return c.width, c.height
}

// ContentWidth returns the width available for content after subtracting
// horizontal padding and borders.
func (c *Container) ContentWidth() int {
	w := c.width - c.paddingLeft - c.paddingRight
	if c.borderLeft {
		w--
	}
	if c.borderRight {
		w--
	}
	if w < 0 {
		return 0
	}
	return w
}

// ContentHeight returns the height available for content after subtracting
// vertical padding and borders.
func (c *Container) ContentHeight() int {
	h := c.height - c.paddingTop - c.paddingBottom
	if c.borderTop {
		h--
	}
	if c.borderBottom {
		h--
	}
	if h < 0 {
		return 0
	}
	return h
}

// ── Render ─────────────────────────────────────────────────────────────

// Render wraps content with the container's border, padding, and
// background. An optional borderFg overrides the border foreground
// colour (useful for focus-dependent styling).
func (c *Container) Render(content string, borderFg ...color.Color) string {
	style := lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg))

	width := c.width
	height := c.height

	// Apply border if any side is enabled.
	if c.borderTop || c.borderRight || c.borderBottom || c.borderLeft {
		if c.borderTop {
			height--
		}
		if c.borderBottom {
			height--
		}
		if c.borderLeft {
			width--
		}
		if c.borderRight {
			width--
		}

		style = style.
			BorderStyle(c.borderStyle).
			BorderTop(c.borderTop).
			BorderRight(c.borderRight).
			BorderBottom(c.borderBottom).
			BorderLeft(c.borderLeft)

		if len(borderFg) > 0 {
			style = style.BorderForeground(borderFg[0])
		} else {
			style = style.BorderForeground(lipgloss.Color(colorBorder))
		}
	}

	style = style.
		Width(width).
		Height(height).
		PaddingTop(c.paddingTop).
		PaddingRight(c.paddingRight).
		PaddingBottom(c.paddingBottom).
		PaddingLeft(c.paddingLeft)

	return style.Render(content)
}

// ── Container options ─────────────────────────────────────────────────

func WithPadding(top, right, bottom, left int) ContainerOption {
	return func(c *Container) {
		c.paddingTop = top
		c.paddingRight = right
		c.paddingBottom = bottom
		c.paddingLeft = left
	}
}

func WithBorder(top, right, bottom, left bool) ContainerOption {
	return func(c *Container) {
		c.borderTop = top
		c.borderRight = right
		c.borderBottom = bottom
		c.borderLeft = left
	}
}

func WithBorderStyle(style lipgloss.Border) ContainerOption {
	return func(c *Container) {
		c.borderStyle = style
	}
}

// ── SplitPaneLayout ───────────────────────────────────────────────────
//
// SplitPaneLayout computes dimension splits for a two-axis layout:
//   - vertical split: top portion (messages) vs bottom portion (editor)
//   - horizontal split: left portion vs right portion (sidebar)
//
// Ratios express the fraction allocated to the top / left panel.

type SplitPaneLayout struct {
	width  int
	height int

	verticalRatio float64 // e.g. 0.9 → 90% top, 10% bottom
	ratio         float64 // e.g. 0.7 → 70% left, 30% right

	hasBottom bool
	hasRight  bool
}

// NewSplitPaneLayout creates a layout with the given ratios.
func NewSplitPaneLayout(verticalRatio, ratio float64) SplitPaneLayout {
	return SplitPaneLayout{
		verticalRatio: verticalRatio,
		ratio:         ratio,
		hasBottom:     true,
	}
}

// SetSize stores the total available dimensions.
func (s *SplitPaneLayout) SetSize(width, height int) {
	s.width = width
	s.height = height
}

// GetSize returns the total dimensions.
func (s *SplitPaneLayout) GetSize() (int, int) {
	return s.width, s.height
}

// SetHasBottom enables or disables the bottom panel.
func (s *SplitPaneLayout) SetHasBottom(v bool) { s.hasBottom = v }

// SetHasRight enables or disables the right panel.
func (s *SplitPaneLayout) SetHasRight(v bool) { s.hasRight = v }

// ── Vertical split ────────────────────────────────────────────────────

// TopHeight returns the height allocated to the top (messages) panel.
func (s *SplitPaneLayout) TopHeight() int {
	if !s.hasBottom {
		return s.height
	}
	return int(float64(s.height) * s.verticalRatio)
}

// BottomHeight returns the height allocated to the bottom (editor) panel.
func (s *SplitPaneLayout) BottomHeight() int {
	if !s.hasBottom {
		return 0
	}
	return s.height - s.TopHeight()
}

// ── Horizontal split ──────────────────────────────────────────────────

// LeftWidth returns the width allocated to the left (main) panel.
func (s *SplitPaneLayout) LeftWidth() int {
	if !s.hasRight {
		return s.width
	}
	return int(float64(s.width) * s.ratio)
}

// RightWidth returns the width allocated to the right (sidebar) panel.
func (s *SplitPaneLayout) RightWidth() int {
	if !s.hasRight {
		return 0
	}
	return s.width - s.LeftWidth()
}
