package ui

import "github.com/charmbracelet/lipgloss"

// Tokyo-Night-inspired palette
const (
	colorBg          = "#1a1b26"
	colorBgAlt       = "#16161e"
	colorBgHighlight = "#292e42"
	colorBorder      = "#3b4261"
	colorComment     = "#565f89"
	colorFg          = "#c0caf5"
	colorFgDim       = "#9aa5ce"
	colorBlue        = "#7aa2f7"
	colorCyan        = "#7dcfff"
	colorGreen       = "#9ece6a"
	colorYellow      = "#e0af68"
	colorOrange      = "#ff9e64"
	colorRed         = "#f7768e"
	colorPurple      = "#bb9af7"
	colorMagenta     = "#bb9af7"
	colorTeal        = "#1abc9c"
)

// Styles groups all lipgloss styles used throughout the app.
type Styles struct {
	// App shell
	App          lipgloss.Style
	StatusBar    lipgloss.Style
	StatusItem   lipgloss.Style
	StatusRAGOn  lipgloss.Style
	StatusRAGOff lipgloss.Style

	// Chat pane
	ChatPane    lipgloss.Style
	MessageUser lipgloss.Style
	MessageAI   lipgloss.Style
	MessageMeta lipgloss.Style
	MessageErr  lipgloss.Style
	Streaming   lipgloss.Style

	// Source / RAG chips
	SourceChip  lipgloss.Style
	SourceScore lipgloss.Style

	// Input pane
	InputPane    lipgloss.Style
	InputBox     lipgloss.Style
	InputFocused lipgloss.Style

	// Session overlay
	OverlayBg     lipgloss.Style
	OverlayTitle  lipgloss.Style
	SessionItem   lipgloss.Style
	SessionActive lipgloss.Style

	// Help bar
	HelpKey  lipgloss.Style
	HelpDesc lipgloss.Style
	HelpSep  lipgloss.Style

	// Misc
	Spinner lipgloss.Style
	Badge   lipgloss.Style
}

func NewStyles() Styles {
	s := Styles{}

	s.App = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))

	s.StatusBar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.StatusItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.StatusRAGOn = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true)

	s.StatusRAGOff = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// Chat
	s.ChatPane = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1)

	s.MessageUser = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true)

	s.MessageAI = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))

	s.MessageMeta = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.MessageErr = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorRed)).
		Bold(true)

	s.Streaming = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan))

	// Sources
	s.SourceChip = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorPurple)).
		Padding(0, 1).
		Margin(0, 1, 0, 0).
		Bold(true)

	s.SourceScore = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	// Input
	s.InputPane = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1)

	s.InputFocused = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorBlue)).
		Padding(0, 1)

	// Session overlay
	s.OverlayBg = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorPurple)).
		Padding(1, 2)

	s.OverlayTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true).
		MarginBottom(1)

	s.SessionItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.SessionActive = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1).
		Bold(true)

	// Help
	s.HelpKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue))

	s.HelpDesc = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.HelpSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder))

	// Misc
	s.Spinner = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan))

	s.Badge = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorBlue)).
		Padding(0, 1).
		Bold(true)

	return s
}
