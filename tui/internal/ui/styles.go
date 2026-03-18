package ui

import (
	"charm.land/lipgloss/v2"
)

// layout constants.
const (
	SidebarWidth       = 30
	ModelBarHeight     = 3
	InputMinHeight     = 3
	InputMaxHeight     = 6
	HelpBarHeight      = 1
	MinWidthForSidebar = 80
)

// tokyo night
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

type Styles struct {
	// app shell
	App lipgloss.Style

	// model bar (centered top bar)
	ModelBar      lipgloss.Style
	ModelProvider lipgloss.Style
	ModelName     lipgloss.Style
	ModelContext  lipgloss.Style
	ModelCost     lipgloss.Style
	ModelSep      lipgloss.Style
	ModelDiamond  lipgloss.Style

	// chat pane
	ChatPane    lipgloss.Style
	MessageUser lipgloss.Style
	MessageAI   lipgloss.Style
	MessageMeta lipgloss.Style
	MessageErr  lipgloss.Style
	Streaming   lipgloss.Style

	// tool calls
	ToolCallHeader lipgloss.Style
	ToolCallName   lipgloss.Style
	ToolCallOutput lipgloss.Style
	ToolCallCheck  lipgloss.Style

	// thinking / reasoning
	ThinkingLabel lipgloss.Style

	// todo / task list
	TodoBlock   lipgloss.Style
	TodoDone    lipgloss.Style
	TodoActive  lipgloss.Style
	TodoPending lipgloss.Style

	// code blocks
	CodeBlock  lipgloss.Style
	CodeInline lipgloss.Style

	// source / RAG chips
	SourceChip  lipgloss.Style
	SourceScore lipgloss.Style

	// input area
	InputPane    lipgloss.Style
	InputFocused lipgloss.Style
	InputPrompt  lipgloss.Style
	InputRAG     lipgloss.Style

	// sidebar
	Sidebar      lipgloss.Style
	SidebarTitle lipgloss.Style
	SidebarItem  lipgloss.Style
	SidebarSep   lipgloss.Style

	// command palette overlay
	Overlay       lipgloss.Style
	OverlayTitle  lipgloss.Style
	OverlayFilter lipgloss.Style
	SessionItem   lipgloss.Style
	SessionActive lipgloss.Style
	CommandItem   lipgloss.Style
	CommandKey    lipgloss.Style

	// RAG status
	StatusRAGOn  lipgloss.Style
	StatusRAGOff lipgloss.Style

	// help bar
	HelpBar  lipgloss.Style
	HelpKey  lipgloss.Style
	HelpDesc lipgloss.Style
	HelpSep  lipgloss.Style

	// misc
	Spinner lipgloss.Style
}

func NewStyles() Styles {
	s := Styles{}

	// ── app shell ──────────────────────────────────────────────
	s.App = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))

	// ── model bar (centered floating) ─────────────────────────
	s.ModelBar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgHighlight)).
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 2)

	s.ModelDiamond = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true)

	s.ModelProvider = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true)

	s.ModelName = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))

	s.ModelContext = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	s.ModelCost = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen))

	s.ModelSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── chat ───────────────────────────────────────────────────
	s.ChatPane = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg))

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

	// ── tool calls ────────────────────────────────────────────
	s.ToolCallHeader = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgHighlight)).
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true).
		Padding(0, 1)

	s.ToolCallName = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Italic(true)

	s.ToolCallOutput = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFgDim)).
		BorderStyle(lipgloss.ThickBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1).
		MarginLeft(2)

	s.ToolCallCheck = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen))

	// ── thinking ──────────────────────────────────────────────
	s.ThinkingLabel = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	// ── todo / task list ──────────────────────────────────────
	s.TodoBlock = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1).
		MarginLeft(2)

	s.TodoDone = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen))

	s.TodoActive = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan)).
		Bold(true)

	s.TodoPending = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── code blocks ───────────────────────────────────────────
	s.CodeBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFg)).
		Padding(0, 1).
		MarginLeft(2)

	s.CodeInline = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgHighlight)).
		Foreground(lipgloss.Color(colorCyan)).
		Padding(0, 1)

	// ── sources ────────────────────────────────────────────────
	s.SourceChip = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorPurple)).
		Padding(0, 1).
		Margin(0, 1, 0, 0).
		Bold(true)

	s.SourceScore = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	// ── input ──────────────────────────────────────────────────
	s.InputPane = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		BorderBottom(false).
		BorderLeft(false).
		BorderRight(false).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1)

	s.InputFocused = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		BorderBottom(false).
		BorderLeft(false).
		BorderRight(false).
		BorderForeground(lipgloss.Color(colorBlue)).
		Padding(0, 1)

	s.InputPrompt = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan)).
		Bold(true)

	s.InputRAG = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorPurple)).
		Padding(0, 1).
		Bold(true).
		MarginRight(1)

	// ── sidebar ────────────────────────────────────────────────
	s.Sidebar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorBorder))

	s.SidebarTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true).
		Padding(0, 1)

	s.SidebarItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.SidebarSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder))

	// ── command palette overlay ───────────────────────────────
	s.Overlay = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorPurple)).
		Padding(1, 2)

	s.OverlayTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true).
		MarginBottom(1)

	s.OverlayFilter = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1)

	s.SessionItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.SessionActive = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1).
		Bold(true)

	s.CommandItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

	s.CommandKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue))

	// ── RAG status ────────────────────────────────────────────
	s.StatusRAGOn = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true)

	s.StatusRAGOff = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── HELP BAR ──────────────────────────────────────────────
	s.HelpBar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorComment)).
		Padding(0, 1)

	s.HelpKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Bold(true)

	s.HelpDesc = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.HelpSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder))

	// ── MISC ───────────────────────────────────────────────────
	s.Spinner = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan))

	return s
}
