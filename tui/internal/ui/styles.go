package ui

import (
	"charm.land/lipgloss/v2"
)

// layout constants.
const (
	SidebarWidth       = 32
	InputMinHeight     = 3
	InputBorderHeight  = 1 // top border on input pane
	InputTotalHeight   = InputMinHeight + InputBorderHeight
	InputMaxHeight     = 6
	HelpBarHeight      = 1
	MinWidthForSidebar = 80
	ChatPadding        = 2 // left/right gutter inside chat area
)

// tokyo night color palette.
// Purple =  "Secondary" (#9d7cd8), Magenta = brighter variant (#bb9af7).
const (
	colorBg          = "#1a1b26"
	colorBgAlt       = "#16161e"
	colorBgDarker    = "#0c0e14" // status bar dark sections
	colorBgHighlight = "#292e42"
	colorBgFloat     = "#1f2335"
	colorBorder      = "#3b4261"
	colorComment     = "#565f89"
	colorFg          = "#c0caf5"
	colorFgDim       = "#9aa5ce"
	colorFgDark      = "#737aa2"
	colorBlue        = "#7aa2f7" // Primary
	colorCyan        = "#7dcfff"
	colorGreen       = "#9ece6a"
	colorYellow      = "#e0af68"
	colorOrange      = "#ff9e64"
	colorRed         = "#f7768e"
	colorPurple      = "#9d7cd8" // Secondary
	colorMagenta     = "#bb9af7" // brighter purple for accents
	colorTeal        = "#1abc9c"
)

type Styles struct {
	// app shell
	App lipgloss.Style

	// chat pane
	ChatPane    lipgloss.Style
	MessageUser lipgloss.Style // "you" label style
	MessageAI   lipgloss.Style // assistant body text
	MessageMeta lipgloss.Style // "assistant" label style
	MessageErr  lipgloss.Style
	Streaming   lipgloss.Style

	// user message block
	UserBlock      lipgloss.Style
	AssistantBlock lipgloss.Style

	// tool calls
	ToolCallIcon   lipgloss.Style // ✱ or $ prefix
	ToolCallHeader lipgloss.Style // tool name (bold, colored)
	ToolCallArgs   lipgloss.Style // arguments after tool name
	ToolCallOutput lipgloss.Style // output block with left border
	ToolCallCheck  lipgloss.Style // ✓ checkmark
	ToolCallStatus lipgloss.Style // status text (done, running)

	// thinking / reasoning
	ThinkingLabel lipgloss.Style
	ThinkingSep   lipgloss.Style

	// todo / task list
	TodoHeader  lipgloss.Style // "# Todos" header
	TodoDone    lipgloss.Style // [✓] done items
	TodoActive  lipgloss.Style // [●] in-progress items
	TodoPending lipgloss.Style // [ ] pending items

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
	InputModel   lipgloss.Style // inverted pill: provider / model

	// sidebar
	Sidebar              lipgloss.Style
	SidebarSection       lipgloss.Style // section headers (Primary, bold)
	SidebarValue         lipgloss.Style // section values (Text)
	SidebarDim           lipgloss.Style // muted text (TextMuted)
	SidebarDot           lipgloss.Style // active dot indicator
	SidebarProgressFull  lipgloss.Style // filled portion of progress bar
	SidebarProgressEmpty lipgloss.Style // empty portion of progress bar

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
	HelpBar      lipgloss.Style
	HelpBarInfo  lipgloss.Style
	HelpKey      lipgloss.Style
	HelpDesc     lipgloss.Style
	HelpSep      lipgloss.Style
	HelpBarRight lipgloss.Style // inverted pill for right-side badges

	// misc
	Spinner lipgloss.Style
}

func NewStyles() Styles {
	s := Styles{}

	// ── app shell ──────────────────────────────────────────────
	s.App = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))

	// ── chat ───────────────────────────────────────────────────
	s.ChatPane = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg))

	s.MessageUser = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPurple)).
		Bold(true)

	// user message block: thick purple (Secondary) left border
	s.UserBlock = lipgloss.NewStyle().
		BorderStyle(lipgloss.ThickBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorPurple)).
		PaddingLeft(1).
		Foreground(lipgloss.Color(colorFg))

	s.AssistantBlock = lipgloss.NewStyle().
		BorderStyle(lipgloss.ThickBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorBlue)).
		PaddingLeft(1).
		Foreground(lipgloss.Color(colorFg))

	s.MessageAI = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))

	s.MessageMeta = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.MessageErr = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorRed)).
		Bold(true)

	s.Streaming = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	// ── tool calls ────────────────────────────────────────────
	s.ToolCallIcon = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	s.ToolCallHeader = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true)

	s.ToolCallArgs = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Italic(true)

	// tool output: left border in TextMuted
	s.ToolCallOutput = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorComment)).
		PaddingLeft(1).
		MarginLeft(3)

	s.ToolCallCheck = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen))

	s.ToolCallStatus = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	// ── thinking ──────────────────────────────────────────────
	s.ThinkingLabel = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.ThinkingSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder))

	// ── todo / task list ──────────────────────────────────────
	s.TodoHeader = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Bold(true)

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
		Foreground(lipgloss.Color(colorFgDim)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1).
		Margin(0, 1, 0, 0)

	s.SourceScore = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	// ── input ──────────────────────────────────────────────────
	s.InputPane = lipgloss.NewStyle().
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		BorderBottom(false).
		BorderLeft(false).
		BorderRight(false).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(0, 1)

	// focused border = Primary (blue)
	s.InputFocused = lipgloss.NewStyle().
		BorderStyle(lipgloss.NormalBorder()).
		BorderTop(true).
		BorderBottom(false).
		BorderLeft(false).
		BorderRight(false).
		BorderForeground(lipgloss.Color(colorBlue)).
		Padding(0, 1)

	// prompt ">" in Primary (blue)
	s.InputPrompt = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true)

	s.InputRAG = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgAlt)).
		Background(lipgloss.Color(colorGreen)).
		Padding(0, 1).
		Bold(true).
		MarginRight(1)

	// inverted pill: Secondary bg, main bg fg
	s.InputModel = lipgloss.NewStyle().
		Background(lipgloss.Color(colorPurple)).
		Foreground(lipgloss.Color(colorBg)).
		Padding(0, 1).
		Bold(true)

	// ── sidebar (vertical border separator, no dark background) ──
	s.Sidebar = lipgloss.NewStyle().
		BorderStyle(lipgloss.NormalBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorBorder)).
		PaddingLeft(1).
		PaddingRight(1)

	s.SidebarSection = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true)

	s.SidebarValue = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))

	s.SidebarDim = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.SidebarDot = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen))

	s.SidebarProgressFull = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue))

	s.SidebarProgressEmpty = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgHighlight))

	// ── command palette overlay ───────────────────────────────
	s.Overlay = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgFloat)).
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(colorComment)).
		Padding(1, 2)

	s.OverlayTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true).
		MarginBottom(1)

	s.OverlayFilter = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1)

	s.SessionItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim))

	s.SessionActive = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBg)).
		Background(lipgloss.Color(colorBlue)).
		Bold(true)

	s.CommandItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim))

	s.CommandKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── RAG status ────────────────────────────────────────────
	s.StatusRAGOn = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true)

	s.StatusRAGOff = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── help bar ──────────────────────────────────────────────
	s.HelpBar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgDarker)).
		Foreground(lipgloss.Color(colorComment)).
		Padding(0, 1)

	s.HelpBarInfo = lipgloss.NewStyle().
		Background(lipgloss.Color(colorFg)).
		Foreground(lipgloss.Color(colorBgAlt)).
		Padding(0, 1)

	s.HelpKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Bold(true)

	s.HelpDesc = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.HelpSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDark))

	// inverted pill for right-side badges (TextMuted bg, BgDarker fg)
	s.HelpBarRight = lipgloss.NewStyle().
		Background(lipgloss.Color(colorComment)).
		Foreground(lipgloss.Color(colorBgDarker)).
		Padding(0, 1).
		Bold(true)

	// ── misc ───────────────────────────────────────────────────
	s.Spinner = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorCyan))

	return s
}
