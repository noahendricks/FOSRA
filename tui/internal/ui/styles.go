package ui

import (
	"charm.land/lipgloss/v2"
)

// layout constants.
const (
	SidebarWidth       = 40
	SidebarMinWidth    = 34
	SidebarMaxWidth    = 46
	InputMinHeight     = 1
	InputMaxHeight     = 6
	HelpBarHeight      = 1
	MinWidthForSidebar = 96
	ChatPadding        = 1 // left/right gutter inside chat area

	// EditorVerticalRatio is the fraction of vertical space given to the
	// top (messages) panel. The remaining fraction goes to the bottom
	// (editor) panel. 0.90 = 90% messages, 10% editor (matches opencode).
	EditorVerticalRatio = 0.90
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
	AssistantMeta  lipgloss.Style

	// tool calls
	ToolCallIcon   lipgloss.Style // ✱ or $ prefix
	ToolCallBlock  lipgloss.Style // standalone tool block
	ToolCallHeader lipgloss.Style // tool name (bold, colored)
	ToolCallArgs   lipgloss.Style // arguments after tool name
	ToolCallOutput lipgloss.Style // output content inside tool block
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
	InputPrompt lipgloss.Style
	InputRAG    lipgloss.Style
	InputModel  lipgloss.Style // inverted pill: provider / model

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
	HelpBarFill  lipgloss.Style
	HelpBarState lipgloss.Style
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
		Background(lipgloss.Color(colorBg)).
		BorderStyle(lipgloss.ThickBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorPurple)).
		PaddingLeft(1).
		Foreground(lipgloss.Color(colorFg))

	s.AssistantBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		BorderStyle(lipgloss.ThickBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorBlue)).
		PaddingLeft(1).
		Foreground(lipgloss.Color(colorFg))

	s.AssistantMeta = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorComment))

	s.MessageAI = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))

	s.MessageMeta = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.MessageErr = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorRed)).
		Bold(true)

	s.Streaming = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	// ── tool calls ────────────────────────────────────────────
	s.ToolCallIcon = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorYellow))

	s.ToolCallBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderLeft(true).
		BorderRight(false).
		BorderTop(false).
		BorderBottom(false).
		BorderForeground(lipgloss.Color(colorComment)).
		PaddingLeft(1).
		Foreground(lipgloss.Color(colorFgDim))

	s.ToolCallHeader = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorGreen)).
		Bold(true)

	s.ToolCallArgs = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Italic(true)

	s.ToolCallOutput = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFgDim)).
		PaddingLeft(0)

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
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg)).
		Padding(0, 1)

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
	// NOTE: Border is now handled by the inputContainer in app.go.
	// Only the prompt style and ancillary badges live here.

	// prompt ">" in Primary (blue)
	s.InputPrompt = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true).
		Padding(0, 0, 0, 1)

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

	// ── sidebar ────────────────────────────────────────────────
	s.Sidebar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		PaddingLeft(4).
		PaddingRight(2)

	s.SidebarSection = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorBlue)).
		Bold(true)

	s.SidebarValue = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg))

	s.SidebarDim = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
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
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorComment)).
		Padding(0, 0)

	s.HelpBarInfo = lipgloss.NewStyle().
		Background(lipgloss.Color(colorFg)).
		Foreground(lipgloss.Color(colorBg)).
		Padding(0, 1)

	s.HelpKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Bold(true)

	s.HelpDesc = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.HelpSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDark))

	s.HelpBarFill = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFgDim))

	s.HelpBarState = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgDarker)).
		Foreground(lipgloss.Color(colorFgDim)).
		Padding(0, 1)

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
