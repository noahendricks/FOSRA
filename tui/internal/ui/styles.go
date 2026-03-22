package ui

import (
	"charm.land/lipgloss/v2"
)

// heavyBorder is a custom border with heavy vertical line (┃) for left border.
var heavyBorder = lipgloss.Border{
	Top:         "",
	Bottom:      "",
	Left:        "┃",
	Right:       "",
	TopLeft:     "",
	TopRight:    "",
	BottomLeft:  "",
	BottomRight: "",
}

// bottomCap is a custom border char for input bottom decoration.
var bottomCapBorder = lipgloss.Border{
	Top:         "",
	Bottom:      "▀",
	Left:        "",
	Right:       "",
	TopLeft:     "",
	TopRight:    "",
	BottomLeft:  "",
	BottomRight: "",
}

// layout constants.
const (
	SidebarWidth       = 42
	SidebarMinWidth    = 34
	SidebarMaxWidth    = 48
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

// opencode-inspired color palette (default dark theme).
// Primary = warm orange (#fab283), Secondary = blue (#5c9cf5).
const (
	colorBg           = "#0a0a0a" // background
	colorBgAlt        = "#141414" // backgroundPanel
	colorBgDarker     = "#0a0a0a" // same as main bg
	colorBgHighlight  = "#1e1e1e" // backgroundElement
	colorBgFloat      = "#141414" // overlay bg = panel
	colorBorder       = "#484848" // default border
	colorBorderActive = "#606060" // active/focused border
	colorBorderSubtle = "#3c3c3c" // subtle border
	colorComment      = "#808080" // textMuted
	colorFg           = "#eeeeee" // text
	colorFgDim        = "#808080" // same as muted
	colorFgDark       = "#606060"
	colorPrimary      = "#fab283" // Primary accent (agent color)
	colorSecondary    = "#5c9cf5" // Secondary accent (blue)
	colorInfo         = "#56b6c2" // info color
	colorSuccess      = "#7fd88f" // success/green
	colorWarning      = "#f5a742" // warning/yellow
	colorError        = "#e06c75" // error/red
	colorAccent       = "#9d7cd8" // Accent (purple)
	colorWarmYellow   = "#e5c07b" // warm yellow for blockquote, emph, KeywordType
	// legacy aliases for compatibility
	colorBlue    = "#5c9cf5" // alias for Secondary
	colorCyan    = "#56b6c2" // alias for Info
	colorGreen   = "#7fd88f" // alias for Success
	colorYellow  = "#f5a742" // alias for Warning
	colorOrange  = "#fab283" // alias for Primary
	colorRed     = "#e06c75" // alias for Error
	colorPurple  = "#9d7cd8" // alias for Accent
	colorMagenta = "#9d7cd8"
	colorTeal    = "#56b6c2" // merged with Info
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
	InlineTool     lipgloss.Style // inline tool (no border, muted)

	// thinking / reasoning
	ThinkingLabel lipgloss.Style
	ThinkingSep   lipgloss.Style
	ThinkingBlock lipgloss.Style // left border for thinking content

	// todo / task list
	TodoHeader  lipgloss.Style // "# Todos" header
	TodoDone    lipgloss.Style // [✓] done items
	TodoActive  lipgloss.Style // [•] in-progress items
	TodoPending lipgloss.Style // [ ] pending items
	TodoBlock   lipgloss.Style // container for todo list with bg

	// code blocks
	CodeBlock  lipgloss.Style
	CodeInline lipgloss.Style

	// source / RAG chips
	SourceChip  lipgloss.Style
	SourceScore lipgloss.Style

	// input area
	InputPrompt lipgloss.Style // removed - no longer used
	InputRAG    lipgloss.Style
	InputModel  lipgloss.Style // inverted pill: provider / model
	// input footer
	InputFooter         lipgloss.Style
	InputFooterAgent    lipgloss.Style // agent name in primary
	InputFooterModel    lipgloss.Style // model name in fg
	InputFooterProvider lipgloss.Style // provider in muted

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

	// completion footer
	CompletionFooter lipgloss.Style // container: MarginTop(1), PaddingLeft(3)
	CompletionSymbol lipgloss.Style // ▣ in agent color (colorPrimary), or colorComment if interrupted
	CompletionMode   lipgloss.Style // mode name in colorFg, bold
	CompletionMeta   lipgloss.Style // model + duration in colorComment

	// interrupt hint
	InterruptHint lipgloss.Style // "esc interrupt" text
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
		Foreground(lipgloss.Color(colorPrimary)).
		Bold(true)

	// user message block: heavy left border in Primary color (agent color)
	s.UserBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(heavyBorder).
		BorderLeft(true).
		BorderForeground(lipgloss.Color(colorPrimary)).
		PaddingTop(1).
		PaddingBottom(1).
		PaddingLeft(2).
		Foreground(lipgloss.Color(colorFg))

	// assistant message block: no border, just left padding (matches opencode)
	s.AssistantBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		PaddingLeft(3).
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
		Foreground(lipgloss.Color(colorError)).
		Bold(true)

	s.Streaming = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	// ── tool calls ────────────────────────────────────────────
	s.ToolCallIcon = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorWarning))

	s.ToolCallBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(heavyBorder).
		BorderLeft(true).
		BorderForeground(lipgloss.Color(colorBg)).
		PaddingLeft(2).
		Foreground(lipgloss.Color(colorFgDim))

	s.ToolCallHeader = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSuccess)).
		Bold(true)

	s.ToolCallArgs = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Italic(true)

	s.ToolCallOutput = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFgDim)).
		PaddingLeft(0)

	s.ToolCallCheck = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSuccess))

	s.ToolCallStatus = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.InlineTool = lipgloss.NewStyle().
		PaddingLeft(3).
		Foreground(lipgloss.Color(colorComment))

	// ── thinking ──────────────────────────────────────────────
	s.ThinkingLabel = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Italic(true)

	s.ThinkingSep = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBorder))

	// thinking block with left border (dimmed, similar to opencode's opacity effect)
	s.ThinkingBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		BorderStyle(heavyBorder).
		BorderLeft(true).
		BorderForeground(lipgloss.Color(colorBgHighlight)).
		PaddingLeft(2).
		Foreground(lipgloss.Color(colorComment))

	// ── todo / task list ──────────────────────────────────────
	s.TodoHeader = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
		Bold(true)

	s.TodoDone = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSuccess))

	s.TodoActive = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorWarning)).
		Bold(true)

	s.TodoPending = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	s.TodoBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(heavyBorder).
		BorderLeft(true).
		BorderForeground(lipgloss.Color(colorBgHighlight)).
		PaddingLeft(2).
		Foreground(lipgloss.Color(colorFgDim))

	// ── code blocks ───────────────────────────────────────────
	s.CodeBlock = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBg)).
		Foreground(lipgloss.Color(colorFg)).
		Padding(0, 1)

	s.CodeInline = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgHighlight)).
		Foreground(lipgloss.Color(colorInfo)).
		Padding(0, 1)

	// ── sources ────────────────────────────────────────────────
	s.SourceChip = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim)).
		Background(lipgloss.Color(colorBgHighlight)).
		Padding(0, 1).
		Margin(0, 1, 0, 0)

	s.SourceScore = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorWarning))

	// ── input ──────────────────────────────────────────────────
	// NOTE: Border is now handled by the inputContainer in app.go.
	// InputPrompt removed - no longer used (opencode has no ">" prompt).

	s.InputPrompt = lipgloss.NewStyle() // unused placeholder

	s.InputRAG = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBg)).
		Background(lipgloss.Color(colorSuccess)).
		Padding(0, 1).
		Bold(true).
		MarginRight(1)

	// inverted pill: Primary bg, main bg fg
	s.InputModel = lipgloss.NewStyle().
		Background(lipgloss.Color(colorPrimary)).
		Foreground(lipgloss.Color(colorBg)).
		Padding(0, 1).
		Bold(true)

	// input footer row
	s.InputFooter = lipgloss.NewStyle().
		PaddingTop(1).
		PaddingLeft(2)

	s.InputFooterAgent = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorPrimary)).
		Bold(true)

	s.InputFooterModel = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg))

	s.InputFooterProvider = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── sidebar ────────────────────────────────────────────────
	s.Sidebar = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		PaddingLeft(4).
		PaddingRight(2)

	s.SidebarSection = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFg)).
		Bold(true)

	s.SidebarValue = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorFg))

	s.SidebarDim = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		Foreground(lipgloss.Color(colorComment))

	s.SidebarDot = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSuccess))

	s.SidebarProgressFull = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSecondary))

	s.SidebarProgressEmpty = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorBgHighlight))

	// ── command palette overlay ───────────────────────────────
	s.Overlay = lipgloss.NewStyle().
		Background(lipgloss.Color(colorBgAlt)).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color(colorBorder)).
		Padding(1, 2)

	s.OverlayTitle = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFg)).
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
		Background(lipgloss.Color(colorPrimary)).
		Bold(true)

	s.CommandItem = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorFgDim))

	s.CommandKey = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment))

	// ── RAG status ────────────────────────────────────────────
	s.StatusRAGOn = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorSuccess)).
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
		Foreground(lipgloss.Color(colorFg))

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

	// inverted pill for right-side badges
	s.HelpBarRight = lipgloss.NewStyle().
		Background(lipgloss.Color(colorComment)).
		Foreground(lipgloss.Color(colorBg)).
		Padding(0, 1)

	// ── misc ───────────────────────────────────────────────────
	s.Spinner = lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorInfo))

	return s
}
