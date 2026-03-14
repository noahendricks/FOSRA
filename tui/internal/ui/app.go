package ui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ─── Focus state ──────────────────────────────────────────────────────────────

type Focus int

const (
	FocusInput Focus = iota
	FocusMessages
	FocusOverlay
)

// ─── Tea messages ─────────────────────────────────────────────────────────────

// LLMResponseMsg carries a completed (or streamed chunk) LLM response.
type LLMResponseMsg struct {
	Content string
	Sources []session.Source
	Done    bool
	Err     error
}

// ─── App model ────────────────────────────────────────────────────────────────

type App struct {
	// Sub-components
	styles    Styles
	chat      ChatPane
	overlay   SessionOverlay
	statusBar StatusBar
	help      help.Model
	input     textarea.Model
	keys      keys.GlobalKeyMap

	// State
	sessions    *session.Manager
	focus       Focus
	width       int
	height      int
	isStreaming bool
	showHelp    bool

	// Animation
	needsAnim bool // true while any spring is unsettled
}

func NewApp() App {
	styles := NewStyles()
	km := keys.DefaultKeyMap
	mgr := session.NewManager()

	ta := textarea.New()
	ta.Placeholder = "Ask anything… (ctrl+a to attach docs)"
	ta.SetWidth(80)
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	ta.Focus()
	ta.CharLimit = 4096

	h := help.New()
	h.ShowAll = false

	return App{
		styles:    styles,
		chat:      NewChatPane(styles),
		overlay:   NewSessionOverlay(styles),
		statusBar: NewStatusBar(styles),
		help:      h,
		input:     ta,
		keys:      km,
		sessions:  mgr,
		focus:     FocusInput,
		needsAnim: false,
	}
}

// ─── Init ─────────────────────────────────────────────────────────────────────

func (a App) Init() tea.Cmd {
	return textarea.Blink
}

// ─── Update ───────────────────────────────────────────────────────────────────

func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	// ── Window resize ──────────────────────────────────────────────────────
	case tea.WindowSizeMsg:
		a.width = msg.Width
		a.height = msg.Height
		a.layout()

	// ── Animation tick ─────────────────────────────────────────────────────
	case AnimTickMsg:
		a.chat.Step()
		a.overlay.Step()

		// Keep ticking while animations are running
		if !a.overlay.AtRest() || a.isStreaming {
			cmds = append(cmds, AnimTick())
		} else {
			a.needsAnim = false
		}

	// ── LLM response ───────────────────────────────────────────────────────
	case LLMResponseMsg:
		sess := a.sessions.Active()
		if sess == nil {
			break
		}
		if msg.Err != nil {
			a.sessions.UpdateLastMessage(session.Message{
				ID:        lastID(sess),
				Role:      session.RoleAssistant,
				Content:   "",
				Error:     msg.Err.Error(),
				Timestamp: time.Now(),
			})
			a.chat.spinner.Stop()
			a.isStreaming = false
			break
		}

		a.sessions.UpdateLastMessage(session.Message{
			ID:          lastID(sess),
			Role:        session.RoleAssistant,
			Content:     msg.Content,
			Sources:     msg.Sources,
			IsStreaming: !msg.Done,
			Timestamp:   time.Now(),
		})

		if msg.Done {
			a.chat.spinner.Stop()
			a.isStreaming = false
		}
		a.chat.ScrollToBottom()
		a.chat.OnNewMessage(len(sess.Messages))

	// ── Keyboard ───────────────────────────────────────────────────────────
	case tea.KeyMsg:
		cmds = append(cmds, a.handleKey(msg)...)

	}

	// Forward to textarea when input is focused
	if a.focus == FocusInput {
		newTA, taCmd := a.input.Update(msg)
		a.input = newTA
		cmds = append(cmds, taCmd)
	}

	return a, tea.Batch(cmds...)
}

func (a *App) handleKey(msg tea.KeyMsg) []tea.Cmd {
	var cmds []tea.Cmd
	km := a.keys

	// ── Overlay is open: overlay navigation takes priority ─────────────────
	if a.overlay.IsOpen() {
		switch {
		case key.Matches(msg, km.FocusMessages): // esc
			a.overlay.Close()
			a.startAnim()

		case key.Matches(msg, km.SessionOverlay):
			a.overlay.Toggle()
			a.startAnim()

		case key.Matches(msg, km.ScrollUp):
			a.overlay.CursorUp(len(a.sessions.Sessions))

		case key.Matches(msg, km.ScrollDown):
			a.overlay.CursorDown(len(a.sessions.Sessions))

		case key.Matches(msg, km.NewSession):
			s := session.NewSession("New conversation")
			a.sessions.Add(s)
			a.overlay.Close()
			a.startAnim()

		case key.Matches(msg, km.Send): // enter = select session
			id := a.overlay.SelectedID(a.sessions.Sessions)
			if id != "" {
				a.sessions.Switch(id)
			}
			a.overlay.Close()
			a.startAnim()
		}
		return cmds
	}

	// ── Global keys ────────────────────────────────────────────────────────
	switch {
	case key.Matches(msg, km.Quit):
		cmds = append(cmds, tea.Quit)

	case key.Matches(msg, km.Help):
		a.showHelp = !a.showHelp
		a.help.ShowAll = a.showHelp

	case key.Matches(msg, km.SessionOverlay):
		a.overlay.Toggle()
		a.startAnim()

	case key.Matches(msg, km.NewSession):
		s := session.NewSession("New conversation")
		a.sessions.Add(s)

	case key.Matches(msg, km.ToggleRAG):
		sess := a.sessions.Active()
		if sess != nil {
			sess.RAGEnabled = !sess.RAGEnabled
		}

	case key.Matches(msg, km.FocusInput):
		if a.focus != FocusInput {
			a.focus = FocusInput
			a.input.Focus()
		}

	case key.Matches(msg, km.FocusMessages):
		if a.focus == FocusInput {
			a.focus = FocusMessages
			a.input.Blur()
		}

	case key.Matches(msg, km.ScrollUp):
		a.chat.ScrollUp()

	case key.Matches(msg, km.ScrollDown):
		a.chat.ScrollDown()

	case key.Matches(msg, km.Send):
		if a.focus == FocusInput {
			cmds = append(cmds, a.submitMessage()...)
		}
	}

	return cmds
}

// submitMessage sends the input content as a user message and fires an LLM call.
func (a *App) submitMessage() []tea.Cmd {
	content := strings.TrimSpace(a.input.Value())
	if content == "" || a.isStreaming {
		return nil
	}

	sess := a.sessions.Active()
	if sess == nil {
		return nil
	}

	// Append user message
	userMsg := session.Message{
		ID:        fmt.Sprintf("u%d", len(sess.Messages)),
		Role:      session.RoleUser,
		Content:   content,
		Timestamp: time.Now(),
	}
	a.sessions.AppendMessage(userMsg)
	a.chat.OnNewMessage(len(sess.Messages))

	// Placeholder assistant message (streaming)
	assistantMsg := session.Message{
		ID:          fmt.Sprintf("a%d", len(sess.Messages)),
		Role:        session.RoleAssistant,
		Content:     "",
		IsStreaming: true,
		Timestamp:   time.Now(),
	}
	a.sessions.AppendMessage(assistantMsg)
	a.chat.OnNewMessage(len(sess.Messages))

	// Start animation and spinner
	a.isStreaming = true
	a.chat.spinner.Start()
	a.input.Reset()
	a.chat.ScrollToBottom()
	a.startAnim()

	// TODO: Replace with your actual LLM call.
	// Return a tea.Cmd that eventually sends LLMResponseMsg back.
	return []tea.Cmd{
		AnimTick(),
		simulateLLMResponse(content, sess.RAGEnabled),
	}
}

// startAnim ensures the animation ticker is running.
func (a *App) startAnim() {
	if !a.needsAnim {
		a.needsAnim = true
	}
}

// layout recomputes sub-component sizes from the current window dimensions.
func (a *App) layout() {
	statusH := 1
	helpH := 1
	if a.showHelp {
		helpH = 5
	}
	inputH := 5
	chatH := a.height - statusH - helpH - inputH - 2

	a.chat.SetSize(a.width, chatH)
	a.overlay.SetSize(a.width, a.height-statusH-helpH)
	a.statusBar.SetWidth(a.width)
	a.input.SetWidth(a.width - 4)
}

// ─── View ─────────────────────────────────────────────────────────────────────

func (a App) View() string {
	sess := a.sessions.Active()

	statusH := 1
	helpH := 1
	if a.showHelp {
		helpH = 5
	}
	inputH := 5
	chatH := a.height - statusH - helpH - inputH - 2

	// Chat
	var messages []session.Message
	if sess != nil {
		messages = sess.Messages
	}
	chatView := a.chat.View(messages)

	// Input box
	inputStyle := a.styles.InputPane
	if a.focus == FocusInput {
		inputStyle = a.styles.InputFocused
	}
	inputView := inputStyle.Width(a.width - 2).Render(a.input.View())

	// Status bar
	var ragEnabled bool
	if sess != nil {
		ragEnabled = sess.RAGEnabled
	}
	statusView := a.statusBar.View(sess, ragEnabled, a.isStreaming)

	// Help
	helpView := lipgloss.NewStyle().
		Foreground(lipgloss.Color(colorComment)).
		Width(a.width).
		Render(a.help.View(a.keys))

	// Stack vertically
	base := lipgloss.JoinVertical(
		lipgloss.Left,
		chatView,
		inputView,
		helpView,
		statusView,
	)

	// Overlay compositing — draw overlay on top of base
	if a.overlay.anim.Progress() > 0.01 {
		overlayStr := a.overlay.View(a.sessions)
		if overlayStr != "" {
			base = compositeOverlay(base, overlayStr, chatH, a.width)
		}
	}

	_ = chatH // suppress unused warning if overlay not compositing
	return base
}

// compositeOverlay naively replaces the top-right lines of base with overlay.
// In a production app you'd use a proper layer compositor.
func compositeOverlay(base, overlay string, startLine, _ int) string {
	baseLines := strings.Split(base, "\n")
	overLines := strings.Split(overlay, "\n")

	for i, ol := range overLines {
		if i+1 < len(baseLines) { // +1 to start below the top border
			baseLines[i+1] = ol
		}
	}
	return strings.Join(baseLines, "\n")
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func lastID(s *session.Session) string {
	if len(s.Messages) == 0 {
		return ""
	}
	return s.Messages[len(s.Messages)-1].ID
}

// simulateLLMResponse is a stub — replace with real LLM streaming logic.
func simulateLLMResponse(prompt string, ragEnabled bool) tea.Cmd {
	return func() tea.Msg {
		time.Sleep(1200 * time.Millisecond)
		content := fmt.Sprintf("This is a stub response to: '%s'\n\nReplace simulateLLMResponse() in app.go with your actual LLM streaming call.", prompt)
		var sources []session.Source
		if ragEnabled {
			sources = []session.Source{
				{DocName: "example.pdf", Excerpt: "relevant chunk…", Score: 0.91, Page: 3},
				{DocName: "notes.md", Excerpt: "another chunk…", Score: 0.78, Page: 1},
			}
		}
		return LLMResponseMsg{Content: content, Sources: sources, Done: true}
	}
}
