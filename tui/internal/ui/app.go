package ui

import (
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type App struct {
	styles Styles

	// components
	chat    ChatPane
	input   ChatInput
	sidebar Sidebar
	helpBar HelpBar
	palette CommandPalette

	// layout
	splitLayout    SplitPaneLayout
	chatContainer  Container
	inputContainer Container

	// state
	sessions     *session.Manager
	sidebarAnim  SidebarToggle
	overlayAnim  OverlayToggle
	windowWidth  int
	windowHeight int
}

func NewApp() App {
	styles := NewStyles()
	mgr := session.NewManager()

	return App{
		styles:  styles,
		chat:    NewChatPane(styles),
		input:   NewChatInput(styles),
		sidebar: NewSidebar(styles),
		helpBar: NewHelpBar(styles),
		palette: NewCommandPalette(styles),

		// vertical 90/10 split (messages / editor); no horizontal ratio used yet
		splitLayout: NewSplitPaneLayout(EditorVerticalRatio, 1.0),

		// chat container: padding 1,1,0,1 (top, right, bottom, left) — matches opencode
		chatContainer: NewContainer(
			WithPadding(1, 1, 0, 1),
		),

		// input container: top border only (drawn by Container.Render)
		inputContainer: NewContainer(
			WithBorder(true, false, false, false),
		),

		sessions:    mgr,
		sidebarAnim: NewSidebarToggle(SidebarWidth),
		overlayAnim: NewOverlayToggle(),
	}
}

func (a App) Init() tea.Cmd {
	return tea.Batch(
		a.input.Focus(),
		AnimTick(),
	)
}

func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.windowWidth = msg.Width
		a.windowHeight = msg.Height
		a.relayout()

	case AnimTickMsg:
		return a.handleAnimTick()

	case mockPipelineStep:
		return a.handleMockStep(msg.step)

	case mockStreamChunk:
		return a.handleMockStream(msg)

	case tea.KeyPressMsg:
		if cmd := a.handleKey(msg); cmd != nil {
			cmds = append(cmds, cmd)
		}
		// if overlay is open, don't forward keys to input/chat
		if a.overlayAnim.IsOpen() {
			return a, tea.Batch(cmds...)
		}
	}

	// forward to input
	if !a.overlayAnim.IsOpen() {
		cmds = append(cmds, a.input.Update(msg))
	}

	// forward to chat viewport (scroll/mouse)
	cmds = append(cmds, a.chat.Update(msg))

	return a, tea.Batch(cmds...)
}

// handleKey processes key events and returns a cmd if needed.
func (a *App) handleKey(msg tea.KeyPressMsg) tea.Cmd {
	km := keys.DefaultKeyMap

	// overlay-specific keys when open
	if a.overlayAnim.IsOpen() {
		switch {
		case key.Matches(msg, km.FocusMessages): // esc = close overlay
			a.closeOverlay()
			return nil
		case key.Matches(msg, km.ScrollUp):
			a.paletteUp()
			return nil
		case key.Matches(msg, km.ScrollDown):
			a.paletteDown()
			return nil
		case key.Matches(msg, km.Send): // enter = select
			return a.paletteSelect()
		}
		// Backspace goes back from sessions to commands
		if msg.String() == "backspace" && a.palette.CurrentView() == OverlaySessions {
			a.palette.ShowCommands()
			return nil
		}
		return nil
	}

	// GLOBAL KEYS
	switch {
	case key.Matches(msg, km.Quit):
		return tea.Quit
	case key.Matches(msg, km.CommandPalette):
		a.openOverlay()
		return nil
	case key.Matches(msg, km.SessionsDirect):
		a.openOverlay()
		a.palette.ShowSessions()
		return nil
	case key.Matches(msg, km.ToggleSidebar):
		a.sidebarAnim.Toggle()
		a.relayout()
		return nil
	case key.Matches(msg, km.NewSession):
		s := session.NewSession("New conversation")
		a.sessions.Add(s)
		a.syncSession()
		return nil
	case key.Matches(msg, km.ToggleRAG):
		if s := a.sessions.Active(); s != nil {
			s.RAG.Active = !s.RAG.Active
		}
		return nil
	case key.Matches(msg, km.FocusInput):
		if a.input.Focused() {
			a.input.Blur()
		} else {
			return a.input.Focus()
		}
		return nil
	case key.Matches(msg, km.Send):
		if a.input.Focused() {
			return a.sendMessage()
		}
		return nil
	}

	return nil
}

func (a *App) openOverlay() {
	a.overlayAnim.Open()
	a.palette.Reset()
	a.input.Blur()
}

func (a *App) closeOverlay() {
	a.overlayAnim.Close()
	a.input.Focus()
}

func (a *App) paletteUp() {
	switch a.palette.CurrentView() {
	case OverlayCommands:
		a.palette.CursorUp(len(a.palette.commands))
	case OverlaySessions:
		a.palette.CursorUp(len(a.sessions.Sessions))
	}
}

func (a *App) paletteDown() {
	switch a.palette.CurrentView() {
	case OverlayCommands:
		a.palette.CursorDown(len(a.palette.commands))
	case OverlaySessions:
		a.palette.CursorDown(len(a.sessions.Sessions))
	}
}

func (a *App) paletteSelect() tea.Cmd {
	switch a.palette.CurrentView() {
	case OverlayCommands:
		cmd := a.palette.SelectedCommand()
		switch cmd.ID {
		case "sessions":
			a.palette.ShowSessions()
		case "new_session":
			s := session.NewSession("New conversation")
			a.sessions.Add(s)
			a.closeOverlay()
			a.syncSession()
		case "toggle_sidebar":
			a.sidebarAnim.Toggle()
			a.closeOverlay()
			a.relayout()
		case "toggle_rag":
			if s := a.sessions.Active(); s != nil {
				s.RAG.Active = !s.RAG.Active
			}
			a.closeOverlay()
		case "quit":
			return tea.Quit
		default:
			a.closeOverlay()
		}
	case OverlaySessions:
		id := a.palette.SelectedSessionID(a.sessions.Sessions)
		if id != "" {
			a.sessions.Switch(id)
			a.syncSession()
		}
		a.closeOverlay()
	}
	return nil
}

func (a *App) sendMessage() tea.Cmd {
	text := a.input.Value()
	if text == "" {
		return nil
	}

	if s := a.sessions.Active(); s != nil {
		s.RAG.SourceCount = 0
		s.RAG.Latency = 0
	}

	a.sessions.AppendMessage(session.Message{
		Role:    session.RoleUser,
		Content: text,
	})
	a.input.Reset()
	a.syncSession()

	// kick off mock pipeline to demonstrate the rendering
	return StartMockPipeline()
}

// syncSession updates the chat pane with current session messages.
func (a *App) syncSession() {
	s := a.sessions.Active()
	if s == nil {
		return
	}
	a.chat.SetMessages(s.Messages)
}

func (a *App) handleAnimTick() (tea.Model, tea.Cmd) {
	a.chat.TickSpinner()
	return a, AnimTick()
}

func (a *App) relayout() {
	sidebarW := a.sidebarAnim.Width(a.windowWidth)

	chatColW := a.windowWidth - sidebarW
	if chatColW < 20 {
		chatColW = 20
	}

	// The split layout governs the vertical space above the help bar.
	contentH := a.windowHeight - HelpBarHeight
	a.splitLayout.SetSize(chatColW, contentH)

	// ── Chat (top panel) ──
	topH := a.splitLayout.TopHeight()
	a.chatContainer.SetSize(chatColW, topH)
	a.chat.SetSize(a.chatContainer.ContentWidth(), a.chatContainer.ContentHeight())

	// ── Input (bottom panel) ──
	bottomH := a.splitLayout.BottomHeight()
	a.inputContainer.SetSize(chatColW, bottomH)
	a.input.SetSize(a.inputContainer.ContentWidth(), a.inputContainer.ContentHeight())

	// ── Sidebar & overlays ──
	a.sidebar.SetSize(sidebarW, a.windowHeight-HelpBarHeight)
	a.helpBar.SetWidth(a.windowWidth)
	a.palette.SetSize(a.windowWidth, a.windowHeight)
}

func (a App) View() tea.View {
	sess := a.sessions.Active()

	// ── Chat area (wrapped in chatContainer) ──
	var chatContent string
	if sess != nil && len(sess.Messages) > 0 {
		chatContent = a.chat.View()
	} else {
		chatContent = a.chat.ViewEmpty()
	}
	chatView := a.chatContainer.Render(chatContent)

	// ── Input area (wrapped in inputContainer with focus-dependent border) ──
	inputContent := a.input.View()

	var inputView string
	if a.input.Focused() {
		inputView = a.inputContainer.Render(inputContent, lipgloss.Color(colorBlue))
	} else {
		inputView = a.inputContainer.Render(inputContent, lipgloss.Color(colorBorder))
	}

	// ── Left column: chat + input ──
	leftCol := lipgloss.JoinVertical(lipgloss.Left,
		chatView,
		inputView,
	)

	// ── Right column: sidebar ──
	sidebarW := a.sidebarAnim.Width(a.windowWidth)
	var layout string
	if sidebarW > 0 {
		rightCol := a.sidebar.View(sess)
		layout = lipgloss.JoinHorizontal(lipgloss.Top, leftCol, rightCol)
	} else {
		layout = leftCol
	}

	// ── Help bar (full width bottom) ──
	helpView := a.helpBar.View(sess)
	fullLayout := lipgloss.JoinVertical(lipgloss.Left, layout, helpView)

	// ── App background ──
	fullScreen := a.styles.App.
		Width(a.windowWidth).
		Height(a.windowHeight).
		Render(fullLayout)

	// ── Overlay (compositor layer) ──
	if a.overlayAnim.IsOpen() {
		overlayContent := a.palette.View(a.sessions.Sessions, a.sessions.ActiveID)
		ox := (a.windowWidth - lipgloss.Width(overlayContent)) / 2
		oy := (a.windowHeight - lipgloss.Height(overlayContent)) / 2
		withOverlay := placeOverlay(ox, oy, overlayContent, fullScreen, true)
		v := tea.NewView(withOverlay)
		v.AltScreen = true
		v.MouseMode = tea.MouseModeCellMotion
		return v
	}

	v := tea.NewView(fullScreen)
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	return v
}
