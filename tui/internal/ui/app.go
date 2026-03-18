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
	modelBar ModelBar
	chat     ChatPane
	input    ChatInput
	sidebar  Sidebar
	helpBar  HelpBar
	palette  CommandPalette

	// state
	sessions     *session.Manager
	sidebarAnim  SidebarAnim
	overlayAnim  OverlayAnim
	animRunning  bool
	overlayOpen  bool
	windowWidth  int
	windowHeight int
}

func NewApp() App {
	styles := NewStyles()
	mgr := session.NewManager()

	return App{
		styles:      styles,
		modelBar:    NewModelBar(styles),
		chat:        NewChatPane(styles),
		input:       NewChatInput(styles),
		sidebar:     NewSidebar(styles),
		helpBar:     NewHelpBar(styles),
		palette:     NewCommandPalette(styles),
		sessions:    mgr,
		sidebarAnim: NewSidebarAnim(SidebarWidth),
		overlayAnim: NewOverlayAnim(),
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

	case tea.KeyPressMsg:
		if cmd := a.handleKey(msg); cmd != nil {
			cmds = append(cmds, cmd)
		}
		// if overlay is open, don't forward keys to input/chat
		if a.overlayOpen {
			return a, tea.Batch(cmds...)
		}
	}

	// forward to input
	if !a.overlayOpen {
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
	if a.overlayOpen {
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
	case key.Matches(msg, km.SessionOverlay):
		a.openOverlay()
		return nil
	case key.Matches(msg, km.ToggleSidebar):
		a.sidebarAnim.Toggle()
		a.startAnim()
		return nil
	case key.Matches(msg, km.NewSession):
		s := session.NewSession("New conversation")
		a.sessions.Add(s)
		a.syncSession()
		return nil
	case key.Matches(msg, km.ToggleRAG):
		if s := a.sessions.Active(); s != nil {
			s.RAGEnabled = !s.RAGEnabled
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
	a.overlayOpen = true
	a.overlayAnim.Open()
	a.palette.Reset()
	a.input.Blur()
	a.startAnim()
}

func (a *App) closeOverlay() {
	a.overlayOpen = false
	a.overlayAnim.Close()
	a.startAnim()
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
			a.startAnim()
		case "toggle_rag":
			if s := a.sessions.Active(); s != nil {
				s.RAGEnabled = !s.RAGEnabled
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

	a.sessions.AppendMessage(session.Message{
		Role:    session.RoleUser,
		Content: text,
	})
	a.input.Reset()
	a.syncSession()
	return nil
}

// syncSession updates the chat pane with current session messages.
func (a *App) syncSession() {
	s := a.sessions.Active()
	if s == nil {
		return
	}
	a.chat.SetMessages(s.Messages)
}

func (a *App) startAnim() {
	a.animRunning = true
}

func (a *App) handleAnimTick() (tea.Model, tea.Cmd) {
	// Step all active animations
	a.sidebarAnim.Step()
	a.overlayAnim.Step()
	a.chat.TickSpinner()

	// Relayout with animated sidebar width
	a.relayout()

	// Check if all animations are at rest
	allRest := a.sidebarAnim.AtRest() && a.overlayAnim.AtRest()
	if allRest && !a.animRunning {
		// Don't stop the tick - spinner always needs it
	}
	a.animRunning = !allRest

	return a, AnimTick()
}

func (a *App) relayout() {
	sidebarW := a.sidebarAnim.Width()

	chatColW := a.windowWidth - sidebarW
	if chatColW < 20 {
		chatColW = 20
	}

	// heights: modelbar + chat + input + helpbar
	chatH := a.windowHeight - ModelBarHeight - InputMinHeight - HelpBarHeight
	if chatH < 4 {
		chatH = 4
	}

	a.modelBar.SetWidth(chatColW)
	a.chat.SetSize(chatColW, chatH)
	a.input.SetWidth(chatColW)
	a.sidebar.SetSize(sidebarW, a.windowHeight-HelpBarHeight)
	a.helpBar.SetWidth(a.windowWidth)
	a.palette.SetSize(a.windowWidth, a.windowHeight)
}

func (a App) View() tea.View {
	sess := a.sessions.Active()

	// ── Model bar ──
	modelBarView := a.modelBar.View(sess)

	// ── Chat area ──
	var chatView string
	if sess != nil && len(sess.Messages) > 0 {
		chatView = a.chat.View()
	} else {
		chatView = a.chat.ViewEmpty()
	}

	// ── Input area ──
	ragEnabled := sess != nil && sess.RAGEnabled
	isStreaming := sess != nil && len(sess.Messages) > 0 && sess.Messages[len(sess.Messages)-1].IsStreaming
	inputView := a.input.View(ragEnabled, isStreaming)

	// ── Left column: modelbar + chat + input ──
	leftCol := lipgloss.JoinVertical(lipgloss.Left,
		modelBarView,
		chatView,
		inputView,
	)

	// ── Right column: sidebar (animated width) ──
	sidebarW := a.sidebarAnim.Width()
	var layout string
	if sidebarW > 0 {
		rightCol := a.sidebar.View()
		layout = lipgloss.JoinHorizontal(lipgloss.Top, leftCol, rightCol)
	} else {
		layout = leftCol
	}

	// ── Help bar (full width bottom) ──
	helpView := a.helpBar.View()
	fullLayout := lipgloss.JoinVertical(lipgloss.Left, layout, helpView)

	// ── App background ──
	fullScreen := a.styles.App.
		Width(a.windowWidth).
		Height(a.windowHeight).
		Render(fullLayout)

	// ── Overlay (compositor layer) ──
	if a.overlayOpen || !a.overlayAnim.AtRest() {
		progress := a.overlayAnim.Progress()
		if progress > 0.01 {
			overlayContent := a.palette.View(a.sessions.Sessions, a.sessions.ActiveID)

			// Center the overlay
			overlayW := lipgloss.Width(overlayContent)
			overlayH := lipgloss.Height(overlayContent)
			ox := (a.windowWidth - overlayW) / 2
			oy := (a.windowHeight-overlayH)/2 - int(float64(3)*(1.0-progress))

			baseLayer := lipgloss.NewLayer(fullScreen)
			overlayLayer := lipgloss.NewLayer(overlayContent).
				X(ox).
				Y(oy).
				Z(1)

			comp := lipgloss.NewCompositor(baseLayer, overlayLayer)
			v := tea.NewView(comp.Render())
			v.AltScreen = true
			v.MouseMode = tea.MouseModeCellMotion
			return v
		}
	}

	v := tea.NewView(fullScreen)
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	return v
}
