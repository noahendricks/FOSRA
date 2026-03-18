package ui

import (
	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/keys"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

type App struct {
	styles       Styles
	header       Header
	chat         ChatPane
	sidebar      Sidebar
	overlay      SessionOverlay
	chatInput    TextInput
	windowWidth  int
	windowHeight int
}

func NewApp() App {
	styles := NewStyles()

	return App{
		styles:    styles,
		header:    NewHeader(styles),
		chat:      NewChatPane(styles),
		sidebar:   NewSidebar(styles),
		overlay:   NewSessionOverlay(styles),
		chatInput: NewTextInput(true),
	}
}

func (a App) Init() tea.Cmd {
	return textinput.Blink
}

func (a App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.windowWidth = msg.Width
		a.windowHeight = msg.Height

		// decide whether sidebar is visible
		showSidebar := a.windowWidth >= MinWidthForSidebar

		sidebarW := 0

		if showSidebar {
			sidebarW = SidebarWidth
		}

		// left column width: total minus sidebar and gap
		gap := 0
		if showSidebar {
			gap = 1
		}

		chatColW := a.windowWidth - sidebarW - gap

		if chatColW < 20 {
			chatColW = 20
		}

		// chat height: total minus header and input
		chatH := a.windowHeight - HeaderHeight - InputHeight

		if chatH < 4 {
			chatH = 4
		}

		a.header.SetWidth(chatColW)
		a.chat.SetSize(chatColW, chatH)
		a.chatInput.SetWidth(chatColW)
		a.sidebar.SetSize(sidebarW, a.windowHeight)
		a.overlay.SetSize(a.windowWidth, a.windowHeight)

	case tea.KeyPressMsg:
		defKeys := keys.DefaultKeyMap
		switch {
		case key.Matches(msg, defKeys.SessionOverlay):
			a.overlay.Toggle()
		case key.Matches(msg, defKeys.Quit):
			return a, tea.Quit
		case key.Matches(msg, defKeys.FocusInput):
			a.chatInput.ToggleFocus()
		}
	}

	var cmd tea.Cmd
	a.chatInput.input, cmd = a.chatInput.input.Update(msg)
	cmds = append(cmds, cmd)

	return a, tea.Batch(cmds...)
}

func (a App) View() tea.View {
	var messages []session.Message

	// ── left column: header + chat + input (stacked vertically) ──
	headerView := a.header.View()

	chatView := a.chat.View(messages)

	inputView := a.chatInput.View(true, a.chatInput.isStreaming)

	leftCol := lipgloss.JoinVertical(lipgloss.Left,
		headerView,
		chatView,
		inputView,
	)

	// ── right column: sidebar (full height) ──
	showSidebar := a.windowWidth >= MinWidthForSidebar

	var layout string
	if showSidebar {
		rightCol := a.sidebar.View()
		layout = lipgloss.JoinHorizontal(lipgloss.Top, leftCol, rightCol)
	} else {
		layout = leftCol
	}

	// ── apply app background ──
	fullScreen := a.styles.App.
		Width(a.windowWidth).
		Height(a.windowHeight).
		Render(layout)

	// ── overlay (compositor layer on top if active) ──
	if a.overlay.active {
		baseLayer := lipgloss.NewLayer(fullScreen)
		overlayLayer := lipgloss.NewLayer(a.overlay.View()).
			X(a.windowWidth / 3).
			Y(a.windowHeight / 4)

		comp := lipgloss.NewCompositor(baseLayer, overlayLayer)
		v := tea.NewView(comp.Render())
		v.AltScreen = true
		return v
	}

	v := tea.NewView(fullScreen)
	v.AltScreen = true
	return v
}
