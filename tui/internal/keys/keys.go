package keys

import "charm.land/bubbles/v2/key"

// GlobalKeyMap holds all app-level keybindings.
type GlobalKeyMap struct {
	// Navigation
	SessionOverlay key.Binding
	NewSession     key.Binding
	Quit           key.Binding

	// Layout
	ToggleSidebar key.Binding

	// Focus
	FocusInput    key.Binding
	FocusMessages key.Binding

	// Chat
	Send       key.Binding
	ScrollUp   key.Binding
	ScrollDown key.Binding

	// RAG
	AttachDoc   key.Binding
	ToggleRAG   key.Binding
	ShowSources key.Binding
}

var DefaultKeyMap = GlobalKeyMap{
	SessionOverlay: key.NewBinding(
		key.WithKeys("ctrl+s"),
		key.WithHelp("ctrl+s", "sessions"),
	),
	NewSession: key.NewBinding(
		key.WithKeys("ctrl+n"),
		key.WithHelp("ctrl+n", "new session"),
	),
	Quit: key.NewBinding(
		key.WithKeys("ctrl+c", "ctrl+q"),
		key.WithHelp("ctrl+c", "quit"),
	),
	ToggleSidebar: key.NewBinding(
		key.WithKeys("ctrl+b"),
		key.WithHelp("ctrl+b", "sidebar"),
	),
	FocusInput: key.NewBinding(
		key.WithKeys("tab"),
		key.WithHelp("tab", "focus"),
	),
	FocusMessages: key.NewBinding(
		key.WithKeys("esc"),
		key.WithHelp("esc", "cancel"),
	),
	Send: key.NewBinding(
		key.WithKeys("enter"),
		key.WithHelp("enter", "send"),
	),
	ScrollUp: key.NewBinding(
		key.WithKeys("up", "k", "pgup"),
		key.WithHelp("↑/k", "scroll up"),
	),
	ScrollDown: key.NewBinding(
		key.WithKeys("down", "j", "pgdn"),
		key.WithHelp("↓/j", "scroll down"),
	),
	AttachDoc: key.NewBinding(
		key.WithKeys("ctrl+a"),
		key.WithHelp("ctrl+a", "attach doc"),
	),
	ToggleRAG: key.NewBinding(
		key.WithKeys("ctrl+r"),
		key.WithHelp("ctrl+r", "toggle RAG"),
	),
	ShowSources: key.NewBinding(
		key.WithKeys("ctrl+e"),
		key.WithHelp("ctrl+e", "sources"),
	),
}

// ShortHelp returns key bindings for the persistent help bar.
func (k GlobalKeyMap) ShortHelp() []key.Binding {
	return []key.Binding{
		k.FocusMessages, k.FocusInput, k.SessionOverlay,
		k.ToggleSidebar, k.ToggleRAG, k.Quit,
	}
}

// FullHelp returns all key bindings grouped by category.
func (k GlobalKeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Send, k.FocusInput, k.FocusMessages},
		{k.SessionOverlay, k.NewSession, k.ToggleSidebar},
		{k.AttachDoc, k.ToggleRAG, k.ShowSources},
		{k.ScrollUp, k.ScrollDown},
		{k.Quit},
	}
}
