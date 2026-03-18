package keys

import "charm.land/bubbles/v2/key"

// GlobalKeyMap holds all app-level keybindings.
type GlobalKeyMap struct {
	// Navigation
	SessionOverlay key.Binding
	NewSession     key.Binding
	Quit           key.Binding
	Help           key.Binding

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

// DefaultKeyMap returns sensible defaults.
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
	Help: key.NewBinding(
		key.WithKeys("?"),
		key.WithHelp("?", "help"),
	),
	FocusInput: key.NewBinding(
		key.WithKeys("i", "tab"),
		key.WithHelp("i/tab", "focus input"),
	),
	FocusMessages: key.NewBinding(
		key.WithKeys("esc"),
		key.WithHelp("esc", "focus messages"),
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

// ShortHelp implements help.KeyMap.
func (k GlobalKeyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Send, k.SessionOverlay, k.ToggleRAG, k.Help, k.Quit}
}

// FullHelp implements help.KeyMap.
func (k GlobalKeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Send, k.FocusInput, k.FocusMessages},
		{k.SessionOverlay, k.NewSession},
		{k.AttachDoc, k.ToggleRAG, k.ShowSources},
		{k.ScrollUp, k.ScrollDown},
		{k.Help, k.Quit},
	}
}
