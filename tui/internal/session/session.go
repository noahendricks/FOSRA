package session

import (
	"time"

	"charm.land/lipgloss/v2"
)

// Role identifies who sent a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Source is a RAG document chunk that was referenced.
type Source struct {
	DocName string
	Excerpt string
	Score   float64 // relevance score 0–1
	Page    int
	ChunkID string
}

// Message is a single chat turn.
type Message struct {
	ID          string
	Role        Role
	Content     string
	Sources     []Source // populated for assistant messages when RAG is on
	Timestamp   time.Time
	IsStreaming bool   // true while the LLM is still writing this message
	Error       string // non-empty if generation failed
}

// Session is a named conversation.
type Session struct {
	ID         string
	Title      string
	Messages   []Message
	RAGEnabled bool
	ModelName  string
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

// Manager holds all sessions and tracks the active one.
type Manager struct {
	Sessions []*Session
	ActiveID string
}

// NewManager creates a Manager with one blank session.
func NewManager() *Manager {
	s := NewSession("New conversation")
	return &Manager{
		Sessions: []*Session{s},
		ActiveID: s.ID,
	}
}

var sessionCounter int

// NewSession creates a blank session with a generated ID.
func NewSession(title string) *Session {
	sessionCounter++
	return &Session{
		ID:         time.Now().Format("20060102-150405"),
		Title:      title,
		RAGEnabled: true,
		ModelName:  "gpt-4o",
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
}

// Active returns the currently active session, or nil.
func (m *Manager) Active() *Session {
	for _, s := range m.Sessions {
		if s.ID == m.ActiveID {
			return s
		}
	}
	return nil
}

// Add appends a session and makes it active.
func (m *Manager) Add(s *Session) {
	m.Sessions = append(m.Sessions, s)
	m.ActiveID = s.ID
}

// Switch changes the active session.
func (m *Manager) Switch(id string) {
	m.ActiveID = id
}

// AppendMessage adds a message to the active session.
func (m *Manager) AppendMessage(msg Message) {
	s := m.Active()
	if s == nil {
		return
	}
	s.Messages = append(s.Messages, msg)
	s.UpdatedAt = time.Now()
}

// UpdateLastMessage replaces the last message (used for streaming).
func (m *Manager) UpdateLastMessage(msg Message) {
	s := m.Active()
	if s == nil || len(s.Messages) == 0 {
		return
	}
	s.Messages[len(s.Messages)-1] = msg
}

// ---- Lipgloss style helpers used in list rendering ----

var (
	ActiveDot   = lipgloss.NewStyle().Foreground(lipgloss.Color("#7DCFFF")).Render("●")
	InactiveDot = lipgloss.NewStyle().Foreground(lipgloss.Color("#414868")).Render("○")
)
