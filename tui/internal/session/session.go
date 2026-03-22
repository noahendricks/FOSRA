package session

import (
	"fmt"
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
	Score   float64 // relevance score 0-1
	Page    int
	ChunkID string
}

// ToolCall represents a tool/function invocation by an agent.
type ToolCall struct {
	Name   string // e.g. "search_codebase", "read_file"
	Args   string // human-readable argument summary
	Output string // tool output content
	Status string // "running", "done", "error"
}

// TodoStatus tracks individual task progress.
type TodoStatus string

const (
	TodoStatusPending    TodoStatus = "pending"
	TodoStatusInProgress TodoStatus = "in_progress"
	TodoStatusDone       TodoStatus = "done"
)

// TodoItem is a single task in an agent's task list.
type TodoItem struct {
	Text   string
	Status TodoStatus
}

// message is a single chat turn.
type Message struct {
	ID        string
	Role      Role
	Content   string
	Timestamp time.Time

	// Streaming state
	IsStreaming bool
	Error       string

	// Completion metadata
	Duration    time.Duration // time from user send to assistant complete
	CompletedAt time.Time
	Mode        string // agent mode name (e.g. "Build", "Chat")
	ModelID     string // full model identifier
	Interrupted bool

	// Rich content blocks (populated by agent/LLM)
	Sources    []Source   // RAG source citations
	ToolCalls  []ToolCall // tool invocations with output
	Todos      []TodoItem // agent task list
	ThinkingMs int        // thinking duration in ms (0 = not shown)
}

// RAGState tracks the current retrieval-augmented generation status.
type RAGState struct {
	IndexName   string        // name of the active index (e.g. "project-docs")
	Active      bool          // whether RAG retrieval is enabled
	SourceCount int           // number of sources retrieved for current query
	Latency     time.Duration // last retrieval latency
}

// Session is a named conversation.
type Session struct {
	ID       string
	Title    string
	Messages []Message
	RAG      RAGState

	// model info
	ModelName    string
	Provider     string
	ContextUsed  float64 // 0-1 fraction of context window used
	ContextTotal int     // total context tokens
	Cost         float64 // running cost in dollars

	CreatedAt time.Time
	UpdatedAt time.Time
}

// manager holds all sessions and tracks the active one.
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
		ID:    fmt.Sprintf("%s-%d", time.Now().Format("20060102-150405"), sessionCounter),
		Title: title,
		RAG: RAGState{
			IndexName: "project-docs",
			Active:    true,
		},
		ModelName: "gpt-4o",
		Provider:  "OpenAI",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// active returns the currently active session, or nil.
func (m *Manager) Active() *Session {
	for _, s := range m.Sessions {
		if s.ID == m.ActiveID {
			return s
		}
	}
	return nil
}

// add appends a session and makes it active.
func (m *Manager) Add(s *Session) {
	m.Sessions = append(m.Sessions, s)
	m.ActiveID = s.ID
}

// switch changes the active session.
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

// style helpers for list rendering.
var (
	ActiveDot   = lipgloss.NewStyle().Foreground(lipgloss.Color("#7DCFFF")).Render("●")
	InactiveDot = lipgloss.NewStyle().Foreground(lipgloss.Color("#414868")).Render("○")
)
