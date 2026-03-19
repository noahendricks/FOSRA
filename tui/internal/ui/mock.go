package ui

import (
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/roccoluxe/fosra-tui/tui/internal/session"
)

// ── Mock message types ───────────────────────────────────────────────
// mockPipelineStep drives the sequenced simulation.
// mockStreamChunk carries token-by-token streaming state.

type mockPipelineStep struct{ step int }

type mockStreamChunk struct {
	token     string
	remaining string
	interval  time.Duration
}

// ── Canned response data ─────────────────────────────────────────────

var mockResponse = "The middleware chain in this codebase follows a standard " +
	"onion model. Each handler wraps the next, processing the request on the " +
	"way in and the response on the way out.\n\n" +
	"The key middleware layers are:\n" +
	"1. **Logging** - captures request/response timing\n" +
	"2. **Auth** - validates JWT tokens and sets user context\n" +
	"3. **RateLimit** - enforces per-IP request limits\n" +
	"4. **CORS** - handles cross-origin preflight\n\n" +
	"The order matters because auth depends on the request being logged, " +
	"and rate limiting should happen after auth to avoid limiting " +
	"unauthenticated requests that will be rejected anyway."

var mockSources = []session.Source{
	{DocName: "middleware.go", Score: 0.94, Page: 1, ChunkID: "mw-001"},
	{DocName: "server.go", Score: 0.87, Page: 1, ChunkID: "srv-002"},
	{DocName: "auth.go", Score: 0.72, Page: 1, ChunkID: "auth-003"},
}

var mockTodosPhase1 = []session.TodoItem{
	{Text: "Search codebase for middleware", Status: session.TodoStatusInProgress},
	{Text: "Analyze middleware chain order", Status: session.TodoStatusPending},
	{Text: "Summarize findings", Status: session.TodoStatusPending},
}

var mockTodosPhase2 = []session.TodoItem{
	{Text: "Search codebase for middleware", Status: session.TodoStatusDone},
	{Text: "Analyze middleware chain order", Status: session.TodoStatusInProgress},
	{Text: "Summarize findings", Status: session.TodoStatusPending},
}

var mockTodosPhase3 = []session.TodoItem{
	{Text: "Search codebase for middleware", Status: session.TodoStatusDone},
	{Text: "Analyze middleware chain order", Status: session.TodoStatusDone},
	{Text: "Summarize findings", Status: session.TodoStatusInProgress},
}

// ── Pipeline launcher ────────────────────────────────────────────────

// StartMockPipeline returns a cmd that kicks off the timed mock sequence.
func StartMockPipeline() tea.Cmd {
	return func() tea.Msg {
		return mockPipelineStep{step: 0}
	}
}

// mockDelay returns a cmd that waits then sends a message.
func mockDelay(d time.Duration, msg tea.Msg) tea.Cmd {
	return tea.Tick(d, func(time.Time) tea.Msg { return msg })
}

// ── Pipeline step handler (method on *App) ───────────────────────────

func (a *App) handleMockStep(step int) (tea.Model, tea.Cmd) {
	s := a.sessions.Active()
	if s == nil {
		return a, nil
	}

	switch step {
	case 0: // create streaming assistant message with thinking indicator
		s.RAG.IndexName = "middleware-index"
		s.RAG.SourceCount = 0
		s.RAG.Latency = 0
		s.Messages = append(s.Messages, session.Message{
			Role:        session.RoleAssistant,
			IsStreaming: true,
			ThinkingMs:  2300,
		})
		a.syncSession()
		return a, mockDelay(800*time.Millisecond, mockPipelineStep{step: 1})

	case 1: // grep tool call (running)
		last := &s.Messages[len(s.Messages)-1]
		last.ToolCalls = append(last.ToolCalls, session.ToolCall{
			Name:   "Grep",
			Args:   `"middleware" in .`,
			Status: "running",
		})
		a.syncSession()
		return a, mockDelay(700*time.Millisecond, mockPipelineStep{step: 2})

	case 2: // grep done + read running
		last := &s.Messages[len(s.Messages)-1]
		last.ToolCalls[0].Status = "done"
		last.ToolCalls[0].Output = "Found 9 matches across 4 files"
		last.ToolCalls = append(last.ToolCalls, session.ToolCall{
			Name:   "Read",
			Args:   "internal/server/middleware.go",
			Status: "running",
		})
		a.syncSession()
		return a, mockDelay(600*time.Millisecond, mockPipelineStep{step: 3})

	case 3: // read done
		last := &s.Messages[len(s.Messages)-1]
		last.ToolCalls[1].Status = "done"
		last.ToolCalls[1].Output = "func SetupMiddleware(r *mux.Router) { ... }"
		s.RAG.Latency = 184 * time.Millisecond
		a.syncSession()
		return a, mockDelay(300*time.Millisecond, mockPipelineStep{step: 4})

	case 4: // todos phase 1
		last := &s.Messages[len(s.Messages)-1]
		last.Todos = mockTodosPhase1
		a.syncSession()
		return a, mockDelay(500*time.Millisecond, mockPipelineStep{step: 5})

	case 5: // todos phase 2
		last := &s.Messages[len(s.Messages)-1]
		last.Todos = mockTodosPhase2
		a.syncSession()
		return a, mockDelay(500*time.Millisecond, mockPipelineStep{step: 6})

	case 6: // todos phase 3 + begin streaming
		last := &s.Messages[len(s.Messages)-1]
		last.Todos = mockTodosPhase3
		a.syncSession()
		return a, mockDelay(300*time.Millisecond, mockPipelineStep{step: 7})

	case 7: // kick off token streaming
		chunkSize := 3
		chunk := mockResponse
		if len(chunk) > chunkSize {
			chunk = mockResponse[:chunkSize]
		}
		return a, func() tea.Msg {
			return mockStreamChunk{
				token:     chunk,
				remaining: mockResponse[len(chunk):],
				interval:  30 * time.Millisecond,
			}
		}
	}

	return a, nil
}

// handleMockStream appends a token chunk and schedules the next one.
func (a *App) handleMockStream(msg mockStreamChunk) (tea.Model, tea.Cmd) {
	s := a.sessions.Active()
	if s == nil || len(s.Messages) == 0 {
		return a, nil
	}

	last := &s.Messages[len(s.Messages)-1]
	last.Content += msg.token
	a.syncSession()

	// more tokens to send
	if msg.remaining != "" {
		chunkSize := 3
		next := msg.remaining
		if len(next) > chunkSize {
			next = msg.remaining[:chunkSize]
		}
		rest := msg.remaining[len(next):]
		interval := msg.interval

		return a, tea.Tick(interval, func(time.Time) tea.Msg {
			return mockStreamChunk{
				token:     next,
				remaining: rest,
				interval:  interval,
			}
		})
	}

	// streaming complete - finalize the message
	last.IsStreaming = false
	last.Todos = []session.TodoItem{
		{Text: "Search codebase for middleware", Status: session.TodoStatusDone},
		{Text: "Analyze middleware chain order", Status: session.TodoStatusDone},
		{Text: "Summarize findings", Status: session.TodoStatusDone},
	}
	last.Sources = mockSources
	s.RAG.SourceCount = len(mockSources)
	a.syncSession()
	return a, nil
}
