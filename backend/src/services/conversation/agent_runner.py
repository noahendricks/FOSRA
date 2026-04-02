"""
background agent execution with tui event emission.

Delegates to backend.src.services.conversation.runner.
Kept for backwards compatibility — prefer importing from runner directly.
"""

from backend.src.services.conversation.runner import run_agent_with_events

__all__ = ["run_agent_with_events"]
