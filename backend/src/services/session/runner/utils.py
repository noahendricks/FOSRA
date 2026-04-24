import json
from typing import Any


def _unwrap_json_text(text: str) -> str:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            str_vals = [v for v in parsed.values() if isinstance(v, str)]
            if len(str_vals) == 1:
                return str_vals[0]
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _parse_todo_output(content: str) -> list[dict[str, Any]]:
    """parse TodoWrite tool output into a list of Todo dicts."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "todos" in parsed:
            return parsed["todos"]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = content.strip().split("\n")
    todos = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- [ ]"):
            todos.append(
                {"content": line[5:].strip(), "status": "pending", "priority": "medium"}
            )
        elif line.startswith("- [x]") or line.startswith("- [X]"):
            todos.append(
                {
                    "content": line[5:].strip(),
                    "status": "completed",
                    "priority": "medium",
                }
            )
        elif line.startswith("-"):
            todos.append(
                {"content": line[1:].strip(), "status": "pending", "priority": "medium"}
            )
    return todos
