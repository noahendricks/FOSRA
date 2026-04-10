"""
skill registry endpoint.

returns available skills by scanning .md skill files in the user
FOSRA directory (~/fosra/skills/) and the backend directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from backend.src.settings.fosra_paths import fosra_paths

router = APIRouter(prefix="/oc/skill", tags=["Skills"])


def _parse_skill_file(path: Path) -> dict[str, Any]:
    content = path.read_text()
    name = path.stem.replace("_skill", "").replace("_", " ").title()
    description = ""
    triggers: list[str] = []

    for line in content.splitlines():
        if line.startswith("# "):
            if not description:
                description = line[2:].strip()
        elif line.startswith("## Triggers"):
            continue
        elif line.startswith("- "):
            triggers.append(line[2:].strip().lstrip("*").strip())

    return {
        "name": name,
        "description": description,
        "triggers": triggers,
        "file": str(path.name),
    }


# @router.get("")
# async def list_skills():
#     """
#     Return all available skills with name, description, and triggers.
#     Scans ~/fosra/skills/ and backend for .skill.md files.
#     """
#     skills: list[dict[str, Any]] = []
#     seen_names: set[str] = set()
#
#     for base_dir in [
#         fosra_paths.skills_dir,
#         Path(__file__).parent.parent.parent.parent,
#     ]:
#         if not base_dir.exists():
#             continue
#         for path in base_dir.rglob("*.skill.md"):
#             try:
#                 skill = _parse_skill_file(path)
#                 if skill["name"] not in seen_names:
#                     seen_names.add(skill["name"])
#                     skills.append(skill)
#             except Exception:
#                 pass
#
#     return skills
