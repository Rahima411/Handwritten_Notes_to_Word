"""Small opt-in local memory for user preferences."""

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_MEMORY = {
    "preferred_output_mode": "structured",
    "preserve_chemistry": True,
    "notes": [],
}


class AgentMemory:
    """Stores non-sensitive user preferences only when the user opts in."""

    def __init__(self, path: str = ".agent_memory.json"):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return DEFAULT_MEMORY.copy()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_MEMORY, **data}
        except (OSError, json.JSONDecodeError):
            return DEFAULT_MEMORY.copy()

    def save_preferences(self, preferences: Dict[str, Any]) -> None:
        data = self.load()
        allowed_keys = {"preferred_output_mode", "preserve_chemistry", "notes"}
        for key, value in preferences.items():
            if key in allowed_keys:
                data[key] = value
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def forget(self) -> None:
        if self.path.exists():
            self.path.unlink()
