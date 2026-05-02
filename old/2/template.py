import json
import os
import random
from typing import Any, Dict, List, Optional


class PersonaManager:
    SCALAR_TRAITS = [
        "friendliness",
        "assertiveness",
        "talkativeness",
        "initiative",
        "agreeableness",
        "flexibility",
        "patience",
        "response_length",
        "contrarian_pressure",   # NEW: 1=always goes with group, 5=natural devil's advocate
    ]

    FOCUS_DIMENSIONS = [
        "cost",
        "comfort",
        "time",
        "safety",
        "flexibility_focus",
    ]

    def __init__(self, dialogue_id: str) -> None:
        # Each dialogue gets its own subfolder: configs/<dialogue_id>/
        self.dialogue_id = dialogue_id
        self.config_dir = os.path.join("configs", dialogue_id)
        os.makedirs(self.config_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persona_file(self, name: str) -> str:
        return os.path.join(self.config_dir, f"{name.lower()}.json")

    def _rand(self) -> int:
        return random.randint(1, 5)

    def _clamp(self, value: Any, default: int = 3) -> int:
        if not isinstance(value, int):
            return default
        return max(1, min(5, value))

    def _normalize_persona(self, persona: Dict[str, Any], name: str) -> Dict[str, Any]:
        """Ensure all required fields exist and values are in range."""
        persona["name"] = name
        persona.setdefault("role", "participant")
        persona.setdefault("is_primary", False)
        persona.setdefault("goal", None)

        for trait in self.SCALAR_TRAITS:
            persona[trait] = self._clamp(persona.get(trait), default=self._rand())

        focus_raw: Any = persona.get("focus")
        raw_focus: Dict[str, Any] = focus_raw if isinstance(focus_raw, dict) else {}
        persona["focus"] = {
            dim: self._clamp(raw_focus.get(dim), default=self._rand())
            for dim in self.FOCUS_DIMENSIONS
        }

        persona["behavior"] = self._build_behavior_text(persona)
        return persona

    def _build_behavior_text(self, persona: Dict[str, Any]) -> str:
        primary_text = (
            "You are the main person in this scenario."
            if persona["is_primary"]
            else "You are not the main person in this scenario."
        )

        focus: Dict[str, int] = persona["focus"]
        focus_str = ", ".join(f"{k}:{v}" for k, v in focus.items())

        # Tone style
        if persona["friendliness"] <= 2:
            style = "grumpy"
        elif persona["assertiveness"] >= 4:
            style = "direct"
        elif persona["friendliness"] >= 4:
            style = "friendly"
        else:
            style = "neutral"

        # Participation level
        if persona["talkativeness"] >= 4:
            participation = "active"
        elif persona["talkativeness"] <= 2:
            participation = "reserved"
        else:
            participation = "balanced"

        # Contrarian tendency
        contrarian = persona.get("contrarian_pressure", 3)
        if contrarian >= 4:
            contrarian_desc = "devil's advocate"
        elif contrarian <= 2:
            contrarian_desc = "consensus-seeker"
        else:
            contrarian_desc = "moderate"

        primary_focus = max(focus, key=lambda k: focus[k])
        traits_str = ", ".join(f"{t} {persona[t]}" for t in self.SCALAR_TRAITS)

        return (
            f"You are {persona['name']}. "
            f"Role: {persona['role']}. "
            f"{primary_text} "
            f"Traits (1-5): {traits_str}. "
            f"Focus: {focus_str}. "
            f"Style: {style}, participation: {participation}, "
            f"contrarian tendency: {contrarian_desc}, "
            f"primary focus: {primary_focus}. "
            f"Behave consistently with these traits."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_path(self, source_path: str, name: str) -> Dict[str, Any]:
        """
        Load a persona from an explicit file path (e.g. a path the user typed
        at startup). The loaded file is copied into this dialogue's config
        folder so the run is self-contained, then normalized and saved.
        """
        with open(source_path, "r", encoding="utf-8") as f:
            persona: Dict[str, Any] = json.load(f)

        # Use the display name the user typed, not whatever is in the file.
        persona = self._normalize_persona(persona, name)
        self._save_persona(persona)
        return persona

    def create_fresh(
        self,
        name: str,
        role: Optional[str] = None,
        is_primary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Always build a brand-new persona with random traits."""
        persona: Dict[str, Any] = {
            "name": name,
            "goal": None,
            "role": role or "participant",
            "is_primary": is_primary if is_primary is not None else False,
            **{trait: self._rand() for trait in self.SCALAR_TRAITS},
            "focus": {dim: self._rand() for dim in self.FOCUS_DIMENSIONS},
        }
        persona = self._normalize_persona(persona, name)
        self._save_persona(persona)
        return persona

    def apply_role(
        self,
        persona: Dict[str, Any],
        role: str,
        is_primary: bool,
    ) -> Dict[str, Any]:
        """Stamp role/primary onto an already-loaded persona and re-save."""
        persona["role"] = role
        persona["is_primary"] = is_primary
        persona = self._normalize_persona(persona, persona["name"])
        self._save_persona(persona)
        return persona

    def _save_persona(self, persona: Dict[str, Any]) -> None:
        file_path = self._persona_file(persona["name"])
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=4)

    def save_persona(self, persona: Dict[str, Any]) -> None:
        """Public save — normalizes before writing."""
        persona = self._normalize_persona(persona, persona["name"])
        self._save_persona(persona)

    def assign_roles(
        self,
        names: List[str],
        generated_roles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {
            name: {"role": "participant", "is_primary": False}
            for name in names
        }

        if generated_roles:
            for name in names:
                if name in generated_roles:
                    info = generated_roles[name]
                    result[name]["role"] = str(info.get("role", "participant"))
                    result[name]["is_primary"] = bool(info.get("is_primary", False))

        primaries = [n for n, v in result.items() if v["is_primary"]]
        if len(primaries) > 1:
            for n in primaries[1:]:
                result[n]["is_primary"] = False
        elif not primaries and names:
            result[names[0]]["is_primary"] = True

        return result
