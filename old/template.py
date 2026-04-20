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
    ]

    FOCUS_DIMENSIONS = [
        "cost",
        "comfort",
        "time",
        "safety",
        "flexibility_focus",
    ]

    def __init__(self) -> None:
        self.config_dir = "configs"
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

        # Separate assignment so Pylance can narrow the type before calling .get().
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

        # Inline style interpretation.
        if persona["friendliness"] <= 2:
            style = "grumpy"
        elif persona["assertiveness"] >= 4:
            style = "direct"
        elif persona["friendliness"] >= 4:
            style = "friendly"
        else:
            style = "neutral"

        if persona["talkativeness"] >= 4:
            participation = "active"
        elif persona["talkativeness"] <= 2:
            participation = "reserved"
        else:
            participation = "balanced"

        primary_focus = max(focus, key=lambda k: focus[k])

        traits_str = ", ".join(f"{t} {persona[t]}" for t in self.SCALAR_TRAITS)

        return (
            f"You are {persona['name']}. "
            f"Role: {persona['role']}. "
            f"{primary_text} "
            f"Traits (1-5): {traits_str}. "
            f"Focus: {focus_str}. "
            f"Style: {style}, participation: {participation}, "
            f"primary focus: {primary_focus}. "
            f"Behave consistently with these traits."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create_persona(
        self,
        name: str,
        role: Optional[str] = None,
        is_primary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        file_path = self._persona_file(name)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                persona: Dict[str, Any] = json.load(f)
        else:
            persona = {
                "name": name,
                "goal": None,
                "role": "participant",
                "is_primary": False,
                **{trait: self._rand() for trait in self.SCALAR_TRAITS},
                "focus": {dim: self._rand() for dim in self.FOCUS_DIMENSIONS},
            }

        # Apply overrides before normalizing so behavior text is correct.
        if role is not None:
            persona["role"] = role
        if is_primary is not None:
            persona["is_primary"] = is_primary

        persona = self._normalize_persona(persona, name)
        self._save_persona(persona)
        return persona

    def _save_persona(self, persona: Dict[str, Any]) -> None:
        """Save a persona that has already been normalized."""
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

        # Enforce exactly one primary participant.
        primaries = [n for n, v in result.items() if v["is_primary"]]
        if len(primaries) > 1:
            for n in primaries[1:]:
                result[n]["is_primary"] = False
        elif not primaries and names:
            result[names[0]]["is_primary"] = True

        return result
