from typing import Dict, List

from modules.llm_client import get_llm_client


class RolePlanner:
    """
    Generates topic-aligned participant roles in one LLM call.

    Output shape:
        {
            "Ami": {"role": "decision_owner", "is_primary": True},
            "Tim": {"role": "budget_focused_peer", "is_primary": False},
        }
    """

    def __init__(self):
        self.llm = get_llm_client()

    def plan_roles(self, topic: str, names: List[str]) -> Dict[str, Dict]:
        if not names:
            return {}

        names_str = ", ".join(names)
        first = names[0]

        prompt = f"""
You are assigning discussion roles for a multi-person simulation.

Topic:
{topic}

Participants:
{names_str}

Task:
Assign one role to each participant so the roles fit the topic naturally.

Requirements:
- Return valid JSON only, using exactly this schema:
{{
  "roles": {{
    "{first}": {{"role": "short_role_name", "is_primary": true}},
    "OTHER_NAME": {{"role": "short_role_name", "is_primary": false}}
  }}
}}
- Every listed participant must appear exactly once.
- Roles must be generic and topic-aligned (e.g. "budget_conscious_traveler").
- Use short role names with underscores, no spaces.
- Exactly one participant must have "is_primary": true.
- The primary participant should be the person most directly affected by the decision.
- Do not invent new participants.
- Do not add explanations outside the JSON.
"""

        fallback = {
            name: {"role": "participant", "is_primary": (name == names[0])}
            for name in names
        }

        try:
            data = self.llm.generate_json(prompt)
            roles = data.get("roles", {})

            if not isinstance(roles, dict):
                return fallback

            cleaned: Dict[str, Dict] = {}
            for name in names:
                info = roles.get(name)
                if not isinstance(info, dict):
                    return fallback
                cleaned[name] = {
                    "role": str(info.get("role", "participant")).strip() or "participant",
                    "is_primary": bool(info.get("is_primary", False)),
                }

            # Enforce exactly one primary.
            primaries = [n for n, v in cleaned.items() if v["is_primary"]]
            if len(primaries) != 1:
                for n in cleaned:
                    cleaned[n]["is_primary"] = False
                cleaned[names[0]]["is_primary"] = True

            return cleaned

        except Exception as e:
            print(f"!! Role planning error: {e}")
            return fallback
