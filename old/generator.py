from typing import Dict, List

from constants import EXCLUDED_SPEAKERS
from modules.llm_client import get_llm_client


class MultiUserSimulator:
    def __init__(self, persona: Dict, setting: str, options: List[str], persona_manager=None):
        self.persona = persona
        self.name = persona["name"]
        self.setting = setting
        self.options = options
        self.persona_manager = persona_manager

        self.role = persona.get("role", "participant")
        self.is_primary = bool(persona.get("is_primary", False))

        self.llm = get_llm_client()

        if not self.persona.get("goal"):
            self.goal = self._generate_initial_goal()
            self.persona["goal"] = self.goal
            if self.persona_manager:
                self.persona_manager.save_persona(self.persona)
        else:
            self.goal = self.persona["goal"]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_focus(self) -> str:
        focus = self.persona.get("focus", {})
        if not isinstance(focus, dict) or not focus:
            return "cost:3, comfort:3, time:3, safety:3, flexibility_focus:3"
        ordered = ["cost", "comfort", "time", "safety", "flexibility_focus"]
        return ", ".join(f"{k}:{focus.get(k, 3)}/5" for k in ordered)

    def _numeric_traits_summary(self) -> str:
        keys = [
            "friendliness", "assertiveness", "talkativeness", "initiative",
            "agreeableness", "flexibility", "patience", "response_length",
        ]
        return ", ".join(f"{k}={self.persona.get(k, 3)}/5" for k in keys)

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _format_recent_history(self, history: List[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _format_recent_points(self, history: List[str], max_count: int = 4) -> str:
        """Last N participant lines (excluding Moderator) as bullet points."""
        points = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in EXCLUDED_SPEAKERS:
                continue
            points.append(f"{speaker.strip()}: {msg.strip()}")
            if len(points) >= max_count:
                break
        if not points:
            return "None."
        points.reverse()
        return "\n".join(f"- {p}" for p in points)

    def _format_state_summary(self, state) -> str:
        events_text = ", ".join(getattr(state, "important_events", []) or []) or "none"
        return (
            f"phase={getattr(state, 'phase', 'negotiation')}; "
            f"last_addressed={getattr(state, 'last_addressed', None)}; "
            f"pending_question_target={getattr(state, 'pending_question_target', None)}; "
            f"pending_reply_target={getattr(state, 'pending_reply_target', None)}; "
            f"repetition_pressure={getattr(state, 'repetition_pressure', 0.0):.2f}; "
            f"important_events={events_text}"
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def _generate_initial_goal(self) -> str:
        prompt = (
            f"Scenario: {self.setting}\n"
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Is primary participant: {self.is_primary}\n"
            f"Numeric traits: {self._numeric_traits_summary()}\n"
            f"Focus values: {self._format_focus()}\n\n"
            "Task:\n"
            "Generate exactly one short internal goal sentence for this participant.\n"
            "The goal must fit the scenario and reflect the participant's role and traits.\n"
            "Do not invent a backstory. No bullet points, numbering, markdown, or quotation marks.\n"
            "Return exactly one sentence only."
        )
        try:
            raw = self._call_llm(prompt).strip()
            return raw if raw else "Support a practical outcome that fits your priorities."
        except Exception as e:
            print(f"!! Goal generation error for {self.name}: {e}")
            return "Support a practical outcome that fits your priorities."

    def generate_turn(self, history: List[str], state) -> str:
        prompt = f"""
Identity:
- Name: {self.name}
- Role: {self.role}
- Primary participant: {self.is_primary}

Scenario:
{self.setting}

Available options (these are the ONLY facts you have — do not invent any other details):
{self._format_options()}

Internal profile:
- Goal: {self.goal}
- Traits: {self._numeric_traits_summary()}
- Focus: {self._format_focus()}

Dialogue state:
{self._format_state_summary(state)}

Recent participant points:
{self._format_recent_points(history)}

Recent history:
{self._format_recent_history(history)}

Instructions:
- Reply with only your next utterance — no speaker label, no stage directions.
- Stay in character with your role, goal, and traits at all times.
- React to what was just said, not just your own preferences.
- If you were directly addressed or questioned, respond to that first.
- If the discussion is going in circles, help move it forward.
- ONLY use information already present in the options or history above.
- Do NOT ask for prices, confirmation numbers, external data, or any detail not listed in the options.
- If options are listed, choose only among those listed options.
- Aim for practical progress toward a shared decision.

Phase-specific behavior:
- opening: briefly introduce your main concern.
- preference_expression: state which option you lean toward and why.
- negotiation: compare trade-offs, react to others, adjust if reasonable.
- narrowing: help converge on one option; a backup is optional.
- confirmation: clearly confirm or reject the emerging agreement with a plain yes/no.
- closure: say a short, natural goodbye or sign-off that fits your personality and the scenario (e.g. 'Great, looking forward to the trip!' or 'Good call, see you there.'). One sentence only.

Style:
- Sound like a real person in a conversation, not an essay writer.
- Be concise. Short replies are fine when appropriate.
- Vary your opening words — do not repeat the same phrase every turn.
- Do not say goodbye unless the phase is closure.
"""
        try:
            raw = self._call_llm(prompt).strip()
            if not raw:
                return "[SILENCE]"
            # Strip accidental "Name: " prefix the model sometimes adds.
            if raw.lower().startswith(f"{self.name.lower()}:"):
                raw = raw.split(":", 1)[1].strip()
            return raw or "[SILENCE]"
        except Exception as e:
            print(f"!! Turn generation error for {self.name}: {e}")
            return "[SILENCE]"
