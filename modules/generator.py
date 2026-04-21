import random
import re
from typing import Dict, List

from constants import EXCLUDED_SPEAKERS
from modules.llm_client import get_llm_client


class MultiUserSimulator:
    # Maps response_length trait (1-5) to a hard word-count ceiling.
    _WORD_LIMIT = {1: 15, 2: 25, 3: 35, 4: 50, 5: 80}

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
            "contrarian_pressure",
        ]
        return ", ".join(f"{k}={self.persona.get(k, 3)}/5" for k in keys)

    def _traits_as_behavior(self) -> str:
        """
        Translate numeric trait values into concrete behavioral instructions
        so the LLM understands what each score actually means in conversation.
        """
        p = self.persona
        lines = []

        assertiveness = p.get("assertiveness", 3)
        if assertiveness >= 4:
            lines.append("You state opinions directly and don't soften disagreements.")
        elif assertiveness <= 2:
            lines.append("You hedge your opinions and avoid direct confrontation.")

        friendliness = p.get("friendliness", 3)
        if friendliness <= 2:
            lines.append("Your tone is blunt, possibly terse — you don't go out of your way to be warm.")
        elif friendliness >= 4:
            lines.append("You are warm and encouraging; you acknowledge others before adding your own view.")

        talkativeness = p.get("talkativeness", 3)
        if talkativeness <= 2:
            lines.append("You speak only when you have something specific to add — short replies are fine.")
        elif talkativeness >= 4:
            lines.append("You tend to elaborate and think out loud.")

        agreeableness = p.get("agreeableness", 3)
        if agreeableness >= 4:
            lines.append("You actively look for common ground and validate others' points.")
        elif agreeableness <= 2:
            lines.append("You don't rush to agree — you'll challenge a point if it doesn't convince you.")

        patience = p.get("patience", 3)
        if patience <= 2:
            lines.append("You get frustrated when the discussion goes in circles and want to move on.")
        elif patience >= 4:
            lines.append("You are happy to let others work through their thoughts without rushing them.")

        contrarian = p.get("contrarian_pressure", 3)
        if contrarian >= 4:
            lines.append(
                "You have a natural tendency to question the obvious choice and play devil's advocate. "
                "When everyone agrees too quickly, you probe for weaknesses."
            )
        elif contrarian <= 2:
            lines.append("You tend to go along with the emerging group consensus once you see it forming.")

        initiative = p.get("initiative", 3)
        if initiative >= 4:
            lines.append("You are comfortable directing questions at specific people to move things forward.")

        return " ".join(lines) if lines else "You engage in a balanced, neutral manner."

    def _word_limit(self) -> int:
        return self._WORD_LIMIT.get(int(self.persona.get("response_length", 3)), 40)

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

    def _recent_openers(self, history: List[str], n: int = 4) -> str:
        """Extract first words of recent turns to forbid the model repeating them."""
        openers = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in EXCLUDED_SPEAKERS:
                continue
            first_word = msg.strip().split()[0].rstrip(",.!?") if msg.strip() else ""
            if first_word and first_word not in openers:
                openers.append(first_word)
            if len(openers) >= n:
                break
        return ", ".join(openers) if openers else ""

    def _who_hasnt_spoken_recently(self, history: List[str], all_names: List[str], n: int = 6) -> List[str]:
        """Return names of participants who have not spoken in the last n participant lines."""
        recent_speakers = set()
        count = 0
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker in EXCLUDED_SPEAKERS:
                continue
            recent_speakers.add(speaker)
            count += 1
            if count >= n:
                break
        return [n for n in all_names if n not in recent_speakers and n != self.name]

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

    def _contrarian_nudge(self, state) -> str:
        """
        If the group is converging on an option and this persona has high
        contrarian_pressure or hasn't gotten traction for their focus, return
        a prompt injection that nudges them to push back or probe.
        """
        leading = getattr(state, "current_leading_option", None)
        if not leading:
            return ""

        contrarian = self.persona.get("contrarian_pressure", 3)
        assertiveness = self.persona.get("assertiveness", 3)

        # High contrarian trait: actively probe the leading option.
        if contrarian >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "Your contrarian streak means you should probe its weaknesses or "
                "raise a concern — even if you end up agreeing, don't just echo the consensus."
            )

        # Medium contrarian + high assertiveness: plant a seed of doubt.
        if contrarian == 3 and assertiveness >= 4:
            return (
                f"\nNote: Option {leading} seems to be gaining traction. "
                "If it doesn't fully align with your focus or goal, you may want to "
                "raise a specific concern or ask a probing question about it."
            )

        return ""

    def _question_nudge(self, state, all_names: List[str]) -> str:
        """
        For high-initiative personas when discussion is stalling, suggest
        directing a question at a specific quiet participant.
        """
        if self.persona.get("initiative", 3) < 4:
            return ""
        rep_pressure = getattr(state, "repetition_pressure", 0.0)
        if rep_pressure < 0.4:
            return ""
        quiet = self._who_hasnt_spoken_recently([], all_names)  # placeholder — passed from generate_turn
        if not quiet:
            return ""
        target = random.choice(quiet)
        return (
            f"\nSince the discussion is going in circles, consider directing a specific "
            f"question at {target} to bring in a fresh perspective."
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def _generate_initial_goal(self) -> str:
        prompt = (
            f"Scenario: {self.setting}\n"
            f"Participant name: {self.name}\n"
            f"Role in this scenario: {self.role}\n"
            f"Is primary participant: {self.is_primary}\n"
            f"Numeric traits (1-5): {self._numeric_traits_summary()}\n"
            f"Focus values (1-5): {self._format_focus()}\n\n"
            "Task:\n"
            "Write exactly one short internal goal sentence for this participant.\n"
            "\n"
            "Rules:\n"
            "- The goal must be specific to the scenario domain. "
            "If the scenario is about booking a restaurant, the goal is about the meal experience, "
            "cuisine, group dynamics, or atmosphere — not about generic concepts like schedules or budgets "
            "unless the scenario explicitly involves those.\n"
            "- Do NOT lift trait names directly into the goal text. "
            "Traits shape personality and behaviour, not the subject matter of the goal. "
            "A low-safety focus does not mean the person cares about safety; "
            "a high-cost focus means they are price-conscious in the context of this scenario.\n"
            "- Do NOT use filler phrases like 'efficiently', 'seamlessly', or 'in a timely manner'.\n"
            "- Write in third person. One sentence only. No bullet points, markdown, or quotation marks."
        )
        try:
            raw = self._call_llm(prompt).strip()
            return raw if raw else "Support a practical outcome that fits your priorities."
        except Exception as e:
            print(f"!! Goal generation error for {self.name}: {e}")
            return "Support a practical outcome that fits your priorities."

    def generate_turn(self, history: List[str], state, all_names: List[str] = None) -> str:
        all_names = all_names or []
        word_limit = self._word_limit()

        # Build dynamic forbidden openers from recent history.
        forbidden_openers = self._recent_openers(history, n=4)

        # Build optional nudges.
        contrarian_nudge = self._contrarian_nudge(state)

        # Question nudge — pass the actual history here.
        quiet_names = self._who_hasnt_spoken_recently(history, all_names)
        question_nudge = ""
        if self.persona.get("initiative", 3) >= 4 and getattr(state, "repetition_pressure", 0.0) >= 0.4 and quiet_names:
            target = random.choice(quiet_names)
            question_nudge = (
                f"\nSince the discussion is going in circles, consider directing a specific "
                f"question at {target} to bring in a fresh perspective."
            )

        forbidden_note = (
            f"\nDo NOT start your reply with any of these words (recently overused): {forbidden_openers}."
            if forbidden_openers else ""
        )

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
- Behavioral style: {self._traits_as_behavior()}
- Focus: {self._format_focus()}

Dialogue state:
{self._format_state_summary(state)}

Recent participant points:
{self._format_recent_points(history)}

Recent history:
{self._format_recent_history(history)}

Instructions:
- Reply with only your next utterance — no speaker label, no stage directions.
- Stay in character with your role, goal, and behavioral style at all times.
- React to what was just said, not just your own preferences.
- If you were directly addressed or questioned, respond to that first.
- If the discussion is going in circles, help move it forward.
- ONLY use information already present in the options or history above.
- Do NOT ask for prices, confirmation numbers, external data, or any detail not listed in the options.
- If options are listed, choose only among those listed options.
- Aim for practical progress toward a shared decision.
- Keep your reply under {word_limit} words. Short replies are natural and welcome.{contrarian_nudge}{question_nudge}{forbidden_note}

Phase-specific behavior:
- opening: briefly introduce your main concern.
- preference_expression: state which option you lean toward and why.
- negotiation: compare trade-offs, react to others, adjust if reasonable.
- narrowing: help converge on one option; a backup is optional.
- confirmation: clearly confirm or reject the emerging agreement with a plain yes/no.
- closure: say a short, natural goodbye or sign-off that fits your personality and the scenario (e.g. 'Great, looking forward to the trip!' or 'Good call, see you there.'). One sentence only.

Style:
- Sound like a real person in a conversation or chat, not an essay writer.
- Mix short reactive replies ("Yeah, makes sense" / "Hmm, I'm not so sure") with longer points.
- Vary your opening words — do not repeat the same phrase every turn.{forbidden_note}
- Do not say goodbye unless the phase is closure.
- Do not summarize the entire discussion — just make your next point.
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
