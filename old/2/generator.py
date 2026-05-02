import random
import re
from typing import Dict, List, Tuple

from constants import EXCLUDED_SPEAKERS
from modules.llm_client import get_llm_client


class MultiUserSimulator:

    # Numeric response_length (1-5) maps to a (label, behavioral description) pair.
    # The numeric value is kept in the persona JSON for evaluation;
    # the label+description is what the LLM actually sees in the prompt.
    _RESPONSE_STYLE: Dict[int, Tuple[str, str]] = {
        1: (
            "brief",
            "You make short replies. Reactions like 'Yeah, fair point.' "
            "or 'Not sure about that.' are perfect. Never elaborate unless directly asked.",
        ),
        2: (
            "concise",
            "You make one short point per turn, cleanly, without elaborating much. "
            "A sentence at most",
        ),
        3: (
            "balanced",
            "You give a clear point with a short reason — up to two short sentences. "
            "You don't pad or over-explain.",
        ),
        4: (
            "talkative",
            "You tend to elaborate: you give context, your reasoning, and sometimes "
            "a follow-up thought or question.",
        ),
        5: (
            "detailed",
            "You think out loud and give thorough, well-reasoned responses. ",
        ),
    }

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
        """Used for goal generation only — keeps raw numbers for the LLM context."""
        keys = [
            "friendliness", "assertiveness", "talkativeness", "initiative",
            "agreeableness", "flexibility", "patience", "response_length",
            "contrarian_pressure",
        ]
        return ", ".join(f"{k}={self.persona.get(k, 3)}/5" for k in keys)

    def _traits_as_behavior(self) -> str:
        """
        Translate numeric trait values into concrete behavioral instructions
        so the LLM understands what each score means in conversation.
        Raw numbers are intentionally NOT shown here — only plain-English cues.
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
            lines.append(
                "Your tone is blunt, possibly terse — you don't go out of your way to be warm."
            )
        elif friendliness >= 4:
            lines.append(
                "You are warm and encouraging; you acknowledge others before adding your own view."
            )

        talkativeness = p.get("talkativeness", 3)
        if talkativeness <= 2:
            lines.append("You speak only when you have something specific to add.")
        elif talkativeness >= 4:
            lines.append("You tend to elaborate and think out loud.")

        agreeableness = p.get("agreeableness", 3)
        if agreeableness >= 4:
            lines.append("You actively look for common ground and validate others' points.")
        elif agreeableness <= 2:
            lines.append(
                "You don't rush to agree — you'll challenge a point if it doesn't convince you."
            )

        patience = p.get("patience", 3)
        if patience <= 2:
            lines.append(
                "You get frustrated when the discussion goes in circles and want to move on."
            )
        elif patience >= 4:
            lines.append(
                "You are happy to let others work through their thoughts without rushing them."
            )

        contrarian = p.get("contrarian_pressure", 3)
        if contrarian >= 4:
            lines.append(
                "You have a natural tendency to question the obvious choice and play devil's advocate. "
                "When everyone agrees too quickly, you probe for weaknesses or raise overlooked trade-offs."
            )
        elif contrarian <= 2:
            lines.append(
                "You tend to go along with the emerging group consensus once you see it forming."
            )

        initiative = p.get("initiative", 3)
        if initiative >= 4:
            lines.append(
                "You are comfortable directing questions at specific people to move things forward."
            )

        return " ".join(lines) if lines else "You engage in a balanced, neutral manner."

    def _response_style_instruction(self) -> str:
        """Return the semantic speaking-style label and description for this persona."""
        level = max(1, min(5, int(self.persona.get("response_length", 3))))
        label, desc = self._RESPONSE_STYLE[level]
        return f"Speaking style ({label}): {desc}"

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _format_recent_history(self, history: List[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _format_recent_points(self, history: List[str], max_count: int = 4) -> str:
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
        """First words of the last N turns — used to forbid repetitive openers."""
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

    def _who_hasnt_spoken_recently(
        self, history: List[str], all_names: List[str], n: int = 6
    ) -> List[str]:
        """Names of participants who have not appeared in the last n participant lines."""
        recent_speakers: set = set()
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
        return [name for name in all_names if name not in recent_speakers and name != self.name]

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
        If the group is converging and this persona has a high contrarian_pressure,
        inject an instruction to push back or probe rather than echo the consensus.
        """
        leading = getattr(state, "current_leading_option", None)
        if not leading:
            return ""

        contrarian = self.persona.get("contrarian_pressure", 3)
        assertiveness = self.persona.get("assertiveness", 3)

        if contrarian >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "Your contrarian streak means you should probe its weaknesses or raise "
                "an overlooked concern — even if you end up agreeing, don't just echo the consensus."
            )
        if contrarian == 3 and assertiveness >= 4:
            return (
                f"\nNote: Option {leading} is gaining traction. "
                "If it doesn't fully align with your focus or goal, raise a specific concern "
                "or ask a probing question rather than simply agreeing."
            )
        return ""

    def _question_nudge(self, history: List[str], state, all_names: List[str]) -> str:
        """
        For high-initiative personas when discussion is stalling, suggest
        directing a question at a specific quiet participant.
        """
        if self.persona.get("initiative", 3) < 4:
            return ""
        if getattr(state, "repetition_pressure", 0.0) < 0.4:
            return ""
        quiet = self._who_hasnt_spoken_recently(history, all_names)
        if not quiet:
            return ""
        target = random.choice(quiet)
        return (
            f"\nThe discussion is going in circles. Consider directing a specific question "
            f"at {target} to bring in a fresh perspective."
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
            "cuisine, group dynamics, or atmosphere — not about generic concepts like schedules "
            "or budgets unless the scenario explicitly involves those.\n"
            "- Do NOT lift trait names directly into the goal text. "
            "Traits shape personality and behaviour, not the subject matter of the goal.\n"
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

        # Build dynamic injections.
        forbidden_openers = self._recent_openers(history, n=4)
        forbidden_note = (
            f"\nDo NOT start your reply with any of these recently overused words: {forbidden_openers}."
            if forbidden_openers else ""
        )
        contrarian_nudge = self._contrarian_nudge(state)
        question_nudge = self._question_nudge(history, state, all_names)
        style_instruction = self._response_style_instruction()

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
- {style_instruction}

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
- ONLY use information already present in the options or history above.
- Do NOT ask for prices, confirmation numbers, external data, or any detail not listed in the options.
- Aim for practical progress toward a shared decision.
- Do NOT summarise what others said before making your own point — just make your point.
- Do NOT use phrases like "As X mentioned..." or "Building on what X said..." as a crutch.{contrarian_nudge}{question_nudge}{forbidden_note}

Phase-specific behavior:
- opening: briefly introduce your main concern or priority.
- preference_expression: state which option you lean toward and why.
- negotiation: compare trade-offs, react to others, adjust if reasonable.
- narrowing: help converge on one option; a backup is optional.
- confirmation: clearly confirm or reject the emerging agreement — a plain yes/no is fine.
- closure: one short, natural sign-off that fits your personality (e.g. 'Great, see you there!' or 'Works for me.'). One sentence only.

Style:
- Sound like a real person in a group chat, not an essay writer.
- Mix short reactive replies ("Yeah, makes sense." / "Hmm, not so sure about that.") with longer points when you have something real to add.
- Vary your opening words every turn.{forbidden_note}
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
