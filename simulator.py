"""
simulator.py
------------
Simulator — wraps a Persona and generates one dialogue turn via LLM.

Responsibilities:
- Format the turn prompt (delegates all text to prompts.py)
- Strip common model artefacts (name prefix, silence)
- Track forbidden openers from recent history
- Detect semantically repeated phrases and add them to the forbidden list
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona

if TYPE_CHECKING:
    from orchestrator import DialogueState


class Simulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options          # empty list in open-ended mode
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: list[str] | None = None,
        forced_adaptation: bool = False,
    ) -> str:
        all_names = all_names or []
        is_open = state.mode == "open"

        if is_open:
            raw = self._generate_open(history, state, forced_adaptation)
        else:
            raw = self._generate_decision(history, state, forced_adaptation)

        if not raw:
            return "[SILENCE]"

        # Strip accidental "Name: " prefix the model sometimes adds.
        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        return raw or "[SILENCE]"

    # ------------------------------------------------------------------
    # Decision mode (options exist)
    # ------------------------------------------------------------------

    def _generate_decision(
        self, history: list[str], state: "DialogueState", forced_adaptation: bool
    ) -> str:
        phase_instructions = {
            "opening": "Say a quick hello if you like, then share your first instinct or main priority. Keep it natural — you are just joining the conversation.",
            "preference_expression": "State which option you lean toward and the one specific reason that matters most to you.",
            "negotiation": "Compare trade-offs, react directly to what was just said, and adjust your position only if genuinely persuaded.",
            "narrowing": (
                "Commit to a preferred option and state it clearly (e.g. 'I prefer Option A'). "
                "A backup is fine if genuinely unsure. "
                "Once you have stated a preference, KEEP IT unless someone raises a specific new reason that genuinely changes your view — "
                "do not switch just because someone repeats their position more forcefully."
            ),
            "confirmation": (
                "The moderator is asking you to confirm or reject the emerging choice. "
                "Answer with a clear yes or no. "
                "If you already stated a preference in the narrowing phase, your answer should be consistent with it — "
                "only say no if you have a genuine objection you have not yet raised."
            ),
            "closure": "One short, natural sign-off that fits your personality. One sentence only.",
        }

        prompt = prompts.sim_turn(
            name=self.name,
            role=self.persona.role,
            is_primary=self.persona.is_primary,
            topic=self.topic,
            options_text=self._format_options(),
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            personality_summary=self.persona.personality_summary(),
            style_rule=self.persona.style_rule(),
            phase=state.phase,
            phase_instruction=phase_instructions.get(state.phase, "React naturally to the conversation."),
            state_summary=self._state_summary(state),
            recent_history=self._recent_history(history),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames),
            contrarian_nudge=self._contrarian_nudge(state),
            forced_adaptation=forced_adaptation,
        )

        try:
            return self._llm.generate(prompt).strip()
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    # ------------------------------------------------------------------
    # Open-ended mode (no options)
    # ------------------------------------------------------------------

    def _generate_open(
        self, history: list[str], state: "DialogueState", forced_adaptation: bool
    ) -> str:
        prompt = prompts.sim_turn_open(
            name=self.name,
            role=self.persona.role,
            is_primary=self.persona.is_primary,
            topic=self.topic,
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            personality_summary=self.persona.personality_summary(),
            style_rule=self.persona.style_rule(),
            phase=state.phase,
            state_summary=self._state_summary(state),
            recent_history=self._recent_history(history),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames),
            dynamic_forbidden_phrases=self._repeated_phrases(history),
            forced_adaptation=forced_adaptation,
        )

        try:
            return self._llm.generate(prompt).strip()
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    # ------------------------------------------------------------------
    # Context formatters
    # ------------------------------------------------------------------

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _recent_history(self, history: list[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _recent_openers(self, history: list[str], n: int = 4) -> str:
        """First words of the last N participant turns — prevents repetitive openings."""
        openers: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            first_word = msg.strip().split()[0].rstrip(",.!?") if msg.strip() else ""
            if first_word and first_word not in openers:
                openers.append(first_word)
            if len(openers) >= n:
                break
        return ", ".join(openers) if openers else ""

    def _repeated_phrases(
        self,
        history: list[str],
        ngram_size: int = 3,
        min_count: int = 3,
        window: int = 16,
    ) -> list[str]:
        """
        Extract n-grams that appear >= min_count times in recent participant turns.
        These are injected as dynamically-forbidden phrases to stop semantic loops
        (e.g. "safe space", "honest dialogue" repeating across 5 turns).
        Only scans the last `window` participant lines.
        """
        stopwords = {
            "the", "a", "an", "and", "or", "but", "so", "to", "of", "in",
            "is", "it", "i", "we", "you", "that", "this", "for", "with",
            "at", "on", "be", "as", "by", "if", "do", "not", "no", "yes",
        }

        texts: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            texts.append(msg.strip().lower())
            if len(texts) >= window:
                break

        all_ngrams: list[str] = []
        for text in texts:
            words = re.sub(r"[^\w\s]", "", text).split()
            for i in range(len(words) - ngram_size + 1):
                gram = words[i: i + ngram_size]
                if all(w in stopwords for w in gram):
                    continue
                all_ngrams.append(" ".join(gram))

        counts = Counter(all_ngrams)
        return [phrase for phrase, count in counts.items() if count >= min_count]

    def _state_summary(self, state: "DialogueState") -> str:
        return (
            f"phase={state.phase}; "
            f"leading_option={state.current_leading_option}; "
            f"repetition_pressure={state.repetition_pressure:.2f}"
        )

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------

    def _contrarian_nudge(self, state: "DialogueState") -> str:
        leading = state.current_leading_option
        if not leading:
            return ""
        if self.persona.contrarian >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "Your contrarian streak means you should probe its weaknesses or raise "
                "an overlooked concern — even if you end up agreeing, do not echo the consensus."
            )
        return ""