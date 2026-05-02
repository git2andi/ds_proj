"""
modules/generator.py
--------------------
MultiUserSimulator — generates one dialogue turn for a single participant.

All prompt text lives in prompts.py; this module handles only the logic of
what context to gather, then delegates rendering to the prompt registry.

Changes:
- dynamic_forbidden_phrases: extracts repeated n-grams from recent history
  and passes them to the prompt so the model avoids emergent repetition.
- prompts.sim_turn now receives dynamic_forbidden_phrases as a separate argument.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import TYPE_CHECKING, Optional

import prompts
from config_loader import cfg
from modules.llm_client import get_llm_client
from modules.persona_builder import Persona

if TYPE_CHECKING:
    from modules.orchestrator import DialogueState


# Numeric response_length (1–5) → (label, hard style rule shown in prompt).
# These are framed as constraints, not suggestions — matching the hard-rule
# block at the top of the sim_turn prompt.
_RESPONSE_STYLE: dict[int, tuple[str, str]] = {
    1: (
        "brief",
        "Maximum 1 sentence. Short reactions only — e.g. 'Yeah, fair point.' or "
        "'Not convinced by that.' Never elaborate unless directly asked a question.",
    ),
    2: (
        "concise",
        "Maximum 1 sentence. Make one clean point without elaborating. "
        "No reasoning chains, no follow-up thoughts.",
    ),
    3: (
        "balanced",
        "Maximum 2 sentences. One point plus a short reason. No padding or repetition.",
    ),
    4: (
        "talkative",
        "Up to 3 sentences. You elaborate: give context, your reasoning, and optionally "
        "a follow-up thought or question.",
    ),
    5: (
        "detailed",
        "Up to 4 sentences. You think out loud with thorough, well-reasoned responses.",
    ),
}


class MultiUserSimulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: Optional[list[str]] = None,
        forced_adaptation: bool = False,
    ) -> str:
        all_names = all_names or []
        prompt = prompts.sim_turn(
            name=self.name,
            role=self.persona.role,
            is_primary=self.persona.is_primary,
            topic=self.topic,
            options_text=self._format_options(),
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            behavior_text=self._behavior_text(),
            focus_text=self._focus_text(),
            style_instruction=self._style_instruction(),
            state_summary=self._state_summary(state),
            recent_points=self._recent_points(history),
            recent_history=self._recent_history(history),
            phase=state.phase,
            contrarian_nudge=self._contrarian_nudge(state),
            question_nudge=self._question_nudge(history, state, all_names),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames),
            dynamic_forbidden_phrases=self._extract_repeated_phrases(history),
            forced_adaptation=forced_adaptation,
        )

        try:
            raw = self._llm.generate(prompt).strip()
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

        if not raw:
            return "[SILENCE]"

        # Strip accidental "Name: " prefix the model sometimes adds.
        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        return raw or "[SILENCE]"

    # ------------------------------------------------------------------
    # Context formatters
    # ------------------------------------------------------------------

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _recent_history(self, history: list[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _recent_points(self, history: list[str], max_count: int = 4) -> str:
        points: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            points.append(f"{speaker.strip()}: {msg.strip()}")
            if len(points) >= max_count:
                break
        if not points:
            return "None."
        points.reverse()
        return "\n".join(f"- {p}" for p in points)

    def _recent_openers(self, history: list[str], n: int = 4) -> str:
        """First words of the last N participant turns — used to vary turn openings."""
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

    def _extract_repeated_phrases(
        self,
        history: list[str],
        ngram_size: int = 4,
        min_count: int = 3,
        window: int = 16,
    ) -> list[str]:
        """
        Extract n-grams that appear >= min_count times in recent participant turns.
        These are injected into the prompt as dynamically-forbidden phrases so the
        model stops echoing emergent repetitions like "potential kid-friendly services".

        Only participant lines within the last `window` lines are scanned.
        Short/stopword-only n-grams are filtered out.
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
                gram = words[i : i + ngram_size]
                # Skip n-grams that are entirely stopwords.
                if all(w in stopwords for w in gram):
                    continue
                all_ngrams.append(" ".join(gram))

        counts = Counter(all_ngrams)
        return [phrase for phrase, count in counts.items() if count >= min_count]

    def _quiet_participants(
        self, history: list[str], all_names: list[str], window: int = 6
    ) -> list[str]:
        """Names of participants who haven't spoken in the last `window` turns."""
        recent_speakers: set[str] = set()
        count = 0
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker = line.split(":", 1)[0].strip()
            if speaker in cfg.EXCLUDED_SPEAKERS:
                continue
            recent_speakers.add(speaker)
            count += 1
            if count >= window:
                break
        return [n for n in all_names if n not in recent_speakers and n != self.name]

    # ------------------------------------------------------------------
    # Behavior text builders
    # ------------------------------------------------------------------

    def _behavior_text(self) -> str:
        """
        Translate numeric trait values into plain-English behavioral cues.
        Raw numbers are intentionally not shown — only natural language instructions.
        """
        p = self.persona
        lines: list[str] = []

        assertiveness = p.get("assertiveness", 3)
        if assertiveness >= 4:
            lines.append("State opinions directly; don't soften disagreements.")
        elif assertiveness <= 2:
            lines.append("Hedge opinions and avoid direct confrontation.")

        friendliness = p.get("friendliness", 3)
        if friendliness <= 2:
            lines.append("Your tone is blunt — you don't go out of your way to be warm.")
        elif friendliness >= 4:
            lines.append("You are warm; acknowledge others before adding your own view.")

        talkativeness = p.get("talkativeness", 3)
        if talkativeness <= 2:
            lines.append("Speak only when you have something specific to add.")
        elif talkativeness >= 4:
            lines.append("You tend to elaborate and think out loud.")

        agreeableness = p.get("agreeableness", 3)
        if agreeableness >= 4:
            lines.append("Look for common ground and validate others' points.")
        elif agreeableness <= 2:
            lines.append("Challenge points that don't convince you.")

        patience = p.get("patience", 3)
        if patience <= 2:
            lines.append("You get frustrated when the discussion goes in circles.")
        elif patience >= 4:
            lines.append("You are happy to let others work through their thoughts.")

        contrarian = p.get("contrarian_pressure", 3)
        if contrarian >= 4:
            lines.append(
                "You naturally question the obvious choice and probe for weaknesses. "
                "When everyone agrees too fast, raise overlooked trade-offs."
            )
        elif contrarian <= 2:
            lines.append("You go along with the emerging group consensus once you see it forming.")

        if p.get("initiative", 3) >= 4:
            lines.append("You direct questions at specific people to move things forward.")

        return " ".join(lines) if lines else "Engage in a balanced, neutral manner."

    def _focus_text(self) -> str:
        """Format focus values with their topic-contextual notes."""
        focus: dict = self.persona.focus
        notes: dict = self.persona.focus_notes or {}
        parts: list[str] = []
        for dim in ["cost", "comfort", "time", "safety", "flexibility_focus"]:
            val = focus.get(dim, 3)
            note = notes.get(dim, "")
            entry = f"{dim}: {val}/5"
            if note:
                entry += f" ({note})"
            parts.append(entry)
        return "; ".join(parts)

    def _style_instruction(self) -> str:
        level = max(1, min(5, int(self.persona.get("response_length", 3))))
        label, desc = _RESPONSE_STYLE[level]
        return f"Speaking style ({label}): {desc}"

    def _state_summary(self, state: "DialogueState") -> str:
        events = ", ".join(state.important_events or []) or "none"
        return (
            f"phase={state.phase}; "
            f"last_addressed={state.last_addressed}; "
            f"pending_question={state.pending_question_target}; "
            f"pending_reply={state.pending_reply_target}; "
            f"repetition_pressure={state.repetition_pressure:.2f}; "
            f"events={events}"
        )

    # ------------------------------------------------------------------
    # Nudge injections
    # ------------------------------------------------------------------

    def _contrarian_nudge(self, state: "DialogueState") -> str:
        leading = state.current_leading_option
        if not leading:
            return ""
        contrarian = self.persona.get("contrarian_pressure", 3)
        assertiveness = self.persona.get("assertiveness", 3)
        if contrarian >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "Your contrarian streak means you should probe its weaknesses or raise "
                "an overlooked concern — even if you end up agreeing, don't echo the consensus."
            )
        if contrarian == 3 and assertiveness >= 4:
            return (
                f"\nNote: Option {leading} is gaining traction. "
                "If it doesn't fully align with your focus or goal, raise a specific concern."
            )
        return ""

    def _question_nudge(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: list[str],
    ) -> str:
        if self.persona.get("initiative", 3) < 4:
            return ""
        if state.repetition_pressure < 0.4:
            return ""
        quiet = self._quiet_participants(history, all_names)
        if not quiet:
            return ""
        target = random.choice(quiet)
        return (
            f"\nThe discussion is going in circles. "
            f"Consider directing a specific question at {target} to bring in a fresh perspective."
        )