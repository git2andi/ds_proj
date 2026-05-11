"""
simulator.py
------------
Simulator — wraps a Persona and generates one dialogue turn via LLM.

Responsibilities:
- Format the turn prompt (delegates all text to prompts.py)
- Strip common model artefacts (name prefix, silence)
- Track forbidden openers from recent history
- Detect repeated phrases and add them to the forbidden list
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import TYPE_CHECKING

import prompts
from config_loader import cfg
from llm_client import get_llm_client
from persona import Persona
from utils import extract_preference_vote

if TYPE_CHECKING:
    from orchestrator import DialogueState


class Simulator:

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
        self._llm = get_llm_client()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    _GOODBYES = ["Later!", "See ya!", "Bye!", "Bye everyone!", "See you!", "Cheers!", "Take care!"]

    def generate_turn(
        self,
        history: list[str],
        state: "DialogueState",
        all_names: list[str] | None = None,
        forced_adaptation: bool = False,
    ) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out). Tokens are 0 when the LLM call fails."""
        all_names = all_names or []

        if state.phase == "closure":
            return random.choice(self._GOODBYES), 0, 0

        raw = self._generate_decision(history, state, forced_adaptation)

        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

        # Strip accidental "Name: " prefix the model sometimes adds.
        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        return raw or "[SILENCE]", tok_in, tok_out

    # ------------------------------------------------------------------
    # Turn generation
    # ------------------------------------------------------------------

    def _has_voted(self, history: list[str]) -> bool:
        """True if this sim has explicitly stated an option letter in any of their turns."""
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            if re.search(r"\boption\s+([a-d])(?![\w'\-])", msg.lower()):
                return True
        return False

    def _generate_decision(
        self, history: list[str], state: "DialogueState", forced_adaptation: bool
    ) -> str:
        narrowing_base = (
            "Commit to a preferred option and state it clearly (e.g. 'I prefer Option A'). "
            "A backup is fine if genuinely unsure. "
            "IMPORTANT — position discipline: once you have stated a preference, you may change it AT MOST ONCE "
            "during this phase, and only if someone raises a specific new argument you had not considered. "
            "Do not switch because someone repeats their view more insistently or because you want to agree. "
            "If you have already changed your position once, hold your current preference and defend it."
        )
        if state.phase == "narrowing" and not self._has_voted(history):
            narrowing_instruction = (
                "URGENT: You have not yet stated a preferred option. "
                "Before anything else this turn, name your preferred option explicitly "
                "(e.g. 'I prefer Option A' or 'My choice is Option C'). "
                "Do not ask questions or comment on others until you have done this. "
                + narrowing_base
            )
        else:
            narrowing_instruction = narrowing_base

        phase_instructions = {
            "greeting": (
                "Say a quick, casual hello — just 'hey', your name, or a simple greeting. "
                "One short line only. Do NOT say your role, job, or any description of yourself. "
                "Do NOT discuss the topic or express any opinions yet."
            ),
            "opening": (
                "The group just said hello. Now give your first honest take on the topic. "
                "One or two casual sentences — what's your gut reaction? "
                "Feel free to briefly mention a relevant personal experience if it fits naturally."
            ),
            "negotiation": (
                "Compare trade-offs, react directly to what was just said, and adjust your position "
                "only if genuinely persuaded. If someone proposes a compromise, engage with the specific "
                "part that works or doesn't — don't just acknowledge and reset to your original stance. "
                "Only reference attributes explicitly listed in the options — do not invent details. "
                "Do not argue both for and against the same option — pick a position and defend it."
            ),
            "narrowing": narrowing_instruction,
            "confirmation": (
                "The moderator is asking for a final confirmation. "
                "Reply with an explicit 'yes' or 'no' — nothing else counts as a confirmation. "
                "A question, a hedge, or silence is treated as a no. "
                "If the option being confirmed matches your stated narrowing preference, say yes. "
                "Only say no if you have a specific objection you have not yet raised."
            ),
            "closure": (
                "The discussion is OVER. Write ONE short goodbye only."
                "Nothing about the topic. No opinions. No questions."
            ),
        }

        phase_instr = phase_instructions.get(state.phase, "React naturally to the conversation.")
        if (state.repetition_pressure >= 0.55 or state.post_narrowing_rounds >= 2) \
                and state.phase not in {"greeting", "opening", "closure", "confirmation"}:
            phase_instr += " One or two sentences only — you've made your case, react don't re-explain."

        is_closure = state.phase == "closure"
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
            phase_instruction=phase_instr,
            state_summary=self._state_summary(state),
            recent_history=self._recent_history(history, max_lines=4 if is_closure else 12),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames) + self._repeated_phrases(history),
            last_speaker_line="" if is_closure else self._last_participant_line(history),
            position_discipline="" if is_closure else self._position_discipline(history, state),
            contrarian_nudge=self._contrarian_nudge(state),
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
        Injected as dynamically-forbidden phrases to stop semantic loops.
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
                "Probe a weakness or raise an overlooked concern — but if you've already made "
                "that point in the recent chat, pick a DIFFERENT angle rather than rephrasing the same objection."
            )
        return ""

    def _last_participant_line(self, history: list[str]) -> str:
        """Most recent line from someone other than self, for explicit anchoring."""
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() not in cfg.EXCLUDED_SPEAKERS and speaker.strip() != self.name:
                return f"{speaker.strip()}: {msg.strip()}"
        return ""

    def _position_discipline(self, history: list[str], state: "DialogueState") -> str:
        """Persistent, context-aware vote-discipline block injected once a vote is on record."""
        if state.phase not in ("negotiation", "narrowing", "confirmation"):
            return ""

        votes_seen: list[str] = []
        for line in history:
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() != self.name:
                continue
            v = extract_preference_vote(msg)
            if v and (not votes_seen or v != votes_seen[-1]):
                votes_seen.append(v)

        if not votes_seen:
            return ""

        current = votes_seen[-1]
        changes = len(votes_seen) - 1
        coherence = f" Your arguments should support Option {current}, not undermine it."

        if state.phase == "negotiation":
            return f"\nYou've stated a preference for Option {current}.{coherence}"

        if changes == 0:
            return (
                f"\nYou've stated a preference for Option {current}.{coherence} "
                "Only change this if someone raises a genuinely new argument you hadn't yet considered."
            )
        if changes == 1:
            return (
                f"\nYou already switched once and now prefer Option {current}.{coherence} "
                "Hold this — defend with a new reason rather than switching again."
            )
        return (
            f"\nYou've changed preference {changes} times. "
            f"Commit to Option {current} — no more switching.{coherence}"
        )
