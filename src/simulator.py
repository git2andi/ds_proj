"""
simulator.py
------------
Simulator — wraps a Persona and generates one dialogue turn via LLM.

Responsibilities:
- Format the turn prompt (delegates all text to prompts.py)
- Strip common model artefacts (name prefix, silence)
- Build the beliefs block injected into every turn
- Provide position discipline grounded in the stable belief state
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

if TYPE_CHECKING:
    from orchestrator import DialogueState


class Simulator:

    _GOODBYES = ["Later!", "See ya!", "Bye!", "Bye everyone!", "See you!", "Cheers!", "Take care!"]

    def __init__(self, persona: Persona, topic: str, options: list[str]) -> None:
        self.persona = persona
        self.name = persona.name
        self.topic = topic
        self.options = options
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
    ) -> tuple[str, int, int]:
        """Returns (text, tokens_in, tokens_out). Tokens are 0 when skipped."""
        all_names = all_names or []

        if state.phase == "closure":
            return random.choice(self._GOODBYES), 0, 0

        raw = self._generate_decision(history, state, forced_adaptation)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

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
            "Do not switch because someone repeats their view more insistently or because you want to agree."
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
                "The group just said hello. Give your first honest take on the topic. "
                "One or two casual sentences — what's your gut reaction? "
                "Feel free to briefly mention a relevant personal experience if it fits naturally."
            ),
            "negotiation": (
                "Compare trade-offs, react directly to what was just said, and adjust your position "
                "only if genuinely persuaded. "
                "Only reference attributes explicitly listed in the options — do not invent details. "
                "Do not argue both for and against the same option — pick a position and defend it."
            ),
            "narrowing": narrowing_instruction,
            "emergence": (
                "The main arguments have been made. This is the moment to soften — not dig in. "
                "If the option gaining traction is in your acceptable range, reduce your resistance: "
                "'maybe there's something to that', 'I'm not fully sold but I can see the appeal', "
                "'if [concern] is addressed, I could live with it'. "
                "Don't flip your vote yet — just let your resistance ease if it's genuinely easing. "
                "Only hold firm if you have a brand-new specific objection not yet raised."
            ),
            "confirmation": (
                "The moderator is asking for a final confirmation. "
                "Reply with an explicit 'yes' or 'no' — nothing else counts as a confirmation. "
                "A question, a hedge, or silence is treated as a no. "
                "If the option being confirmed matches your preference, say yes. "
                "Only say no if you have a specific objection you have not yet raised."
            ),
            "closure": (
                "The discussion is OVER. Write ONE short goodbye only. "
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
            topic=self.topic,
            options_text=self._format_options(),
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            personality_summary=self.persona.personality_summary(),
            style_rule=self.persona.style_rule(),
            phase=state.phase,
            phase_instruction=phase_instr,
            recent_history=self._recent_history(history, max_lines=4 if is_closure else 12),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames) + self._repeated_phrases(history),
            beliefs_block="" if is_closure else self._beliefs_block(),
            last_speaker_line="" if is_closure else self._last_participant_line(history),
            position_discipline="" if is_closure else self._position_discipline(state),
            contrarian_nudge=self._contrarian_nudge(state),
            forced_adaptation=forced_adaptation,
        )

        try:
            return self._llm.generate(prompt).strip()
        except Exception as exc:
            print(f"!! Turn generation error for {self.name}: {exc}")
            return "[SILENCE]"

    # ------------------------------------------------------------------
    # Beliefs block
    # ------------------------------------------------------------------

    def _beliefs_block(self) -> str:
        """Compact belief-state anchor injected into every non-closure turn."""
        b = self.persona.beliefs
        if not b:
            return ""

        parts = ["Your internal lean (private anchor — don't recite this, stay coherent):"]
        parts.append(f"  Prefer Option {b.preferred} | Core concern: {b.key_concern}")

        other_acceptable = [x for x in b.acceptable if x != b.preferred]
        if other_acceptable:
            opts_str = ", ".join(f"Option {x}" for x in other_acceptable)
            parts.append(f"  Would compromise on: {opts_str} | Condition: {b.concession}")

        if b.rejected:
            opts_str = ", ".join(f"Option {x}" for x in b.rejected)
            parts.append(f"  Opposed to: {opts_str}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Position discipline
    # ------------------------------------------------------------------

    def _position_discipline(self, state: "DialogueState") -> str:
        """
        Inject a coherence anchor derived from the stable belief state.
        In negotiation: soft reminder of their lean.
        In narrowing/confirmation: escalates with flip count.
        In emergence: facilitates softening toward the candidate option.
        """
        if state.phase not in ("negotiation", "narrowing", "emergence", "confirmation"):
            return ""

        beliefs = self.persona.beliefs
        if not beliefs:
            return ""

        preferred = beliefs.preferred
        flips = state.vote_changes.get(self.name, 0)
        current_text_vote = state.last_known_vote.get(self.name)

        if current_text_vote and current_text_vote != preferred and flips >= 1:
            anchor = current_text_vote
            prefix = f"You switched to Option {anchor}."
        else:
            anchor = preferred
            prefix = f"You lean toward Option {anchor} ({beliefs.key_concern})."

        # Emergence mode: facilitate gradual softening rather than position defense
        if state.phase == "emergence":
            candidate = state.candidate_option or state.current_leading_option
            if candidate and candidate in beliefs.acceptable and candidate != anchor:
                cond = f" Concession condition: {beliefs.concession}." if beliefs.concession else ""
                if self.persona.agreeableness >= 4:
                    return (
                        f"\n{prefix} Option {candidate} is gaining traction and is in your acceptable range."
                        f"{cond} You seek consensus — express conditional openness: "
                        "'I can see why others lean that way', 'if [concern] is sorted, I'm open'."
                    )
                elif self.persona.contrarian >= 4:
                    return (
                        f"\n{prefix} Option {candidate} is gaining ground and within your acceptable range."
                        f"{cond} Before softening, name one remaining specific concern — then signal openness."
                    )
                else:
                    return (
                        f"\n{prefix} Option {candidate} is gaining traction and is in your acceptable range."
                        f"{cond} Reduce resistance — conditional openness, not a full flip."
                    )
            elif candidate and candidate in (beliefs.rejected or []):
                return (
                    f"\n{prefix} Option {candidate} is gaining ground but you genuinely oppose it. "
                    "Acknowledge what others see in it, then state your specific remaining objection."
                )
            elif candidate and candidate == anchor:
                return (
                    f"\n{prefix} Option {candidate} is gaining ground — it's your preferred choice. "
                    "Help it land without being heavy-handed."
                )
            return (
                f"\n{prefix} The group is moving toward resolution. "
                "Let your position soften if it's genuinely softening."
            )

        # Standard coherence anchor for negotiation / narrowing / confirmation
        coherence = f" Keep your arguments consistent with preferring Option {anchor}."

        if state.phase == "negotiation":
            return f"\n{prefix}{coherence}"

        if flips == 0:
            return (
                f"\n{prefix}{coherence} "
                "Only change if someone raises a genuinely new argument you hadn't considered."
            )
        if flips == 1:
            return (
                f"\n{prefix} Already switched once.{coherence} "
                "Hold this — defend it with a new reason, not a repeat."
            )
        return (
            f"\n{prefix} Switched {flips} times. Commit now — no more switching.{coherence}"
        )

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
        """Extract n-grams appearing >= min_count times in recent turns — injected as forbidden."""
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

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------

    def _contrarian_nudge(self, state: "DialogueState") -> str:
        if state.phase == "emergence":
            # Emergence phase: contrarian softening is handled by _position_discipline
            return ""
        leading = state.current_leading_option
        if not leading:
            return ""
        if self.persona.contrarian >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "Probe a weakness or raise an overlooked concern — "
                "but pick a DIFFERENT angle if you've already made that point recently."
            )
        return ""

    def _last_participant_line(self, history: list[str]) -> str:
        """Most recent line from someone other than self, for soft context."""
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() not in cfg.EXCLUDED_SPEAKERS and speaker.strip() != self.name:
                return f"{speaker.strip()}: {msg.strip()}"
        return ""
