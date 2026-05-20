"""
simulator.py
------------
Simulator - wraps a Persona and generates one dialogue turn via LLM.

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
            return self._closure_line(state), 0, 0

        raw = self._generate_decision(history, state, forced_adaptation)
        tok_in = self._llm.last_tokens_in
        tok_out = self._llm.last_tokens_out

        if not raw:
            return "[SILENCE]", tok_in, tok_out

        if raw.lower().startswith(f"{self.name.lower()}:"):
            raw = raw.split(":", 1)[1].strip()

        raw = self._enforce_word_budget(raw, state.phase)

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
            "Name your current preferred option clearly (e.g. 'I prefer Option A'). "
            "You may mention a backup if it is genuinely plausible. "
            "Once you have stated a preference, change it at most once, and only because a specific new point "
            "actually changes your mind. Do not switch just because the group repeats itself."
        )
        if state.phase == "narrowing" and not self._has_voted(history):
            narrowing_instruction = (
                "You have not yet stated a preferred option. "
                "Before anything else this turn, name your preferred option explicitly. "
                + narrowing_base
            )
        elif state.phase == "narrowing":
            narrowing_instruction = (
                "You already named your preferred option. Do not repeat 'I prefer Option X'. "
                "Now give one condition, concern, or possible compromise."
            )
        else:
            narrowing_instruction = narrowing_base

        phase_instructions = {
            "greeting": (
                "Say a quick, casual hello. One short line only. "
                "Do not introduce your role and do not discuss the topic yet."
            ),
            "opening": (
                "Give your first real reaction. Name what matters to you and why, briefly. "
                "A short concern or question is fine, but do not make the whole turn only a vague question."
            ),
            "negotiation": (
                "Discuss the actual point on the table. Answer if someone asked something, or respond to a claim. "
                "You may disagree, build on it, give a reason, or make a concrete trade-off. "
                "Do not simply say your preferred option is better for the same reason again. "
                "Only use facts listed in the options; do not invent hidden attributes."
            ),
            "narrowing": narrowing_instruction,
            "emergence": (
                "The group is trying to land somewhere. "
                "Say what would make the leading option workable, or name the concrete concern that still blocks you. "
                "Do not repeat your original reason; respond to someone else's objection or offer a compromise condition."
            ),
            "confirmation": (
                "The moderator is asking for final confirmation. "
                "Reply with an explicit yes or no. A question, hedge, or silence is treated as no. "
                "Only say no if you have a specific objection you have not yet raised."
            ),
            "closure": (
                "The discussion is over. Write one short goodbye only. "
                "Nothing about the topic. No opinions. No questions."
            ),
        }

        phase_instr = phase_instructions.get(state.phase, "React naturally to the conversation.")
        if (state.repetition_pressure >= 0.55 or state.post_narrowing_rounds >= 2) \
                and state.phase not in {"greeting", "opening", "closure", "confirmation"}:
            phase_instr += " Add only one new useful thing; do not re-explain your full case."

        is_closure = state.phase == "closure"
        prompt = prompts.sim_turn(
            name=self.name,
            topic=self.topic,
            options_text=self._format_options(),
            goal=self.persona.goal,
            backstory=self.persona.backstory,
            personality_summary=self.persona.personality_summary(),
            style_rule=self.persona.style_rule(),
            max_words=self.persona.max_words(state.phase),
            phase=state.phase,
            phase_instruction=phase_instr,
            interaction_instruction=self._interaction_instruction(history, state),
            own_recent_points=self._own_recent_points_block(history),
            recent_history=self._recent_history(history, max_lines=4 if is_closure else 8),
            forbidden_openers=self._recent_openers(history),
            forbidden_frames=list(cfg.repetition.forbidden_frames),
            beliefs_block="" if is_closure else self._beliefs_block(),
            last_speaker_line="" if is_closure else self._last_participant_line(history),
            position_discipline="" if is_closure else self._position_discipline(state),
            skepticism_nudge=self._skepticism_nudge(state),
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

        parts = ["Private anchor - stay coherent, do not recite:"]
        parts.append(f"  Prefer {b.preferred}; concern: {b.key_concern}")

        other_acceptable = [x for x in b.acceptable if x != b.preferred]
        if other_acceptable:
            opts_str = ", ".join(f"Option {x}" for x in other_acceptable)
            parts.append(f"  Could accept: {opts_str}; if: {b.concession}")

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

        if state.phase == "emergence":
            candidate = state.candidate_option or state.current_leading_option
            if candidate and candidate in beliefs.acceptable and candidate != anchor:
                cond = f" Concession condition: {beliefs.concession}." if beliefs.concession else ""
                if self.persona.agreeableness >= 4:
                    return (
                        f"\n{prefix} Option {candidate} is acceptable."
                        f"{cond} Give one short condition."
                    )
                if self.persona.agreeableness <= 2 or self.persona.neuroticism >= 4:
                    return (
                        f"\n{prefix} Option {candidate} is acceptable."
                        f"{cond} Name one concern before softening."
                    )
                return (
                    f"\n{prefix} Option {candidate} is acceptable."
                    f"{cond} Soften briefly."
                )
            if candidate and candidate in (beliefs.rejected or []):
                # Soften if: moderately agreeable (>=3) OR spontaneous/flexible (conscientiousness <=2),
                # AND emergence has been going long enough to warrant a decision.
                can_soften = (
                    (self.persona.agreeableness >= 3 or self.persona.conscientiousness <= 2)
                    and state.emergence_rounds >= 3
                )
                if can_soften:
                    cond = f" Your concession condition: {beliefs.concession}." if beliefs.concession else ""
                    return (
                        f"\n{prefix} Option {candidate} is where the group is heading and you can find common ground when needed."
                        f"{cond} You've raised concerns already — decide now: name one specific condition that would let you accept it,"
                        " or say it is a genuine dealbreaker and why."
                    )
                return (
                    f"\n{prefix} Option {candidate} is gaining ground but you genuinely oppose it. "
                    "Acknowledge the direction briefly, then state your specific remaining objection."
                )
            if candidate and candidate == anchor:
                return (
                    f"\n{prefix} Option {candidate} is gaining ground - it is your preferred choice. "
                    "Help it land without being heavy-handed."
                )
            return (
                f"\n{prefix} The group is moving toward resolution. "
                "Let your position soften if it is genuinely softening."
            )

        coherence = f" Stay consistent with Option {anchor}."

        if state.phase == "negotiation":
            return f"\n{prefix}{coherence}"

        if flips == 0:
            return (
                f"\n{prefix}{coherence} "
                "Only change for a genuinely new reason."
            )
        if flips == 1:
            return (
                f"\n{prefix} Already switched once.{coherence} "
                "Hold this; no repeat."
            )
        return (
            f"\n{prefix} Switched {flips} times. Commit now - no more switching.{coherence}"
        )

    # ------------------------------------------------------------------
    # Context formatters
    # ------------------------------------------------------------------

    def _format_options(self) -> str:
        return "\n".join(f"  {opt}" for opt in self.options)

    def _recent_history(self, history: list[str], max_lines: int = 12) -> str:
        return "\n".join(history[-max_lines:])

    def _interaction_instruction(self, history: list[str], state: "DialogueState") -> str:
        """Dynamic prompt guidance for conversational obligations."""
        if state.phase in {"greeting", "closure", "confirmation"}:
            return ""

        last = self._last_participant_line(history)
        recent = self._recent_participant_messages(history, limit=5)
        question_count = sum(1 for msg in recent if "?" in msg)

        parts: list[str] = []
        if last and "?" in last:
            speaker = last.split(":", 1)[0].strip()
            if speaker != self.name:
                parts.append(
                    " The last participant asked a question. Answer it directly if you can; "
                    "if the options do not contain enough information, say what you can infer and then give your view."
                )

        if question_count >= 3:
            parts.append(
                " The chat has too many unanswered questions. Do not ask another one; make a claim, answer, or choose a trade-off."
            )

        compromise = getattr(state, "compromise_option", None)
        if compromise and state.phase == "emergence":
            b = self.persona.beliefs
            if b and compromise in b.acceptable:
                parts.append(
                    f" The moderator is testing Option {compromise} as a compromise. "
                    f"Say whether you could live with Option {compromise}; name a condition that applies to that option."
                )
            else:
                parts.append(
                    f" The moderator is testing Option {compromise} as a compromise. "
                    f"If Option {compromise} does not work for you, say no and give the specific objection to that option."
                )

        rejected = getattr(state, "last_rejected_option", None)
        rejecting_speaker = getattr(state, "last_rejecting_speaker", None)
        if rejected and rejecting_speaker == self.name and state.phase in {"narrowing", "emergence"}:
            # Count how many turns this sim has spoken since rejecting
            turns_since_rejection = len(self._own_recent_turns(history, limit=10))
            if turns_since_rejection >= 3:
                parts.append(
                    f" You have raised multiple concerns about Option {rejected} and the group has tried to address them."
                    f" Name your single concrete dealbreaker for Option {rejected}, or say whether you can accept it with one specific condition."
                    " Do not raise a new objection — commit to a position."
                )
            else:
                parts.append(
                    f" You just rejected Option {rejected}. Explain the main blocker clearly, "
                    "and say what would need to change for you to consider it."
                )

        own_recent = self._last_own_turn(history)
        if own_recent and "?" in own_recent:
            parts.append(
                " Your previous turn was a question, so this turn should not be another bare question."
            )

        # Detect "but what if / what if" as a speculative-question crutch
        own_turns = self._own_recent_turns(history, limit=4)
        speculative_count = sum(
            1 for t in own_turns
            if re.search(r"\bwhat if\b", t, re.IGNORECASE)
        )
        if speculative_count >= 2:
            parts.append(
                " You've asked 'what if' hypotheticals multiple times. Stop — make a direct claim or statement instead."
            )

        repeated_kws = self._repeated_concern_keywords(history)
        if len(repeated_kws) >= 2:
            kw_str = ", ".join(repeated_kws[:3])
            parts.append(
                f" You keep returning to the same theme ({kw_str}). That point is on the table — don't repeat it."
                " Either name a concrete dealbreaker, propose a condition, or move to a completely different angle."
            )
        elif self._own_recent_repetition(history):
            parts.append(
                " Your recent turns are repeating the same point. Change move: answer someone, name a downside of your own option, or offer a compromise."
            )

        return "\n" + "".join(parts) if parts else ""

    def _own_recent_points_block(self, history: list[str], limit: int = 3) -> str:
        turns = self._own_recent_turns(history, limit=limit)
        if not turns:
            return ""
        compact = " | ".join(turns)
        return (
            "\nYour recent points: "
            f"{compact}\nDo not repeat these points. Add a new angle or respond to someone else's point."
        )

    def _own_recent_repetition(self, history: list[str]) -> bool:
        turns = self._own_recent_turns(history, limit=2)
        if len(turns) < 2:
            return False
        a = set(re.sub(r"[^\w\s]", "", turns[0].lower()).split())
        b = set(re.sub(r"[^\w\s]", "", turns[1].lower()).split())
        if not a or not b:
            return False
        return len(a & b) / max(1, min(len(a), len(b))) >= 0.45

    def _repeated_concern_keywords(self, history: list[str]) -> list[str]:
        """Keywords this sim mentioned in 2+ of their last 5 turns — signals semantic looping."""
        turns = self._own_recent_turns(history, limit=5)
        if len(turns) < 2:
            return []
        stopwords = {
            "that", "this", "with", "from", "have", "would", "could", "should",
            "there", "their", "about", "option", "think", "just", "like", "work",
            "good", "need", "want", "make", "sure", "know", "okay", "also", "even",
            "very", "more", "some", "than", "when", "what", "which", "been", "will",
            "still", "into", "back", "each", "much", "both", "your", "they", "them",
            "then", "well", "here", "feel", "does", "might", "only", "most", "over",
            "really", "think", "right", "actually", "though", "going", "mean",
        }
        from collections import Counter as _Counter
        turn_word_sets: list[set[str]] = []
        for turn in turns:
            words = set(re.sub(r"[^\w\s]", "", turn.lower()).split())
            meaningful = {w for w in words if len(w) >= 4 and w not in stopwords}
            turn_word_sets.append(meaningful)
        word_counts: _Counter = _Counter()
        for ws in turn_word_sets:
            word_counts.update(ws)
        return [w for w, count in word_counts.items() if count >= 2]

    def _own_recent_turns(self, history: list[str], limit: int) -> list[str]:
        turns: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                turns.append(msg.strip())
                if len(turns) >= limit:
                    break
        return turns

    def _recent_participant_messages(self, history: list[str], limit: int) -> list[str]:
        messages: list[str] = []
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            messages.append(msg.strip())
            if len(messages) >= limit:
                break
        return messages

    def _last_own_turn(self, history: list[str]) -> str:
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == self.name:
                return msg.strip()
        return ""

    def _enforce_word_budget(self, text: str, phase: str) -> str:
        """Hard trim overlong LLM output so the transcript remains chat-like."""
        max_words = self.persona.max_words(phase)
        clean = re.sub(r"\s+", " ", text).strip()
        words = clean.split()
        if len(words) <= max_words:
            return clean

        sentences = re.split(r"(?<=[.!?])\s+", clean)
        kept: list[str] = []
        count = 0
        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue
            if count + len(s_words) <= max_words:
                kept.append(sentence.strip())
                count += len(s_words)
            elif kept:
                break
            else:
                kept = s_words[:max_words]
                break

        if kept and isinstance(kept[0], str) and " " in kept[0]:
            trimmed = " ".join(kept)
        else:
            trimmed = " ".join(kept)

        trimmed = trimmed.strip(" ,;:")
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed

    def _recent_openers(self, history: list[str], n: int = 6) -> str:
        """First 1-2 words of recent participant turns — prevents repetitive openings.

        Tracks both single-word openers (e.g. "honestly") and two-word phrases
        (e.g. "hmm yeah", "but what") so the prompt can forbid them by exact string.
        """
        openers: list[str] = []
        seen: set[str] = set()
        for line in reversed(history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() in cfg.EXCLUDED_SPEAKERS:
                continue
            words = msg.strip().split()
            if not words:
                continue
            w1 = words[0].rstrip(",.!?").lower()
            w2 = (words[1].rstrip(",.!?").lower() if len(words) > 1 else "")
            two_word = f"{w1} {w2}".strip() if w2 else w1
            key = two_word
            if key and key not in seen:
                openers.append(two_word)
                seen.add(key)
                seen.add(w1)  # also block the bare first word
            if len(openers) >= n:
                break
        return ", ".join(f'"{o}"' for o in openers) if openers else ""

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------

    def _skepticism_nudge(self, state: "DialogueState") -> str:
        if state.phase == "emergence":
            return ""
        leading = state.current_leading_option
        if not leading:
            return ""
        if self.persona.agreeableness <= 2 or self.persona.neuroticism >= 4:
            return (
                f"\nIMPORTANT: The group is leaning toward Option {leading}. "
                "If you are not convinced, raise one specific weakness or risk from a fresh angle."
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

    def _closure_line(self, state: "DialogueState") -> str:
        option = getattr(state, "preferred_option", None)
        if option and random.random() < 0.65:
            templates = [
                f"Okay, Option {option} works for me. Bye!",
                f"Option {option} it is then. See you!",
                f"Sounds good, Option {option}. Bye!",
                f"Alright, let's go with Option {option}. See you.",
            ]
            return random.choice(templates)
        return random.choice(self._GOODBYES)
