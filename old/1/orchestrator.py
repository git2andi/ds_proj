import datetime
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from modules.llm_client import get_llm_client
from modules.turn_manager import TurnManager


@dataclass
class DialogueState:
    phase: str = "opening"
    turn_index: int = 0

    has_asked_narrowing: bool = False
    agreement_reached: bool = False

    preferred_option: Optional[str] = None
    backup_option: Optional[str] = None
    current_leading_option: Optional[str] = None

    last_addressed: Optional[str] = None
    pending_question_target: Optional[str] = None
    pending_reply_target: Optional[str] = None

    repetition_pressure: float = 0.0
    important_events: List[str] = field(default_factory=list)

    # Counts consecutive rounds of high repetition after narrowing was asked.
    stall_rounds: int = 0


class Orchestrator:
    def __init__(self, topic: str):
        self.topic = topic
        self.sims = []

        self.llm = get_llm_client()
        self.turn_manager = TurnManager()
        self.state = DialogueState()

        self.options, self.opening_question = self._generate_options(topic)
        self.history = self._build_initial_history()
        self.log_file = self._create_log_file()

    def add_sim(self, sim) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _create_log_file(self) -> str:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"logs/conv_{timestamp}.txt"

    def _generate_options(self, topic: str) -> Tuple[List[str], str]:
        """
        Generate 4 concrete options with key attributes and a topic-specific
        opening question. Returns (options_list, opening_question).
        """
        prompt = f"""
You are preparing a facilitated group decision discussion.

Topic:
{topic}

Task:
1. Generate exactly 4 concrete, comparable decision options for this topic.
2. Write a short opening question the moderator will ask to kick off discussion.

Requirements for options:
- Each option must include 2-3 concrete attributes that participants can actually compare
  (e.g. for travel: approx. price range, travel time, comfort level;
       for a restaurant choice: cuisine type, price range, distance;
       for a hiring decision: salary, experience level, availability).
- Infer sensible approximate values from the topic context — do NOT use placeholders like "TBD" or "varies".
- Keep each option to one concise line.
- All 4 options must be genuinely different trade-offs.

Requirements for opening_question:
- One short sentence tailored to this specific topic and its options.
- Should prompt participants to share what matters most to them.
- Keep it natural and conversational, not generic.

Return valid JSON only, using exactly this schema:
{{
  "options": [
    "Option A - [label]: [attr1], [attr2], [attr3]",
    "Option B - [label]: [attr1], [attr2], [attr3]",
    "Option C - [label]: [attr1], [attr2], [attr3]",
    "Option D - [label]: [attr1], [attr2], [attr3]"
  ],
  "opening_question": "..."
}}
Do not include markdown or explanations outside the JSON.
"""

        fallback_options = [
            "Option A - Budget choice: lowest cost, basic features, some trade-offs.",
            "Option B - Convenience choice: faster or easier, moderately priced.",
            "Option C - Quality choice: best comfort or outcome, higher cost.",
            "Option D - Flexible choice: mixed trade-offs, adaptable to needs.",
        ]
        fallback_question = "What matters most to each of you, and which option looks best right now?"

        try:
            data = self.llm.generate_json(prompt)
            options_raw = data.get("options", [])
            question = str(data.get("opening_question", "")).strip() or fallback_question

            if not isinstance(options_raw, list) or len(options_raw) != 4:
                return fallback_options, fallback_question

            cleaned = []
            for i, option in enumerate(options_raw):
                if not isinstance(option, str) or not option.strip():
                    return fallback_options, fallback_question
                label = chr(ord("A") + i)
                text = option.strip()
                if not text.lower().startswith(f"option {label.lower()}"):
                    text = f"Option {label} - {text}"
                cleaned.append(text)

            return cleaned, question

        except Exception as e:
            print(f"!! Option generation error: {e}")
            return fallback_options, fallback_question

    def _build_initial_history(self) -> List[str]:
        history = [f"Moderator: Let's discuss: {self.topic}"]
        history.append("Moderator: Here are the options on the table:")
        for option in self.options:
            history.append(f"Moderator: {option}")
        history.append(f"Moderator: {self.opening_question}")
        return history

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log_header(self) -> None:
        names = [sim.name for sim in self.sims]
        header = (
            f"Participants: {', '.join(names)}\n"
            f"Topic: {self.topic}\n"
            + "=" * 40 + "\n"
        )
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in self.history:
                f.write(f"{line}\n")
            f.write("\n")

    def _store_line(self, line: str) -> None:
        self.history.append(line)
        print(f"-> {line}\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{line}\n\n")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _participant_turn_count(self) -> int:
        return sum(
            1 for line in self.history
            if ":" in line and line.split(":", 1)[0].strip() != "Moderator"
        )

    def _extract_option_mentions(self, text: str) -> List[str]:
        return [m.upper() for m in re.findall(r"\boption\s+([A-D])\b", text.lower())]

    def _update_leading_option(self) -> None:
        mentions = []
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, msg = line.split(":", 1)
            if speaker.strip() == "Moderator":
                continue
            mentions.extend(self._extract_option_mentions(msg))
            if len(mentions) >= max(5, len(self.sims) * 2):
                break
        if mentions:
            self.state.current_leading_option = Counter(mentions).most_common(1)[0][0]

    def _update_phase(self) -> None:
        if self.state.agreement_reached:
            self.state.phase = "closure"
            return
        turns = self._participant_turn_count()
        if turns == 0:
            self.state.phase = "opening"
        elif turns < len(self.sims):
            self.state.phase = "preference_expression"
        elif self.state.has_asked_narrowing:
            self.state.phase = "narrowing"
        else:
            self.state.phase = "negotiation"

    def _should_narrow(self) -> bool:
        if self.state.has_asked_narrowing:
            return False
        turns = self._participant_turn_count()
        return turns >= len(self.sims) * 2 or self.state.repetition_pressure >= 0.55

    def _add_narrowing_prompt(self) -> None:
        self.state.has_asked_narrowing = True
        self.state.phase = "narrowing"
        self._store_line(
            "Moderator: Let's narrow this down — which option do you each prefer? "
            "A backup is fine if you're unsure."
        )

    # ------------------------------------------------------------------
    # Consensus detection
    # ------------------------------------------------------------------

    def _regex_detect_consensus(self) -> Optional[Tuple[str, Optional[str]]]:
        """
        Fast regex-based check. Requires a majority of participants to have
        mentioned the same option in recent lines.
        """
        recent = []
        for line in reversed(self.history):
            if ":" not in line:
                continue
            speaker, _ = line.split(":", 1)
            if speaker.strip() == "Moderator":
                continue
            recent.append(line)
            if len(recent) >= max(6, len(self.sims) * 3):
                break

        if len(recent) < len(self.sims):
            return None

        # Collect the most recent vote per speaker (iterate oldest→newest).
        latest_vote: dict = {}
        for line in reversed(recent):
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            mentions = self._extract_option_mentions(msg)
            if not mentions:
                continue
            # Overwrite so the newest mention wins.
            latest_vote[speaker] = mentions[0]

        if len(latest_vote) < len(self.sims):
            return None

        counts = Counter(latest_vote.values())
        top_option, top_count = counts.most_common(1)[0]

        if top_count < max(2, len(self.sims) - 1):
            return None

        return top_option, None

    def _llm_detect_consensus(self) -> Optional[Tuple[str, Optional[str]]]:
        """
        LLM-based consensus check. More reliable than regex when participants
        express agreement in natural language without repeating option letters.
        Only called after narrowing has been requested.
        """
        names = [sim.name for sim in self.sims]
        recent_dialogue = "\n".join(self.history[-20:])
        n_needed = max(2, len(self.sims) - 1)

        prompt = f"""
Participants: {", ".join(names)}
Options available: {", ".join(self.options)}

Recent dialogue:
{recent_dialogue}

Task:
Has a clear majority (at least {n_needed} out of {len(self.sims)} participants) agreed on one option?

Rules:
- A participant "agrees" if they have clearly expressed support for one option in the recent dialogue.
- Asking a question about an option does NOT count as agreeing to it.
- Do not invent votes that are not in the dialogue.
- preferred_option must be a single letter A, B, C, or D.

Return valid JSON only:
{{
  "consensus_reached": true or false,
  "preferred_option": "A" or "B" or "C" or "D" or null,
  "backup_option": "A" or "B" or "C" or "D" or null
}}
"""
        try:
            data = self.llm.generate_json(prompt)
            if not data.get("consensus_reached"):
                return None
            opt = str(data.get("preferred_option") or "").strip().upper()
            bak = str(data.get("backup_option") or "").strip().upper() or None
            if opt in {"A", "B", "C", "D"}:
                if bak not in {"A", "B", "C", "D"} or bak == opt:
                    bak = None
                return opt, bak
        except Exception as e:
            print(f"!! LLM consensus check error: {e}")
        return None

    def _detect_consensus(self) -> Optional[Tuple[str, Optional[str]]]:
        """Try regex first; fall back to LLM after narrowing."""
        result = self._regex_detect_consensus()
        if result:
            return result
        if self.state.has_asked_narrowing:
            return self._llm_detect_consensus()
        return None

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_participant_round(self) -> bool:
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        self._update_phase()

        max_speakers = 2 if len(self.sims) <= 3 else 3
        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=max_speakers,
        )

        active_round = False
        for sim in selected:
            text = sim.generate_turn(self.history, self.state)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}")
                active_round = True

        # Re-extract after new lines are added.
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        return active_round

    def _run_confirmation(self) -> None:
        self.state.phase = "confirmation"
        preferred = self.state.preferred_option
        backup = self.state.backup_option

        if preferred is None:
            return

        if backup:
            self._store_line(
                f"Moderator: It sounds like Option {preferred} is the preferred choice, "
                f"with Option {backup} as a backup. Can everyone confirm briefly?"
            )
        else:
            self._store_line(
                f"Moderator: It sounds like Option {preferred} is the preferred choice. "
                f"Can everyone confirm briefly?"
            )

        self.turn_manager.extract_events(self.history, self.state, self.sims)
        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=min(2, len(self.sims)),
        )
        for sim in selected:
            text = sim.generate_turn(self.history, self.state)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}")

    def _run_goodbye(self) -> None:
        """Let each participant say a short natural closing remark."""
        self.state.phase = "closure"
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=len(self.sims),
        )
        for sim in selected:
            text = sim.generate_turn(self.history, self.state)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}")

    def _close(self) -> None:
        preferred = self.state.preferred_option
        backup = self.state.backup_option

        if preferred and backup:
            self._store_line(
                f"Moderator: Agreed — Option {preferred} is the final choice, "
                f"with Option {backup} as backup. Discussion concluded."
            )
        elif preferred:
            self._store_line(
                f"Moderator: Agreed — Option {preferred} is the final choice. "
                f"Discussion concluded."
            )
        else:
            self._store_line("Moderator: Discussion concluded.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self, max_turns: int = 12) -> None:
        self._write_log_header()

        print("\n--- Simulation Started ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        for _ in range(max_turns):
            self.state.turn_index += 1

            active_round = self._run_participant_round()

            # Check for agreement after every round — can fire before narrowing too.
            consensus = self._detect_consensus()
            if consensus:
                self.state.preferred_option, self.state.backup_option = consensus
                self.state.agreement_reached = True
                self._run_confirmation()
                self._run_goodbye()
                self._close()
                return

            if not active_round:
                self._store_line("Moderator: No further progress. Discussion concluded.")
                return

            if self._should_narrow():
                self._add_narrowing_prompt()
                continue

            # Stall detection: if repetition stays high after narrowing, force close.
            if self.state.has_asked_narrowing:
                if self.state.repetition_pressure >= 0.75:
                    self.state.stall_rounds += 1
                else:
                    self.state.stall_rounds = 0

                if self.state.stall_rounds >= 2:
                    # One last LLM consensus attempt before giving up.
                    forced = self._llm_detect_consensus()
                    if forced:
                        self.state.preferred_option, self.state.backup_option = forced
                        self.state.agreement_reached = True
                        self._run_confirmation()
                        self._run_goodbye()
                        self._close()
                    else:
                        leading = self.state.current_leading_option
                        if leading:
                            self.state.preferred_option = leading
                            self._store_line(
                                f"Moderator: We seem to be going in circles. "
                                f"Based on the discussion, Option {leading} appears to be "
                                f"the most supported choice. Discussion concluded."
                            )
                        else:
                            self._store_line(
                                "Moderator: We were unable to reach a clear agreement. "
                                "Discussion concluded."
                            )
                    return

        self._store_line("Moderator: Maximum discussion length reached. Discussion concluded.")
