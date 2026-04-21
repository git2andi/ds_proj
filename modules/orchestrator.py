import csv
import datetime
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    stall_rounds: int = 0


class Orchestrator:
    CSV_COLUMNS = [
        "dialogue_id",
        "turn_index",
        "phase",
        "speaker",
        "is_moderator",
        "text",
        "selected_reason",
        "last_addressed",
        "pending_question_target",
        "pending_reply_target",
        "repetition_pressure",
        "role",
        "is_primary",
        "friendliness",
        "assertiveness",
        "talkativeness",
        "initiative",
        "agreeableness",
        "flexibility",
        "patience",
        "response_length",
        "contrarian_pressure",
        "focus_cost",
        "focus_comfort",
        "focus_time",
        "focus_safety",
        "focus_flexibility",
    ]

    # ------------------------------------------------------------------
    # moderator_style controls how actively the moderator intervenes:
    #   "active"  — intervenes to narrow, confirm, and close (original behaviour)
    #   "minimal" — only intervenes if the group is genuinely stuck (stall_rounds >= 3)
    #   "passive" — never intervenes after the opening; participants self-organise
    # ------------------------------------------------------------------
    def __init__(self, topic: str, moderator_style: str = "active") -> None:
        self.topic = topic
        self.moderator_style = moderator_style.lower()
        self.sims: List[Any] = []

        self.llm = get_llm_client()
        self.turn_manager = TurnManager()
        self.state = DialogueState()

        self.dialogue_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.options, self.opening_question = self._generate_options(topic)
        self.history = self._build_initial_history()

        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/{self.dialogue_id}.txt"
        self.csv_file = f"logs/{self.dialogue_id}.csv"

        self._csv_rows: List[Dict[str, Any]] = []
        self._forced_names: set = set()

    def add_sim(self, sim: Any) -> None:
        self.sims.append(sim)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _generate_options(self, topic: str) -> Tuple[List[str], str]:
        prompt = f"""
You are preparing a facilitated group decision discussion.

Topic:
{topic}

Task:
1. Generate exactly 4 concrete, comparable decision options for this topic.
2. Write a short opening question the moderator will ask to kick off discussion.

Requirements for options:
- Each option must include 2-3 concrete attributes that participants can actually compare and fit to the topic.
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
    # Logging — txt
    # ------------------------------------------------------------------

    def _write_log_header(self) -> None:
        names = [sim.name for sim in self.sims]
        header = (
            f"Dialogue ID: {self.dialogue_id}\n"
            f"Participants: {', '.join(names)}\n"
            f"Topic: {self.topic}\n"
            f"Moderator style: {self.moderator_style}\n"
            + "=" * 40 + "\n"
        )
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(header)
            for line in self.history:
                f.write(f"{line}\n")
            f.write("\n")

    def _store_line(self, line: str, selected_reason: str = "") -> None:
        self.history.append(line)
        print(f"-> {line}\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{line}\n\n")
        self._buffer_csv_row(line, selected_reason)

    def _store_moderator(self, text: str) -> None:
        self._store_line(f"Moderator: {text}", selected_reason="moderator")

    # ------------------------------------------------------------------
    # Logging — CSV
    # ------------------------------------------------------------------

    def _sim_by_name(self, name: str) -> Optional[Any]:
        for sim in self.sims:
            if sim.name == name:
                return sim
        return None

    def _buffer_csv_row(self, line: str, selected_reason: str) -> None:
        if ":" not in line:
            return

        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()
        is_moderator = speaker == "Moderator"

        row: Dict[str, Any] = {
            "dialogue_id": self.dialogue_id,
            "turn_index": self.state.turn_index,
            "phase": self.state.phase,
            "speaker": speaker,
            "is_moderator": is_moderator,
            "text": text,
            "selected_reason": selected_reason,
            "last_addressed": self.state.last_addressed or "",
            "pending_question_target": self.state.pending_question_target or "",
            "pending_reply_target": self.state.pending_reply_target or "",
            "repetition_pressure": round(self.state.repetition_pressure, 3),
            "role": "",
            "is_primary": "",
            "friendliness": "",
            "assertiveness": "",
            "talkativeness": "",
            "initiative": "",
            "agreeableness": "",
            "flexibility": "",
            "patience": "",
            "response_length": "",
            "contrarian_pressure": "",
            "focus_cost": "",
            "focus_comfort": "",
            "focus_time": "",
            "focus_safety": "",
            "focus_flexibility": "",
        }

        if not is_moderator:
            sim = self._sim_by_name(speaker)
            if sim:
                p = sim.persona
                focus = p.get("focus", {})
                row.update({
                    "role": p.get("role", ""),
                    "is_primary": p.get("is_primary", False),
                    "friendliness": p.get("friendliness", ""),
                    "assertiveness": p.get("assertiveness", ""),
                    "talkativeness": p.get("talkativeness", ""),
                    "initiative": p.get("initiative", ""),
                    "agreeableness": p.get("agreeableness", ""),
                    "flexibility": p.get("flexibility", ""),
                    "patience": p.get("patience", ""),
                    "response_length": p.get("response_length", ""),
                    "contrarian_pressure": p.get("contrarian_pressure", ""),
                    "focus_cost": focus.get("cost", ""),
                    "focus_comfort": focus.get("comfort", ""),
                    "focus_time": focus.get("time", ""),
                    "focus_safety": focus.get("safety", ""),
                    "focus_flexibility": focus.get("flexibility_focus", ""),
                })

        self._csv_rows.append(row)

    def _flush_csv(self) -> None:
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._csv_rows)

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

    # ------------------------------------------------------------------
    # Moderator narrowing — style-aware and organically triggered
    # ------------------------------------------------------------------

    def _should_narrow(self) -> bool:
        """
        Decide whether the moderator should prompt participants to narrow down.
        Behaviour depends on moderator_style:

          passive  — never narrows; participants must converge on their own.
          minimal  — only narrows as an absolute last resort when severely stalled.
          active   — narrows when the conversation has genuinely run its course
                     (not on a fixed turn count, but on stall signals).
        """
        if self.state.has_asked_narrowing:
            return False
        if self.moderator_style == "passive":
            return False

        turns = self._participant_turn_count()
        n = len(self.sims)

        # Everyone must have spoken at least twice before any narrowing.
        if turns < n * 2:
            return False

        genuinely_stalling = (
            self.state.repetition_pressure >= 0.75
            and self.state.stall_rounds >= 1
        )
        # Absolute ceiling: ~5 full rounds each without resolution.
        talked_plenty = turns >= n * 5

        if self.moderator_style == "minimal":
            # Only step in when truly stuck AND talked a lot.
            return genuinely_stalling and talked_plenty

        # "active": step in on genuine stall OR after plenty of discussion.
        return genuinely_stalling or talked_plenty

    def _add_narrowing_prompt(self) -> None:
        self.state.has_asked_narrowing = True
        self.state.phase = "narrowing"
        self._store_moderator(
            "Let's narrow this down — which option do you each prefer? "
            "A backup is fine if you're unsure."
        )

    # ------------------------------------------------------------------
    # Dynamic max_speakers
    # ------------------------------------------------------------------

    def _dynamic_max_speakers(self) -> int:
        """
        Vary speaker count per round stochastically based on phase and
        repetition pressure, so rounds don't feel templated.
        """
        phase = self.state.phase
        rep = self.state.repetition_pressure
        n = len(self.sims)

        if phase in ("opening", "closure"):
            return 1

        # A single fresh voice breaks a repetitive loop better than more of the same.
        if rep >= 0.65:
            return 1

        if phase == "confirmation":
            return min(2, n)

        # Normal rounds: weighted random leaning toward 2 speakers.
        weights = [0.25, 0.50, 0.25] if n >= 3 else [0.40, 0.60]
        choices = list(range(1, min(4, n + 1)))
        return random.choices(choices, weights=weights[:len(choices)])[0]

    # ------------------------------------------------------------------
    # Consensus detection
    # ------------------------------------------------------------------

    def _regex_detect_consensus(self) -> Optional[Tuple[str, Optional[str]]]:
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

        latest_vote: Dict[str, str] = {}
        for line in reversed(recent):
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            mentions = self._extract_option_mentions(msg)
            if mentions:
                latest_vote[speaker] = mentions[0]

        # Require all participants to have voted explicitly.
        if len(latest_vote) < len(self.sims):
            return None

        counts = Counter(latest_vote.values())
        top_option, top_count = counts.most_common(1)[0]

        # In passive/minimal mode require full unanimity;
        # in active mode allow one dissenter in larger groups.
        if self.moderator_style == "active":
            required = max(2, len(self.sims) - 1)
        else:
            required = len(self.sims)

        if top_count < required:
            return None

        return top_option, None

    def _llm_detect_consensus(self) -> Optional[Tuple[str, Optional[str]]]:
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
        result = self._regex_detect_consensus()
        if result:
            return result
        # LLM consensus check only after narrowing has been asked,
        # or in passive/minimal mode where participants self-organise.
        if self.state.has_asked_narrowing or self.moderator_style != "active":
            return self._llm_detect_consensus()
        return None

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_participant_round(self) -> bool:
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        self._update_phase()

        max_speakers = self._dynamic_max_speakers()
        forced_candidates = self.turn_manager.forced_names(self.state)

        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=max_speakers,
        )

        all_names = [sim.name for sim in self.sims]
        active_round = False
        for sim in selected:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name in forced_candidates else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason)
                active_round = True

        self.turn_manager.extract_events(self.history, self.state, self.sims)
        self._update_leading_option()
        return active_round

    def _run_confirmation(self) -> None:
        """
        In active mode the moderator explicitly calls for confirmation.
        In passive/minimal mode the participants confirm among themselves
        and the moderator stays silent.
        """
        self.state.phase = "confirmation"
        preferred = self.state.preferred_option
        backup = self.state.backup_option

        if preferred is None:
            return

        if self.moderator_style == "active":
            if backup:
                self._store_moderator(
                    f"It sounds like Option {preferred} is the preferred choice, "
                    f"with Option {backup} as a backup. Can everyone confirm briefly?"
                )
            else:
                self._store_moderator(
                    f"It sounds like Option {preferred} is the preferred choice. "
                    "Can everyone confirm briefly?"
                )

        # Let participants react to the consensus regardless of style.
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        max_speakers = min(2, len(self.sims))
        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=max_speakers,
        )
        forced_candidates = self.turn_manager.forced_names(self.state)
        all_names = [sim.name for sim in self.sims]
        for sim in selected:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                reason = "forced" if sim.name in forced_candidates else "weighted"
                self._store_line(f"{sim.name}: {text}", selected_reason=reason)

    def _run_goodbye(self) -> None:
        self.state.phase = "closure"
        self.turn_manager.extract_events(self.history, self.state, self.sims)
        selected = self.turn_manager.select_speakers(
            self.sims, self.history, self.state, max_speakers=len(self.sims),
        )
        all_names = [sim.name for sim in self.sims]
        for sim in selected:
            text = sim.generate_turn(self.history, self.state, all_names=all_names)
            if text and "[SILENCE]" not in text.upper():
                self._store_line(f"{sim.name}: {text}", selected_reason="closure")

    def _close(self) -> None:
        preferred = self.state.preferred_option
        backup = self.state.backup_option

        # In passive mode the moderator stays silent at the end too.
        if self.moderator_style == "passive":
            return

        if preferred and backup:
            self._store_moderator(
                f"Agreed — Option {preferred} is the final choice, "
                f"with Option {backup} as backup. Discussion concluded."
            )
        elif preferred:
            self._store_moderator(
                f"Agreed — Option {preferred} is the final choice. Discussion concluded."
            )
        else:
            self._store_moderator("Discussion concluded.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_simulation(self, max_turns: int = 15) -> None:
        self._write_log_header()

        for line in self.history:
            self._buffer_csv_row(line, selected_reason="moderator")

        print(f"\n--- Simulation Started (moderator: {self.moderator_style}) ---")
        for line in self.history:
            print(f"-> {line}")
        print()

        try:
            for _ in range(max_turns):
                self.state.turn_index += 1

                active_round = self._run_participant_round()

                consensus = self._detect_consensus()
                if consensus:
                    self.state.preferred_option, self.state.backup_option = consensus
                    self.state.agreement_reached = True
                    self._run_confirmation()
                    self._run_goodbye()
                    self._close()
                    return

                if not active_round:
                    if self.moderator_style != "passive":
                        self._store_moderator("No further progress. Discussion concluded.")
                    return

                if self._should_narrow():
                    self._add_narrowing_prompt()
                    continue

                if self.state.has_asked_narrowing:
                    if self.state.repetition_pressure >= 0.75:
                        self.state.stall_rounds += 1
                    else:
                        self.state.stall_rounds = 0

                    # Stall threshold: active=2, minimal=3, passive never forces.
                    stall_limit = {"active": 2, "minimal": 3, "passive": 999}.get(
                        self.moderator_style, 2
                    )

                    if self.state.stall_rounds >= stall_limit:
                        forced = self._llm_detect_consensus()
                        if forced:
                            self.state.preferred_option, self.state.backup_option = forced
                            self.state.agreement_reached = True
                            self._run_confirmation()
                            self._run_goodbye()
                            self._close()
                        else:
                            leading = self.state.current_leading_option
                            if leading and self.moderator_style != "passive":
                                self.state.preferred_option = leading
                                self._store_moderator(
                                    f"We seem to be going in circles. "
                                    f"Based on the discussion, Option {leading} appears to be "
                                    f"the most supported choice. Discussion concluded."
                                )
                            elif self.moderator_style != "passive":
                                self._store_moderator(
                                    "We were unable to reach a clear agreement. "
                                    "Discussion concluded."
                                )
                        return

            if self.moderator_style != "passive":
                self._store_moderator(
                    "Maximum discussion length reached. Discussion concluded."
                )

        finally:
            self._flush_csv()
            print(f"\n[Logs saved: {self.log_file} | {self.csv_file}]")
