"""
state.py
--------
Structured dialogue state -- the single source of truth the orchestrator
decides from.

  - DialogueAct enum (acts actually used in planning / detection)
  - TurnRecord, StanceUpdate dataclasses
  - StanceTable + OptionState (per speaker x option public stance)
  - DiscourseGraph (Ouchi & Tsuboi 2016) -- now also tracks challenges and
    answers so deliberation-gated narrowing can read it (Stage 5).
  - ParticipantState -- public stance, cooldowns, debt, AND the compact
    per-sim memory introduced in Stage 6 (Park 2023 scaled down):
      * points_made          : the speaker's own recent points (anti-repeat)
      * open_challenges_to_me: pushbacks aimed at this sim that aren't answered
      * others_key_arguments : recent live arguments from other sims
      * stated_priority      : what they said in the opening (their priority)
  - StructuredState container
  - StateTracker -- deterministic raw line -> structured update
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, TYPE_CHECKING

from config_loader import cfg
from utils import OptionResolver

if TYPE_CHECKING:
    from persona import Persona


# =============================================================================
# Dialogue acts
# =============================================================================

class DialogueAct(Enum):
    OPEN_PRIORITY      = "OPEN_PRIORITY"
    ASSERT_SUPPORT     = "ASSERT_SUPPORT"
    ASSERT_OPPOSITION  = "ASSERT_OPPOSITION"
    ASSERT_AMBIGUOUS   = "ASSERT_AMBIGUOUS"
    ASK_CLARIFICATION  = "ASK_CLARIFICATION"
    ANSWER             = "ANSWER"
    CHALLENGE          = "CHALLENGE"           # explicit pushback at another sim (Stage 5)
    CONCEDE            = "CONCEDE"
    CONDITIONAL_ACCEPT = "CONDITIONAL_ACCEPT"
    COMMIT_VOTE        = "COMMIT_VOTE"
    CONFIRM            = "CONFIRM"
    REJECT_WITH_REASON = "REJECT_WITH_REASON"
    GOODBYE            = "GOODBYE"
    MODERATOR          = "MODERATOR"


@dataclass
class StanceUpdate:
    speaker: str
    option: str                        # A|B|C|D
    stance: str                        # support|oppose|ambiguous|conditional_support|blocker|neutral
    confidence: float = 1.0
    condition: Optional[str] = None


@dataclass
class TurnRecord:
    turn_id: int
    speaker: str
    text: str
    phase: str
    is_moderator: bool

    addressees: list[str]              = field(default_factory=list)
    reply_to: Optional[int]            = None
    is_question: bool                  = False
    answers_question_id: Optional[int] = None

    dialogue_act: DialogueAct          = DialogueAct.ASSERT_AMBIGUOUS
    mentioned_options: list[str]       = field(default_factory=list)
    stance_updates: list[StanceUpdate] = field(default_factory=list)

    # Stage 5 -- challenge/response tracking. challenges_turn_id points at the
    # turn this turn pushes back on; answered_challenge_id points at a prior
    # challenge this turn responds to.
    challenges_turn_id: Optional[int]   = None
    answered_challenge_id: Optional[int] = None

    # Stage 1c -- captured priority from the opening phase (one phrase).
    stated_priority: Optional[str]      = None

    selected_reason: str = ""
    tokens_in: int       = 0
    tokens_out: int      = 0


# =============================================================================
# OptionState + StanceTable
# =============================================================================

@dataclass
class OptionState:
    option_id: str
    text: str

    supporters: set[str]                       = field(default_factory=set)
    opponents: set[str]                        = field(default_factory=set)
    ambiguous: set[str]                        = field(default_factory=set)
    conditional_supporters: dict[str, str]     = field(default_factory=dict)
    hard_blockers: dict[str, str]              = field(default_factory=dict)

    last_mentioned_turn: Optional[int]         = None

    def apply_update(self, update: StanceUpdate) -> None:
        name = update.speaker
        self.supporters.discard(name)
        self.opponents.discard(name)
        self.ambiguous.discard(name)
        self.conditional_supporters.pop(name, None)
        self.hard_blockers.pop(name, None)

        if update.stance == "support":
            self.supporters.add(name)
        elif update.stance == "oppose":
            self.opponents.add(name)
        elif update.stance == "blocker":
            self.opponents.add(name)
            self.hard_blockers[name] = update.condition or "unspecified"
        elif update.stance == "conditional_support":
            self.conditional_supporters[name] = update.condition or ""
        elif update.stance == "ambiguous":
            self.ambiguous.add(name)


class StanceTable:

    def __init__(self) -> None:
        self._current: dict[tuple[str, str], StanceUpdate] = {}
        self._history: list[StanceUpdate] = []

    def current_stance_label(self, speaker: str, option: str) -> str:
        su = self._current.get((speaker, option))
        return su.stance if su else "neutral"

    def apply(self, update: StanceUpdate, option_state: Optional[OptionState] = None) -> None:
        self._current[(update.speaker, update.option)] = update
        self._history.append(update)
        if option_state is not None:
            option_state.apply_update(update)

    def history(self) -> list[StanceUpdate]:
        return list(self._history)

    def current_items(self) -> list[tuple[tuple[str, str], StanceUpdate]]:
        return list(self._current.items())

    def unresolved_blockers(self, option: str) -> dict[str, str]:
        return {
            speaker: su.condition or "unspecified"
            for (speaker, opt), su in self._current.items()
            if opt == option and su.stance == "blocker"
        }


# =============================================================================
# DiscourseGraph -- Ouchi & Tsuboi (2016), extended with challenge tracking.
# =============================================================================

@dataclass
class ChallengeRecord:
    """A pushback aimed at another sim. 'answered' once the target rebuts or
    concedes within `cfg.deliberation.challenge_window_turns`."""
    challenge_turn_id: int
    challenger: str
    target: str
    target_option: Optional[str] = None
    answered_turn_id: Optional[int] = None


@dataclass
class DiscourseGraph:
    pending_questions: dict[int, list[str]] = field(default_factory=dict)
    reply_edges: dict[int, int]             = field(default_factory=dict)
    open_invitations: dict[int, str]        = field(default_factory=dict)
    last_addressed: Optional[str]           = None

    # Stage 5 -- challenges + answered challenges (the deliberation signal).
    challenges: list[ChallengeRecord]       = field(default_factory=list)

    def has_obligation_for(self, name: str) -> bool:
        """Only directed questions create turn-taking obligations.

        Earlier versions treated every open participant question as an
        obligation for some arbitrary non-asker. That created question/answer
        chains and let the wrong person answer implicit questions. Open
        invitations are still recorded for evaluation, but they do not force
        speaker selection.
        """
        return any(name in addressees for addressees in self.pending_questions.values())

    def oldest_pending_addressees(self) -> list[str]:
        if not self.pending_questions:
            return []
        return self.pending_questions[min(self.pending_questions)]

    def resolve_question(self, answering_turn_id: int, speaker: str) -> Optional[int]:
        for q_id, addressees in list(self.pending_questions.items()):
            if speaker in addressees:
                self.reply_edges[answering_turn_id] = q_id
                addressees.remove(speaker)
                if not addressees:
                    del self.pending_questions[q_id]
                else:
                    self.pending_questions[q_id] = addressees
                return q_id
        for q_id in sorted(self.open_invitations):
            asker = self.open_invitations[q_id]
            if speaker != asker:
                self.reply_edges[answering_turn_id] = q_id
                del self.open_invitations[q_id]
                return q_id
        return None

    def add_question(self, turn_id: int, addressees: list[str],
                     asker: Optional[str] = None) -> None:
        if addressees:
            self.pending_questions[turn_id] = list(addressees)
        elif asker:
            self.open_invitations[turn_id] = asker

    # ----- challenge tracking (Stage 5) --------------------------------

    def add_challenge(self, challenge_turn_id: int, challenger: str, target: str,
                      target_option: Optional[str] = None) -> None:
        self.challenges.append(ChallengeRecord(
            challenge_turn_id=challenge_turn_id, challenger=challenger,
            target=target, target_option=target_option,
        ))

    def open_challenge_against(self, name: str) -> Optional[ChallengeRecord]:
        """Most recent unanswered challenge aimed at `name`, within the
        configured response window."""
        window = cfg.deliberation.challenge_window_turns
        for ch in reversed(self.challenges):
            if ch.target != name or ch.answered_turn_id is not None:
                continue
            return ch
        del window  # window enforcement is done by the caller against a turn_id
        return None

    def answer_challenge(self, answering_turn_id: int, speaker: str,
                         current_turn_id: int) -> Optional[ChallengeRecord]:
        """Mark the most recent in-window challenge aimed at `speaker` as
        answered by `answering_turn_id`. Returns the answered record if any."""
        window = cfg.deliberation.challenge_window_turns
        for ch in reversed(self.challenges):
            if ch.target != speaker or ch.answered_turn_id is not None:
                continue
            if current_turn_id - ch.challenge_turn_id > window:
                continue
            ch.answered_turn_id = answering_turn_id
            return ch
        return None

    def answered_challenges(self) -> list[ChallengeRecord]:
        return [c for c in self.challenges if c.answered_turn_id is not None]


# =============================================================================
# ParticipantState -- now with per-sim memory (Stage 6).
# =============================================================================

@dataclass
class ParticipantState:
    name: str

    public_preference: Optional[str]          = None

    turn_count: int                           = 0
    last_spoke_turn: Optional[int]            = None
    participation_debt: float                 = 0.0

    is_true_hard_blocker: bool                = False

    # Stage 1c -- captured from the opening phase (one phrase). Becomes the
    # "what this sim cares about" entry in everyone else's perceived_priorities.
    stated_priority: Optional[str]            = None

    # Deliberation signal: has this sim stated a position with at least one reason?
    position_with_reason_stated: bool         = False

    # Relevance-filtered memory (Park 2023 scaled down). Compact lines, capped
    # per cfg.memory. Replaces feeding raw last-N turns.
    points_made: list[str]                    = field(default_factory=list)

    persona_ref: Optional["Persona"]          = field(default=None, repr=False, compare=False)

    def record_point(self, text: str) -> None:
        """Append a compact summary of a point this sim made (anti-repeat)."""
        text = text.strip()
        if not text:
            return
        self.points_made.append(text)
        cap = cfg.memory.points_made_max
        if len(self.points_made) > cap:
            self.points_made = self.points_made[-cap:]


# =============================================================================
# StructuredState
# =============================================================================

@dataclass
class StructuredState:
    turns: list[TurnRecord]                          = field(default_factory=list)
    turn_id_counter: int                             = 0
    participants: dict[str, ParticipantState]        = field(default_factory=dict)
    options: dict[str, OptionState]                  = field(default_factory=dict)
    stance_table: StanceTable                        = field(default_factory=StanceTable)
    discourse: DiscourseGraph                        = field(default_factory=DiscourseGraph)
    consensus_state: Literal[
        "none", "candidate_emerging", "majority_candidate",
        "conditional_consensus", "full_consensus", "blocked", "failed"
    ] = "none"
    candidate_option: Optional[str]                  = None


# =============================================================================
# Heuristic patterns (compiled once)
# =============================================================================

# Challenge detection: pushback markers + addressing of another sim or a claim.
_CHALLENGE_MARKERS = re.compile(
    r"\b(but\b|however\b|actually\b|disagree\b|i\s+don'?t\s+(?:think|buy|agree)|"
    r"not\s+sure\s+(?:about|that)|that'?s\s+not|won'?t\s+work|"
    r"problem\s+with\b|issue\s+with\b|push\s+back|hard\s+to\s+see)",
    re.I,
)

_CONCESSION_LEADS = re.compile(
    r"\b(i\s+agree|fair\s+enough|fair\s+point|good\s+point|valid\s+point|"
    r"you'?re\s+right|.{0,5}'?s\s+right|that'?s\s+a\s+(?:fair|good)\s+point|"
    r"ok(?:ay)?,?\s+(?:that|fair)|sounds\s+good|works\s+for\s+me|"
    r"i'?m\s+in|on\s+board|true\b|conceded?\b|"
    r"yeah,?\s+(?:that|true|fair|right|agree)|"
    r"granted\b|noted\b)",
    re.I,
)

_REASON_MARKERS = re.compile(
    r"\b(because\b|since\b|the\s+reason|that'?s\s+why|given\s+that|due\s+to|"
    r"i\s+(?:know|saw|did|worked|studied)|in\s+my\s+experience|when\s+i\s+|"
    r"i'?ve\s+(?:seen|done|worked)|"
    r"(?:it|that|this)\s+(?:aligns|fits|matches|maps|helps|addresses|"
    r"covers|handles|lets|allows|enables|gives|offers|supports|tackles)|"
    r"directly\s+(?:addresses|relates|tackles)|"
    r"for\s+(?:my|its|the)\s+(?:goal|purpose|reason)|"
    r"would\s+let\s+(?:me|us)|works\s+for\s+(?:my|the)\b)",
    re.I,
)

_PRIORITY_KEYWORDS = re.compile(
    r"\b(care\s+about|matters?\s+(?:to\s+me|most)|priorit|important|"
    r"hoping\s+(?:for|to)|looking\s+for|want\s+(?:to|something)|need\s+something|worried)\b",
    re.I,
)


def _is_addressing(name: str, text: str, *, is_question: bool, has_challenge: bool) -> bool:
    """True iff `text` ADDRESSES `name` (vocative or directed Q/challenge),
    not merely cites them. See _extract_addressees for the shape catalogue."""
    n = re.escape(name)

    # Citation guards -- these forms NEVER count as addressing.
    # Possessive 's -- "Liam's point", "Liam's argument", "Liam's idea"
    if re.search(rf"\b{n}'s\b", text, re.I):
        return False
    # "I agree with X" / "like X said" / "as X pointed out" / "what X claims/thinks"
    if re.search(
        rf"\b(?:with|like|as|what)\s+{n}\b",
        text, re.I,
    ):
        return False

    # Vocative at start: "Liam, ..." or "Liam: ..."
    if re.match(rf"\s*{n}\s*[,:]", text, re.I):
        return True

    # Vocative at end: ", Liam." or ", Liam?" or ", Liam!"
    if re.search(rf",\s*{n}\s*[.?!]?\s*$", text, re.I):
        return True

    # Direct question containing the bare name (citation forms already ruled out above)
    if is_question and re.search(rf"\b{n}\b", text, re.I):
        return True

    # Directed challenge containing the bare name
    if has_challenge and re.search(rf"\b{n}\b", text, re.I):
        return True

    return False


# =============================================================================
# StateTracker -- deterministic raw-line -> structured update
# =============================================================================

class StateTracker:

    def __init__(self, participant_names: list[str], options: list[str],
                 resolver: OptionResolver) -> None:
        self._participant_names: set[str] = set(participant_names)
        self._all_names: list[str] = list(participant_names)
        self._resolver = resolver
        self._option_map: dict[str, str] = {}

        for opt in options:
            m = re.match(r"^Option\s+([A-D])\b", opt, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                self._option_map[letter] = opt

        self.state = StructuredState(
            participants={n: ParticipantState(name=n) for n in participant_names},
            options={l: OptionState(option_id=l, text=t) for l, t in self._option_map.items()},
        )

    def attach_personas(self, sims: list) -> None:
        for sim in sims:
            ps = self.state.participants.get(sim.name)
            if ps:
                ps.persona_ref = sim.persona

    def update(self, line: str, phase: str, selected_reason: str = "",
               tokens_in: int = 0, tokens_out: int = 0) -> Optional[TurnRecord]:
        if ":" not in line:
            return None

        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        is_moderator = speaker not in self._participant_names

        self.state.turn_id_counter += 1
        turn_id = self.state.turn_id_counter

        addressees = self._extract_addressees(text, speaker, is_moderator)
        is_question = "?" in text
        mentioned_options = self._resolver.options_in(text)
        act = self._estimate_act(text, phase, is_moderator, is_question, mentioned_options,
                                 speaker, addressees)
        answers_id = None if is_moderator else self.state.discourse.resolve_question(turn_id, speaker)

        reply_to = None
        if not is_moderator:
            for q_id, addressed in self.state.discourse.pending_questions.items():
                if speaker in addressed:
                    reply_to = q_id
                    break

        stance_updates = self._extract_stances(text, speaker, act, mentioned_options)

        # Question routing.
        if is_question and addressees:
            self.state.discourse.add_question(turn_id, addressees)
            if not is_moderator:
                self.state.discourse.last_addressed = addressees[0]
        elif is_question and is_moderator:
            self.state.discourse.add_question(turn_id, list(self._participant_names))
        elif addressees and not is_moderator:
            self.state.discourse.last_addressed = addressees[0]
        # Participant open questions are deliberately NOT added as obligations.
        # They are often rhetorical or exploratory. If the question implicitly
        # targets the previous speaker, _extract_addressees() resolves it before
        # this branch. Otherwise speaker selection should remain free.

        # Stage 5 -- challenge / answer detection. Only meaningful between sims.
        challenges_turn_id: Optional[int] = None
        answered_challenge_id: Optional[int] = None
        if not is_moderator:
            if act == DialogueAct.CHALLENGE and addressees:
                target = addressees[0]
                if target in self._participant_names and target != speaker:
                    target_opt = mentioned_options[0] if mentioned_options else None
                    self.state.discourse.add_challenge(turn_id, speaker, target, target_opt)
                    challenges_turn_id = turn_id
            answered = self.state.discourse.answer_challenge(turn_id, speaker, turn_id)
            if answered:
                answered_challenge_id = answered.challenge_turn_id

        # Apply stance updates.
        for su in stance_updates:
            opt_state = self.state.options.get(su.option)
            self.state.stance_table.apply(su, opt_state)
            if opt_state and mentioned_options:
                opt_state.last_mentioned_turn = turn_id

        # Captured priority from opening phase (Stage 1c).
        stated_priority: Optional[str] = None
        if not is_moderator and phase == "opening":
            stated_priority = self._extract_priority(text)
            ps_for_priority = self.state.participants.get(speaker)
            if ps_for_priority and stated_priority and not ps_for_priority.stated_priority:
                ps_for_priority.stated_priority = stated_priority

        # Participant-state bookkeeping.
        if not is_moderator:
            ps = self.state.participants.get(speaker)
            if ps:
                ps.turn_count += 1
                ps.last_spoke_turn = turn_id
                if act == DialogueAct.COMMIT_VOTE:
                    vote = self._resolver.vote_in(text)
                    if vote:
                        ps.public_preference = vote
                # Stage 5 -- did this turn state a position with a reason?
                if not ps.position_with_reason_stated and self._is_position_with_reason(
                    act, text, mentioned_options
                ):
                    ps.position_with_reason_stated = True
                # Stage 6 -- compact memory of the speaker's own substantive points.
                if self._counts_as_substantive_point(act, text):
                    ps.record_point(_summarise_point(text))

                # Participation-debt update.
                n = len(self._all_names)
                if n > 1:
                    for nm, other_ps in self.state.participants.items():
                        if nm == speaker:
                            other_ps.participation_debt -= 1.0
                        else:
                            other_ps.participation_debt += 1.0 / (n - 1)

        record = TurnRecord(
            turn_id=turn_id, speaker=speaker, text=text, phase=phase,
            is_moderator=is_moderator, addressees=addressees, reply_to=reply_to,
            is_question=is_question, answers_question_id=answers_id,
            dialogue_act=act, mentioned_options=mentioned_options,
            stance_updates=stance_updates,
            challenges_turn_id=challenges_turn_id,
            answered_challenge_id=answered_challenge_id,
            stated_priority=stated_priority,
            selected_reason=selected_reason,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
        self.state.turns.append(record)
        return record

    # ── helpers ────────────────────────────────────────────────────────

    def _extract_addressees(self, text: str, speaker: str, is_moderator: bool) -> list[str]:
        """Return names being ADDRESSED (vocative or question-to-them),
        NOT names being merely cited.

        Address shapes that count:
          - Vocative at start:  "Liam, what do you think?"  /  "Liam: yeah"
          - Vocative at end:    "..., Liam."  /  "..., Liam?"
          - Direct question containing the name (and not a citation form
            like 'what Liam said').
          - Addressed challenge (CHALLENGE markers + the name as something
            other than a possessive/citation).

        Shapes treated as citation, NOT address (never set last_addressed):
          - Possessive: "Liam's point", "Liam's argument"
          - Concession-cite: "I agree with Liam", "like Liam said", "as Liam pointed out"
          - Reference-cite: "what Liam thinks", "Liam's experience"
        """
        targets = self._participant_names if is_moderator else (self._participant_names - {speaker})
        if not targets:
            return []

        is_question = "?" in text
        has_challenge = bool(_CHALLENGE_MARKERS.search(text))
        result: list[str] = []

        for name in targets:
            if _is_addressing(name, text, is_question=is_question, has_challenge=has_challenge):
                result.append(name)

        # Implicit addressee: if B asks a question immediately after A introduced
        # a priority/concern and the question repeats that keyword, route the
        # answer back to A. Example: "Julian: eco-friendly spot" -> "Liam: How
        # important is being eco-friendly?" should obligate Julian, not a random
        # third speaker.
        if is_question and not result and not is_moderator:
            implicit = self._implicit_previous_speaker_target(text, speaker)
            if implicit and implicit in targets:
                result.append(implicit)
        return result

    def _implicit_previous_speaker_target(self, question_text: str, speaker: str) -> Optional[str]:
        """Infer target from the immediately previous participant turn.

        This is intentionally conservative: it only fires when the current
        question reuses a non-trivial token from the previous speaker's text.
        It fixes the common "A states priority, B asks how important that is,
        C answers" failure without turning all open questions into obligations.
        """
        previous = None
        for t in reversed(self.state.turns):
            if t.is_moderator or t.speaker == speaker:
                continue
            previous = t
            break
        if previous is None:
            return None

        def toks(x: str) -> set[str]:
            stop = {
                "what", "about", "which", "option", "important", "choice", "really",
                "still", "think", "would", "could", "should", "there", "their", "your",
                "with", "that", "this", "from", "have", "does", "need", "want", "like",
                "nice", "spot", "place", "choice", "decision", "best", "good",
            }
            return {
                re.sub(r"[^a-z0-9]", "", w.lower())
                for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", x)
                if re.sub(r"[^a-z0-9]", "", w.lower()) not in stop
            }

        q = toks(question_text)
        p = toks(previous.text)
        if q and p and (q & p):
            return previous.speaker
        return None

    def _estimate_act(self, text: str, phase: str, is_moderator: bool,
                      is_question: bool, mentioned_options: list[str],
                      speaker: str, addressees: list[str]) -> DialogueAct:
        if is_moderator:
            return DialogueAct.MODERATOR

        lower = text.lower().strip()
        stripped = lower.rstrip(".,!?")

        if phase == "closure":
            return DialogueAct.GOODBYE

        if self._resolver.vote_in(text):
            return DialogueAct.COMMIT_VOTE

        if phase == "confirmation":
            if stripped in {"yes", "yeah", "sure", "ok", "okay", "yep", "fine", "agreed"} \
                    or stripped.startswith("yes "):
                return DialogueAct.CONFIRM
            if stripped in {"no", "nope", "nah"} or lower.startswith("no "):
                return DialogueAct.REJECT_WITH_REASON

        if re.search(r"\bcould\s+(accept|live\s+with|work\s+with)\b.*\bif\b", lower):
            return DialogueAct.CONDITIONAL_ACCEPT
        if re.search(r"\bif\b.{0,60}\b(work|accept|fine|okay|deal)\b", lower) and mentioned_options:
            return DialogueAct.CONDITIONAL_ACCEPT

        # Concession vs challenge: when a turn carries both signals, the one
        # appearing FIRST wins. "Fair point about Option D actually" is a
        # concession with a filler caveat -- the user's intent is conceded.
        # "Sure, that's fair, but Option B is bad" is also concession-led.
        # Only when challenge markers precede a (late) concession do we
        # treat the turn as pushback.
        concession_match = _CONCESSION_LEADS.search(lower)
        challenge_match = _CHALLENGE_MARKERS.search(lower)
        if concession_match and (
            challenge_match is None
            or concession_match.start() < challenge_match.start()
        ):
            return DialogueAct.CONCEDE

        # Stage 5 -- explicit challenge act (addressed pushback at another sim).
        # A directed question that carries challenge markers ("Julian, doesn't
        # Option D...?") is pushback, not a neutral clarification, so the
        # challenge classifier runs BEFORE the generic question branch below.
        if challenge_match and addressees:
            return DialogueAct.CHALLENGE

        if is_question:
            return DialogueAct.ASK_CLARIFICATION

        if challenge_match:
            return DialogueAct.ASSERT_OPPOSITION

        if phase == "opening":
            return DialogueAct.OPEN_PRIORITY
        return DialogueAct.ASSERT_AMBIGUOUS

    def _extract_stances(self, text: str, speaker: str, act: DialogueAct,
                         mentioned_options: list[str]) -> list[StanceUpdate]:
        updates: list[StanceUpdate] = []
        cond = _extract_condition(text)

        if act == DialogueAct.COMMIT_VOTE:
            vote = self._resolver.vote_in(text)
            if vote:
                updates.append(StanceUpdate(speaker, vote, "support", 1.0))

        elif act == DialogueAct.CONFIRM:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "support", 0.9))

        elif act == DialogueAct.REJECT_WITH_REASON:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "blocker", 0.8, cond))

        elif act == DialogueAct.CONDITIONAL_ACCEPT:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "conditional_support", 0.8, cond))

        elif act == DialogueAct.CONCEDE:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "conditional_support", 0.6))

        elif act in (DialogueAct.ASSERT_OPPOSITION, DialogueAct.CHALLENGE):
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "oppose", 0.5))

        elif act == DialogueAct.ASSERT_SUPPORT:
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "support", 0.4))

        elif act in (DialogueAct.ASSERT_AMBIGUOUS, DialogueAct.OPEN_PRIORITY):
            for opt in mentioned_options:
                updates.append(StanceUpdate(speaker, opt, "ambiguous", 0.3))

        return updates

    def _is_position_with_reason(self, act: DialogueAct, text: str,
                                 mentioned_options: list[str]) -> bool:
        """Stage 5 deliberation signal: did this turn state a stance about an
        option AND carry a reason marker (because/since/in my experience/...)?"""
        stance_acts = {
            DialogueAct.COMMIT_VOTE, DialogueAct.ASSERT_SUPPORT,
            DialogueAct.ASSERT_OPPOSITION, DialogueAct.CHALLENGE,
            DialogueAct.CONDITIONAL_ACCEPT, DialogueAct.REJECT_WITH_REASON,
        }
        if act not in stance_acts:
            return False
        if not mentioned_options and act != DialogueAct.COMMIT_VOTE:
            return False
        return bool(_REASON_MARKERS.search(text))

    def _counts_as_substantive_point(self, act: DialogueAct, text: str) -> bool:
        if act in (DialogueAct.OPEN_PRIORITY, DialogueAct.GOODBYE, DialogueAct.MODERATOR,
                   DialogueAct.CONFIRM):
            return False
        return len(text.split()) >= 6

    def _extract_priority(self, text: str) -> Optional[str]:
        """Best-effort: pick the clause after a priority keyword in the opening
        turn ('I care about depth, ...' -> 'depth'). Falls back to the first
        clause when nothing matches."""
        m = _PRIORITY_KEYWORDS.search(text)
        if m:
            tail = text[m.end():].lstrip(" ,:;")
            phrase = re.split(r"[.;!?]", tail, maxsplit=1)[0].strip()
            return _truncate(phrase, cfg.memory.perceived_priorities_max_chars) or None
        # fallback: first sentence-ish, stripped of leading hello
        cleaned = re.sub(r"^\W*(hi|hey|hello|yo)[,!\s]+", "", text, flags=re.I).strip()
        first = re.split(r"[.;!?]", cleaned, maxsplit=1)[0].strip()
        return _truncate(first, cfg.memory.perceived_priorities_max_chars) or None


# =============================================================================
# Helpers
# =============================================================================

def _extract_condition(text: str) -> Optional[str]:
    m = re.search(r"\bif\b(.{5,100})", text.lower())
    return m.group(1).strip()[:100] if m else None


def _truncate(text: str, n: int) -> str:
    t = text.strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _summarise_point(text: str) -> str:
    """Compact one-clause summary for memory (anti-repetition). Strip leading
    addressee names and keep up to 12 words from the first clause."""
    cleaned = re.sub(r"^\s*[A-Z][a-zA-Z]*,\s*", "", text).strip()
    first = re.split(r"[.;!?]", cleaned, maxsplit=1)[0].strip()
    words = first.split()
    return " ".join(words[:12])
