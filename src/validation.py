"""Deterministic validation for generated participant messages."""

from __future__ import annotations

import re

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, ValidationIssue, ValidationResult
from parsing import OptionResolver, TurnMove
from utils import extract_numbers, jaccard_text


class MessageValidator:
    def __init__(self, resolver: OptionResolver, participant_names: dict[str, str]) -> None:
        self.resolver = resolver
        self.participant_names = participant_names
        # Whole option/participant-name phrases are masked in the echo guard so two turns
        # that merely name the same option/person don't read as a lifted phrase. Masking
        # the phrase (not each word) keeps a common word that happens to sit inside a name
        # (e.g. "research" in "National Cancer Research") catchable when it stands alone.
        phrases = [o.name.lower() for o in resolver.options]
        phrases += [n.lower() for n in participant_names.values()]
        self._mask_phrases = sorted({p for p in phrases if p.strip()}, key=lambda s: len(s.split()), reverse=True)
        # Line-initial "<Option>'s <feature>" is the single most robotic, repetitive tic the
        # model falls into (it opened ~40% of turns in large-group runs). Catch it from the
        # actual option names so the check stays topic-agnostic.
        self._possessive_openers = [
            re.compile(rf"^(?:\w+\s+){{0,2}}{re.escape(o.name)}\s*['’]s\b", re.I)
            for o in resolver.options if o.name.strip()
        ]

    def validate(self, text: str, state: DialogueState, intent: MoveIntent, move: TurnMove) -> ValidationResult:
        if not bool(cfg.validation.enabled):
            return ValidationResult(ok=True)
        issues: list[ValidationIssue] = []
        stripped = text.strip()
        if not stripped:
            return ValidationResult(False, [ValidationIssue("EMPTY", "fatal", "Generated message is empty.")])
        self._check_structure(stripped, issues)
        self._check_option_refs(stripped, issues)
        self._check_grounding_numbers(stripped, intent, issues)
        self._check_repetition(stripped, state, intent, issues)
        self._check_decision_clarity(stripped, intent, move, issues)
        self._check_question_chain(stripped, state, intent, issues)
        self._check_completeness(stripped, issues)
        self._check_unwanted_question(stripped, intent, issues)
        self._check_robotic_phrasing(stripped, issues)
        ok = not any(issue.severity in {"repair", "fatal"} for issue in issues)
        return ValidationResult(ok=ok, issues=issues)

    def _check_structure(self, text: str, issues: list[ValidationIssue]) -> None:
        pattern = r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß .'-]{1,30}:"
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and any(re.search(pattern, line.strip()) for line in lines[1:]):
            issues.append(ValidationIssue("MULTI_TURN_OUTPUT", "fatal", "Message appears to contain multiple turns."))
        if re.search(pattern, text):
            issues.append(ValidationIssue("SPEAKER_PREFIX", "repair", "Message starts with a speaker prefix."))

    def _check_option_refs(self, text: str, issues: list[ValidationIssue]) -> None:
        invalid = self.resolver.invalid_option_refs(text)
        if invalid:
            issues.append(ValidationIssue("INVALID_OPTION_REFERENCE", "fatal", f"Invalid options referenced: {invalid}"))

    def _check_grounding_numbers(self, text: str, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        numbers = extract_numbers(text)
        if not numbers:
            return
        refs = self.resolver.ids_in_text(text) or intent.option_focus
        if not refs:
            if len(numbers) > int(cfg.validation.max_new_numbers_without_source):
                issues.append(ValidationIssue("UNGROUNDED_NUMERIC_FACT", "repair", "Numeric facts appear without option grounding."))
            return
        source = "\n".join(self.resolver.option_text(ref) for ref in refs if ref in self.resolver.by_id).lower()
        invented = [n for n in numbers if _normalise_number(n) not in _normalise_number(source)]
        if invented:
            issues.append(ValidationIssue("INVENTED_OPTION_ATTRIBUTE", "repair", f"Numbers not found in option source: {invented}"))

    def _check_repetition(self, text: str, state: DialogueState, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        # A near-verbatim copy of another speaker's recent line is never natural — flag it
        # even for confirmations (two people shouldn't say the exact same sentence).
        others = [t for t in reversed(state.turns) if t.speaker_id not in {"moderator", intent.speaker_id}]
        window = int(cfg.validation.repeated_start_window)
        for turn in others[:window]:
            if jaccard_text(text, turn.text) >= float(cfg.validation.duplicate_threshold):
                issues.append(ValidationIssue("DUPLICATE_TURN", "repair", "Near-identical to another speaker's recent turn."))
                break
        # Echo guard: catch a long verbatim phrase lifted from another speaker even when an
        # added clause keeps overall jaccard low (the vote-round "...feels like our most
        # practical choice overall" echo). Runs through accepts/votes too, where chorusing
        # is worst; option/participant-name words are masked so naming the same pick is fine.
        masked = self._masked_tokens(text)
        min_run = int(cfg.validation.max_shared_phrase_words)
        if not any(i.code == "DUPLICATE_TURN" for i in issues):
            for turn in others[:window]:
                if _longest_run(masked, self._masked_tokens(turn.text)) >= min_run:
                    issues.append(ValidationIssue("ECHOED_PHRASE", "repair", "Reuses a long phrase from another speaker's recent turn."))
                    break
        # Confirmations are otherwise naturally similar, so we don't flag ordinary repetition there.
        if intent.act in {ActType.ACCEPT, ActType.REJECT}:
            return
        rt = state.runtimes[intent.speaker_id]
        for prev in rt.already_said[-int(cfg.validation.repeated_start_window):]:
            score = jaccard_text(text, prev)
            if score >= float(cfg.validation.own_similarity_threshold):
                issues.append(ValidationIssue("SELF_REPETITION", "repair", f"Too similar to own recent turn: {score:.2f}"))
                break
        recent = [t for t in reversed(state.turns) if t.speaker_id != "moderator"][: int(cfg.validation.repeated_start_window)]
        first = _first_words(text, int(cfg.validation.repeated_start_word_count))
        for turn in recent:
            score = jaccard_text(text, turn.text)
            if score >= float(cfg.validation.group_similarity_threshold):
                issues.append(ValidationIssue("GROUP_REPETITION", "warn", f"Similar to a recent participant turn: {score:.2f}"))
                break
            if first and first == _first_words(turn.text, int(cfg.validation.repeated_start_word_count)):
                issues.append(ValidationIssue("REPEATED_START", "warn", "Starts like a recent turn."))
                break

    def _check_decision_clarity(self, text: str, intent: MoveIntent, move: TurnMove, issues: list[ValidationIssue]) -> None:
        # Only flag a decision move when the model emitted a trailer that contradicts
        # the routed move. With no trailer we trust the routed intent (see parsing).
        if move.present:
            if intent.act == ActType.VOTE and move.stance != "vote":
                issues.append(ValidationIssue("UNCLEAR_VOTE", "repair", "Vote move did not report a vote."))
            if intent.act == ActType.ACCEPT and move.stance != "accept":
                issues.append(ValidationIssue("UNCLEAR_ACCEPT", "repair", "Accept move did not report acceptance."))
            if intent.act == ActType.REJECT and move.stance not in {"reject", "object"}:
                issues.append(ValidationIssue("UNCLEAR_REJECT", "repair", "Reject move did not report a rejection or objection."))
        if intent.act == ActType.ACCEPT and "?" in text:
            issues.append(ValidationIssue("QUESTION_IN_CONFIRMATION", "repair", "Confirmation should be an answer, not a new question."))

    def _check_completeness(self, text: str, issues: list[ValidationIssue]) -> None:
        # Catch turns that trail off mid-thought, e.g. "...compare to Kyoto Delight's"
        # or "let's go with the". A casual line may skip a final period, so we only flag
        # when it ALSO ends on a possessive or a word a sentence cannot end on.
        words = re.findall(r"[\w'’]+", text)
        if not words:
            return
        ends_punct = text.rstrip()[-1:] in ".!?…\"'’)"
        last = words[-1].lower()
        dangling = last.endswith("'s") or last.endswith("’s") or last in _DANGLING_END
        if dangling and not ends_punct:
            issues.append(ValidationIssue("INCOMPLETE_TURN", "repair", "Message appears cut off mid-thought."))

    def _check_unwanted_question(self, text: str, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        # Decision and opening turns must commit or state, not deflect into a question.
        # Discussion moves may carry an occasional rhetorical or open question that invites
        # others in — the question-chain check still stops it becoming an interrogation.
        statement_only = {ActType.VOTE, ActType.ACCEPT, ActType.REJECT, ActType.OPENING}
        if "?" in text and intent.act in statement_only:
            issues.append(ValidationIssue("UNWANTED_QUESTION", "repair", "This move should be a statement, not a question."))

    def _check_robotic_phrasing(self, text: str, issues: list[ValidationIssue]) -> None:
        # The prompt bans these formulaic frames, but the model still reaches for them often
        # enough to read as a template. A deterministic backstop forces the worst ones to be
        # rewritten instead of leaking into the chat. Topic-agnostic: no scenario words.
        if any(p.search(text) for p in _ROBOTIC_TEMPLATES):
            issues.append(ValidationIssue("ROBOTIC_TEMPLATE", "repair", "Uses a banned formulaic template."))
        if any(p.match(text) for p in self._possessive_openers):
            issues.append(ValidationIssue("POSSESSIVE_SUBJECT", "repair", "Opens with an option's possessive ('X's ...')."))

    def _check_question_chain(self, text: str, state: DialogueState, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        if "?" not in text:
            return
        recent_questions = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if "?" in turn.text:
                recent_questions += 1
            else:
                break
        if recent_questions >= int(cfg.validation.max_question_chain) and intent.act != ActType.ANSWER:
            issues.append(ValidationIssue("QUESTION_CHAIN", "repair", "Too many consecutive questions."))

    def _masked_tokens(self, text: str) -> list[str]:
        # Replace whole name phrases with a sentinel that never matches, then tokenise, so
        # the echo guard measures shared *phrasing* rather than a shared option/person name.
        low = text.lower()
        for phrase in self._mask_phrases:
            low = re.sub(rf"\b{re.escape(phrase)}\b", f" {_MASK} ", low)
        toks: list[str] = []
        for raw in re.findall(rf"{re.escape(_MASK)}|[\w'’]+", low):
            if raw == _MASK:
                toks.append(_MASK)
                continue
            base = raw[:-2] if raw.endswith(("'s", "’s")) else raw
            if base:
                toks.append(base)
        return toks


# Sentinel for masked name words in the echo guard; cannot occur in real tokenised text.
_MASK = "\x00"

# Formulaic frames the prompt forbids but the model still emits. Kept small and word-based
# (no option/topic words) so it generalises across every scenario. Repair-level: the worst
# offenders get rewritten rather than reaching the transcript.
_ROBOTIC_TEMPLATES = [
    re.compile(r"\bout ?weigh", re.I),                                            # "X outweighs Y"
    re.compile(r"\bmakes? me (?:think|consider|reconsider|wonder|real[iz]se)\b", re.I),
    re.compile(r"\b(?:point|points|concern|concerns|argument|idea)\b.{0,75}\b(?:valid|fair|well[- ]taken|compelling|spot[- ]on|resonate)\b", re.I),
    re.compile(r"\bgiven the discussion\b", re.I),
    re.compile(r"^\s*considering\b", re.I),                                       # "Considering ..." opener
    re.compile(r"\b(?:seems?|feels?|sounds?)\s+like\s+the\s+(?:best|right)\s+(?:fit|choice|option|pick|one)\b", re.I),  # stock vote closer
    re.compile(r"^\s*since \w+ (?:mentioned|said|raised|brought up|expressed)\b", re.I),  # meeting-minutes recap
]


def _longest_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous run of tokens shared by a and b. Masked tokens
    never match, so a run cannot be built from — or span across — a name word."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        if ai != _MASK:
            for j in range(1, len(b) + 1):
                if ai == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
        prev = cur
    return best


# Words a complete sentence cannot end on; a turn ending here (without punctuation) is cut off.
_DANGLING_END = {
    "to", "of", "for", "and", "but", "or", "with", "the", "a", "an", "than", "vs", "versus",
    "compared", "between", "while", "because", "so", "that", "is", "are", "its", "their",
    "our", "my", "your", "his", "her", "in", "on", "at", "as", "if", "over", "from",
}


def _normalise_number(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def _first_words(text: str, n: int) -> str:
    words = re.findall(r"[\w'’]+", text.lower())
    return " ".join(words[:n])
