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
        self._possessive_openers = []
        for o in resolver.options:
            if not o.name.strip():
                continue
            # Drop parenthetical annotations like "(2010)" — the model omits them in speech.
            base = re.sub(r"\s*\([^)]*\)", "", o.name).strip()
            seen: set[str] = set()
            for candidate in filter(None, [base, o.short_name.strip()]):
                if candidate.lower() in seen:
                    continue
                seen.add(candidate.lower())
                self._possessive_openers.append(
                    re.compile(rf"^(?:\w+\s+){{0,2}}{re.escape(candidate)}\s*[\x27’’]s\b", re.I)
                )
        # Phrases from option card prose fields (upside/tradeoff/concern/best_for) that
        # are long enough to be distinctive. A turn parroting these verbatim sounds like
        # someone reading the card aloud instead of speaking naturally.
        self._card_phrases: list[str] = []
        for o in resolver.options:
            for field in (o.upside, o.tradeoff, o.concern, o.best_for):
                words = field.strip().split()
                if len(words) >= 4:
                    self._card_phrases.append(field.strip().lower())

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
        self._check_opener_variety(stripped, state, intent, issues)
        self._check_decision_clarity(stripped, intent, move, issues)
        self._check_question_chain(stripped, state, intent, issues)
        self._check_completeness(stripped, issues)
        self._check_unwanted_question(stripped, intent, issues)
        self._check_robotic_phrasing(stripped, issues)
        self._check_collective_voice(stripped, issues)
        self._check_card_reading(stripped, issues)
        self._check_self_narration(stripped, issues)
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
                issues.append(ValidationIssue("DUPLICATE_TURN", "warn", "Near-identical to another speaker's recent turn."))
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
                    issues.append(ValidationIssue("ECHOED_PHRASE", "warn", "Reuses a long phrase from another speaker's recent turn."))
                    break
        rt = state.runtimes[intent.speaker_id]
        for prev in rt.already_said[-int(cfg.validation.repeated_start_window):]:
            score = jaccard_text(text, prev)
            if score >= float(cfg.validation.own_similarity_threshold):
                issues.append(ValidationIssue("SELF_REPETITION", "warn", f"Too similar to own recent turn: {score:.2f}"))
                break
        recent = [t for t in reversed(state.turns) if t.speaker_id != "moderator"][: int(cfg.validation.repeated_start_window)]
        first = _first_words(text, int(cfg.validation.repeated_start_word_count))
        for turn in recent:
            score = jaccard_text(text, turn.text)
            if score >= float(cfg.validation.group_similarity_threshold):
                if "?" in text and "?" in turn.text:
                    # Echoing a question is strictly wrong — repair so the model hedges instead
                    issues.append(ValidationIssue("QUESTION_ECHO", "repair", f"Re-asks a question already asked: {score:.2f}"))
                else:
                    issues.append(ValidationIssue("GROUP_REPETITION", "warn", f"Similar to a recent participant turn: {score:.2f}"))
                break
            if first and first == _first_words(turn.text, int(cfg.validation.repeated_start_word_count)):
                issues.append(ValidationIssue("REPEATED_START", "repair", "Starts like a recent turn."))
                break

    def _check_opener_variety(self, text: str, state: DialogueState, intent: MoveIntent, issues: list[ValidationIssue]) -> None:
        # The dominant naturalness failure across runs: turn after turn opens with a self-anchored
        # stance ("I'm drawn to X", "I worry that Y", "We need Z", "Our budget..."), so the chat
        # reads as parallel monologues instead of people talking to each other. A few are fine; once
        # several in a row have led that way, force the next one to open some other way (a reaction,
        # the option/detail, a question, the prior point). Topic-agnostic — pronoun only, no scenario
        # words. Commitment turns are exempt: a vote/accept/reject naturally states "I..." and the
        # anti-chorus guidance + echo guard already handle the closing round.
        if intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
            return
        if not _STANCE_OPENER.match(text):
            return
        limit = int(cfg.validation.max_consecutive_first_person_openers)
        streak = 0
        for turn in reversed(state.turns):
            if turn.speaker_id == "moderator":
                continue
            if _STANCE_OPENER.match(turn.text):
                streak += 1
            else:
                break
        if streak >= limit:
            issues.append(ValidationIssue("REPETITIVE_OPENER", "warn", "Several turns in a row open with 'I'/'we' — vary the opening."))

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
        # ANSWER turns must answer or hedge, not re-ask the question.
        statement_only = {ActType.VOTE, ActType.ACCEPT, ActType.REJECT, ActType.OPENING, ActType.ANSWER}
        if "?" in text and intent.act in statement_only:
            issues.append(ValidationIssue("UNWANTED_QUESTION", "repair", "This move should be a statement, not a question."))

    def _check_robotic_phrasing(self, text: str, issues: list[ValidationIssue]) -> None:
        # The prompt bans these formulaic frames, but the model still reaches for them often
        # enough to read as a template. A deterministic backstop forces the worst ones to be
        # rewritten instead of leaking into the chat. Topic-agnostic: no scenario words.
        # "Considering..." as a turn opener is escalated to repair (same class as POSSESSIVE_SUBJECT,
        # REPEATED_START); other robotic templates remain warn-only.
        if _CONSIDERING_OPENER.match(text):
            issues.append(ValidationIssue("ROBOTIC_TEMPLATE", "repair", "Opens with 'Considering' — start with the point, a reaction, or a question."))
        elif any(p.search(text) for p in _ROBOTIC_TEMPLATES):
            issues.append(ValidationIssue("ROBOTIC_TEMPLATE", "warn", "Uses a banned formulaic template."))
        if any(p.match(text) for p in self._possessive_openers):
            issues.append(ValidationIssue("POSSESSIVE_SUBJECT", "repair", "Opens with an option's possessive ('X's ...')."))

    def _check_collective_voice(self, text: str, issues: list[ValidationIssue]) -> None:
        # "We consider...", "We prioritize...", "We're drawn to...", "We can appreciate..." —
        # an individual stating a *private* view as the committee "we" was the most consistent
        # robotic tell across runs (ANALYSIS #6). The 'we'-led case is corrected deterministically
        # upstream (fix_collective_voice in clean_generated), so this is a warn-level backstop:
        # it surfaces anything that slips through (e.g. an 'our group prioritizes ...' form) in
        # the logs without spending an LLM repair on a flaky rewrite.
        if _COLLECTIVE_VOICE.search(text):
            issues.append(ValidationIssue("COLLECTIVE_VOICE", "warn", "States a personal view as the committee 'we'."))

    def _check_card_reading(self, text: str, issues: list[ValidationIssue]) -> None:
        low = text.lower()
        for phrase in self._card_phrases:
            if phrase in low:
                issues.append(ValidationIssue("CARD_READING", "warn", "Parrots an option card's description verbatim."))
                return

    def _check_self_narration(self, text: str, issues: list[ValidationIssue]) -> None:
        if _SELF_NARRATION.match(text):
            issues.append(ValidationIssue("SELF_NARRATION", "warn", "Narrates own thinking instead of just saying it."))

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
            issues.append(ValidationIssue("QUESTION_CHAIN", "warn", "Too many consecutive questions."))

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

# A line-initial self-anchored opener: first person singular ("I", "I'm", "I'd", "I've",
# "I'll") OR the committee plural ("we", "we're", "we've", "we'd", "we'll", "our"). Used to
# detect a run of consecutive stance-led turns ("I'm drawn to...", "I worry...", "We need...",
# "Our budget...") so the chat doesn't collapse into parallel monologues — the plural is
# included because, once "I" is discouraged, the model otherwise just swaps in a "we"-wall.
# Pronoun only — stays topic-agnostic.
_STANCE_OPENER = re.compile(r"\s*(?:i(?:['’](?:m|d|ve|ll))?|we(?:['’](?:re|ve|d|ll))?|our)\b", re.I)

# "Considering..." as a turn opener — escalated to repair-level in _check_robotic_phrasing.
# Kept separate so it can fire at repair severity while other _ROBOTIC_TEMPLATES stay warn-only.
_CONSIDERING_OPENER = re.compile(r"^\s*considering\b", re.I)

# Formulaic frames the prompt forbids but the model still emits. Kept small and word-based
# (no option/topic words) so it generalises across every scenario. Warn-level: logged but not
# forced to rewrite — except _CONSIDERING_OPENER above which is repair-level.
_ROBOTIC_TEMPLATES = [
    re.compile(r"\bout ?weigh", re.I),
    re.compile(r"\bmakes? me (?:think|consider|reconsider|wonder|real[iz]se)\b", re.I),
    re.compile(r"\b(?:point|points|concern|concerns|argument|idea)\b.{0,75}\b(?:valid|fair|well[- ]taken|compelling|spot[- ]on|resonate)\b", re.I),
    re.compile(r"\bgiven the discussion\b", re.I),
    re.compile(r"\b(?:seems?|feels?|sounds?)\s+like\s+the\s+(?:best|right)\s+(?:fit|choice|option|pick|one)\b", re.I),
    re.compile(r"\b(?:seems?|feels?|sounds?)\s+like\s+a\s+(?:good|great|solid|natural|obvious|perfect|ideal|clear)\s+fit\b", re.I),
    re.compile(r"\bstill\s+(?:beats?|wins?|comes?\s+out\s+(?:ahead|on\s+top)|edges?\s+out)\b", re.I),
    re.compile(r"\bedges?\s+ahead\b", re.I),
    re.compile(r"\bstill\s+pick\b", re.I),
    re.compile(r"^\s*since \w+ (?:mentioned|said|raised|brought up|expressed)\b", re.I),
    re.compile(r"\bwins? me over\b", re.I),
    re.compile(r"\bseals? the deal\b", re.I),
    re.compile(r"\b(?:is|seems?\s+like)\s+(?:the\s+)?way\s+to\s+go\b", re.I),
    re.compile(r"\b(?:is|are)\s+a\s+(?:must|big draw|major draw|huge draw|key factor|top priority)\b", re.I),
    re.compile(r"\b(?:is|are)\s+(?:key|crucial|essential|paramount)\s+(?:for|to)\s+me\b", re.I),
    re.compile(r"\bdo they offer\b", re.I),
    re.compile(r"\bwhy that matters\b", re.I),
]

# Line-initial "we/our + cognition/preference verb": an individual hiding a private stance
# behind the committee "we". Genuine group actions ("we should...", "we could...", "we need to
# decide") are not touched. The {0,18} window lets contractions and a modal sit between.
_COLLECTIVE_VOICE = re.compile(
    r"^\s*(?:we|our)\b[^.?!]{0,18}?\b(?:consider|prioriti[sz]e|value|appreciate|"
    r"believe|think|feel|prefer|favou?r|drawn|love|enjoy|"
    r"benefit|deserve|evaluate|weigh|assess|factor)\b",
    re.I,
)

# The leading "we[ aux]" of a committee-voiced cognition statement, used to rewrite it to the
# first person. Only the pronoun (+ its auxiliary) is matched; the rest of the line is kept
# verbatim, so this corrects register without touching content.
_WE_LEAD = re.compile(r"^(\s*)[Ww]e(’re|'re|’ve|'ve|’ll|'ll|’d|'d| are| have| will)?\b")
_WE_AUX_TO_I = {
    "'re": "I'm", "’re": "I'm", " are": "I am",
    "'ve": "I've", "’ve": "I've", " have": "I have",
    "'ll": "I'll", "’ll": "I'll", " will": "I will",
    "'d": "I'd", "’d": "I'd",
}


def fix_collective_voice(text: str) -> str:
    """Deterministically rewrite a 'we'-led personal-opinion line to the first person
    ('We prioritize comfort' -> 'I prioritize comfort', "We're drawn to X" -> "I'm drawn to X").

    Only fires on the same line-initial we+cognition-verb pattern the validator flags, and only
    for the 'we'-led form (an 'our group ...' phrasing is genuinely ambiguous and left alone).
    The LLM reliably ignores a "say 'I' not 'we'" repair hint, so this fixes the single most
    common robotic tell in code — no extra LLM call, content fully preserved."""
    if not _COLLECTIVE_VOICE.search(text):
        return text
    m = _WE_LEAD.match(text)
    if not m:
        return text  # 'our ...' or an odd form -> leave for the warn-level backstop
    indent, aux = m.group(1), (m.group(2) or "")
    head = _WE_AUX_TO_I.get(aux.lower(), "I")
    return f"{indent}{head}{text[m.end():]}"


_STOCK_PHRASE_REWRITES = [
    (re.compile(r"\b(?:is|are)\s+a\s+must\s+for\s+me\b", re.I), "matters to me"),
    (re.compile(r"\b(?:is|are)\s+a\s+(?:big|major|huge)\s+draw(?:\s+for\s+me)?\b", re.I), "appeals to me"),
    (re.compile(r"\b(?:is|are)\s+(?:key|crucial|essential)\s+(?:for|to)\s+me\b", re.I), "matters to me"),
    (re.compile(r"\b(?:is|are)\s+a\s+(?:key|top)\s+(?:factor|priority)(?:\s+for\s+me)?\b", re.I), "matters to me"),
]


def fix_stock_phrases(text: str) -> str:
    for pattern, replacement in _STOCK_PHRASE_REWRITES:
        text = pattern.sub(replacement, text)
    return text


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


# "I should consider/prioritize/think about/look at/evaluate" — self-narration where the
# speaker describes their own cognitive process instead of just stating their view. Nobody
# talks like this in real conversation; it's an LLM tell.
_SELF_NARRATION = re.compile(
    r"^\s*I\s+(?:should|need to|have to|ought to|want to|must)\s+"
    r"(?:consider|prioriti[sz]e|think about|look at|evaluate|weigh|assess|factor in|focus on|take into account)\b",
    re.I,
)

# Words a complete sentence cannot end on; a turn ending here (without punctuation) is cut off.
_DANGLING_END = {
    "to", "of", "for", "and", "but", "or", "with", "the", "a", "an", "than", "vs", "versus",
    "compared", "between", "while", "because", "so", "that", "is", "are", "its", "their",
    "our", "my", "your", "his", "her", "in", "on", "at", "as", "if", "over", "from",
}


def _normalise_number(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def _first_words(text: str, n: int) -> str:
    words = re.findall(r"[\w’’]+", text.lower())
    return " ".join(words[:n])


# ---------------------------------------------------------------------------
# Discourse-frame classification (O1)
# ---------------------------------------------------------------------------

_FRAME_PATTERNS: dict[str, re.Pattern] = {
    "echo_pivot": re.compile(
        r"^\s*(?:\w[\w\s]{0,30}?)\s+"
        r"(?:is|are|sounds?|seems?|feels?)\s+"
        r"(?:really\s+|definitely\s+|certainly\s+|absolutely\s+)?"
        r"(?:great|good|nice|fair|true|key|important|valid|solid|appealing|exciting|"
        r"a\s+great\s+point|a\s+good\s+point|a\s+fair\s+point)"
        r"(?:\s+(?:too|though|and\s+all))?"
        r"\s*(?:,|but\b)", re.I,
    ),
    "agreement_preface": re.compile(
        r"\b(?:(?:that(?:[\x27’’]s|\s+is))\s+a\s+(?:fair|good|great|valid|solid)\s+point|"
        r"you(?:[\x27’’]re| are)\s+right|"
        r"(?:fair|good|great)\s+point|"
        r"i\s+agree\s+(?:with|that)|"
        r"absolutely|exactly)\b", re.I,
    ),
    "concession_preface": re.compile(
        r"\b(?:i\s+(?:can\s+see|understand|hear\s+you|get\s+(?:that|where))|"
        r"you\s+(?:make|raise)\s+a\s+(?:good|fair|valid|strong|great|solid|compelling)\s+(?:point|case|argument))\b", re.I,
    ),
    "option_endorsement": re.compile(
        r"\b(?:seals?\s+the\s+deal|wins?\s+me\s+over|"
        r"(?:is|seems?|feels?)\s+like\s+(?:the\s+)?(?:way\s+to\s+go|(?:best|right|top|clear)\s+(?:choice|pick|option))|"
        r"(?:is\s+)?a\s+major\s+draw|"
        r"(?:is|seems?)\s+(?:like\s+)?(?:a\s+)?(?:no[- ]brainer|slam[- ]dunk)|"
        r"can’?t\s+(?:go|argue)\s+wrong\s+with)\b", re.I,
    ),
    "given_the_x": re.compile(
        r"^\s*(?:given\s+(?:the|our|that)|considering\s+(?:the|our|that)|"
        r"with\s+(?:the|our)\s+\w+\s+(?:in\s+mind|constraint|requirement))\b", re.I,
    ),
    "compromise_acceptance": re.compile(
        r"\b(?:(?:i\s+)?can\s+live\s+with|works?\s+for\s+me|"
        r"(?:i’?m|i\s+am)\s+on\s+board|"
        r"(?:i’?m|i\s+am)\s+(?:fine|okay|ok)\s+with)\b", re.I,
    ),
    "what_if_opener": re.compile(
        r"^\s*(?:what\s+if\s+(?:we|they|everyone|the\s+group)|how\s+about\s+we)\b", re.I,
    ),
    "wait_what_about": re.compile(
        r"^\s*wait[,.]?\s+(?:but\s+)?what\s+about\b", re.I,
    ),
}

_FRAME_WINDOW = 4


def classify_discourse_frames(text: str) -> list[str]:
    return [name for name, pat in _FRAME_PATTERNS.items() if pat.search(text)]


def recent_frame_hint(recent_frames: list[str], speaker_frames: list[str]) -> str:
    """If a discourse frame was used recently (by anyone) or by this speaker,
    return a short prompt hint asking for variety. Empty string if no issue."""
    from collections import Counter
    recent = Counter(recent_frames[-_FRAME_WINDOW * 2:])
    speaker = Counter(speaker_frames[-_FRAME_WINDOW:])
    overused: set[str] = set()
    for frame, count in recent.items():
        if frame == "echo_pivot" and count >= 1:
            overused.add(frame)
        elif count >= 2:
            overused.add(frame)
    for frame, count in speaker.items():
        if count >= 1:
            overused.add(frame)
    if not overused:
        return ""
    labels = {
        "echo_pivot": "echo-pivot openers (‘X is great, but’ / ‘X is key, but’) — skip the recap and lead with your own thought",
        "agreement_preface": "agreeing prefaces (‘good point’, ‘fair point’, ‘you’re right’)",
        "concession_preface": "concession prefaces (‘I can see’, ‘I understand’)",
        "option_endorsement": "stock endorsement phrases (‘seals the deal’, ‘way to go’, ‘best choice’)",
        "given_the_x": "’given the...’ / ‘considering the...’ openers",
        "compromise_acceptance": "acceptance clichés (‘works for me’, ‘can live with’)",
        "what_if_opener": "’What if we...’ / ‘How about we...’ openers — state the fix directly instead",
        "wait_what_about": "’Wait, what about...’ openers — ask or react without the ‘wait’ setup",
    }
    avoid = "; ".join(labels.get(f, f) for f in sorted(overused))
    return f"Avoid: {avoid}. Vary your phrasing — react directly instead."


# ---------------------------------------------------------------------------
# Semantic claim-slot tracking (O3)
# ---------------------------------------------------------------------------

_CLAIM_SLOT_PATTERNS: dict[str, re.Pattern] = {
    "cost": re.compile(r"\b(?:cost|price|budget|cheap|expensive|afford|spend|money|\$\d|per\s+person|dollars?|euros?)\b", re.I),
    "time": re.compile(r"\b(?:time|duration|playtime|hours?|minutes?|quick|fast|slow|long|short|speed)\b", re.I),
    "comfort": re.compile(r"\b(?:comfort|cozy|intimate|ambiance|atmosphere|vibe|setting|room|seat|space|upscale|relaxed)\b", re.I),
    "flexibility": re.compile(r"\b(?:flexib\w*|refund\w*|cancel\w*|adaptab\w*|versatil\w*|variety|wide\s+(?:range|variety|selection))\b", re.I),
    "quality": re.compile(r"\b(?:quality|premium|high[- ]end|gourmet|authentic|fresh|craft|artisan)\b", re.I),
    "effort": re.compile(r"\b(?:effort|difficulty|hard|easy|complex|simple|learn|beginner|casual|demanding)\b", re.I),
    "distance": re.compile(r"\b(?:distance|far|close|near|walk|drive|travel|commute|location|convenient)\b", re.I),
    "group_fit": re.compile(r"\b(?:group|team|everyone|together|player|social|crowd|people|friend|party)\b", re.I),
    "novelty": re.compile(r"\b(?:novel|new|unique|different|exciting|fun|creative|interesting|boring|fresh)\b", re.I),
    "risk": re.compile(r"\b(?:risk|safe|danger|worry|concern|reliable|trust|proven|unknown)\b", re.I),
}


def classify_claim_slots(text: str) -> list[str]:
    return [slot for slot, pat in _CLAIM_SLOT_PATTERNS.items() if pat.search(text)]


def covered_slots_hint(option_id: str, covered: set[str]) -> str:
    """Return a nudge toward a new argument dimension when 3+ claim slots are already
    covered for this option — prevents multiple speakers repeating the same substance
    in different words (cost/quality/comfort chorus)."""
    if len(covered) < 3:
        return ""
    covered_names = ", ".join(sorted(covered)[:4])
    uncovered = [s for s in _CLAIM_SLOT_PATTERNS if s not in covered]
    if uncovered:
        suggestions = ", ".join(uncovered[:3])
        return f"The group already argued {covered_names} for this option. Try a new angle: {suggestions}."
    return f"The group already argued {covered_names} for this option. Add a new detail or respond to an objection instead of restating."
