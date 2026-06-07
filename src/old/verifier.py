"""
verifier.py
-----------
Deterministic verification layer. Runs after each LLM-generated message to
catch common generation failures without an LLM call.

Checks for participant turns:
  - invalid option reference (Option E, etc.)
  - option denial (claiming a listed option doesn't exist)
  - invented option attributes (numbers/prices near an option mention)
  - self-repetition (too similar to own recent turns)
  - missing explicit vote during narrowing
  - unclear confirmation during confirmation phase
  - name prefix (speaker prefixing their own name)

No LLM calls here -- fast and deterministic only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from config_loader import cfg
from utils import OptionResolver

if TYPE_CHECKING:
    from state import ParticipantState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VerificationIssue:
    code: str
    severity: str   # "repair" | "warn" | "fatal"
    message: str


@dataclass
class VerificationResult:
    ok: bool
    issues: list[VerificationIssue] = field(default_factory=list)
    needs_repair: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [{"code": i.code, "severity": i.severity, "message": i.message}
                       for i in self.issues],
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
        }


# ---------------------------------------------------------------------------
# Config helpers (graceful defaults when section is absent)
# ---------------------------------------------------------------------------

def _vcfg(attr: str, default: Any) -> Any:
    try:
        sec = getattr(cfg, "verification", None)
        return getattr(sec, attr, default) if sec is not None else default
    except AttributeError:
        return default


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase words longer than 2 chars, stripped of punctuation."""
    return {
        re.sub(r"[^\w]", "", w).lower()
        for w in text.split()
        if len(w) > 2
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Option denial: "option A is not available / doesn't exist / not on the table"
_OPTION_DENIAL = re.compile(
    r"\bOption\s+[A-Da-d]\b.{0,40}\b(?:is\s+)?(?:not\s+(?:an?\s+option|available|listed|"
    r"on\s+the\s+table|valid|possible)|isn'?t\s+(?:an?\s+option|available|listed|valid)|"
    r"doesn'?t\s+exist|was\s+(?:removed|dropped)|no\s+longer(?:\s+an?\s+option)?)",
    re.I,
)

# Confirmation signals
_YES_PAT = re.compile(
    r"\b(?:yes|yeah|yep|yep|sure|ok|okay|fine|agreed|works?\s+for\s+me|"
    r"i'?m\s+in|sounds?\s+(?:good|fine)|that\s+works?|go\s+ahead|"
    r"i\s+can\s+live\s+with\s+that|no\s+objection|i'?m\s+good\s+with\s+(?:it|that)|"
    r"i\s+agree|all\s+good|good\s+(?:with|for)\s+me|happy\s+with\s+that|"
    r"works?\s+(?:fine|well)|that'?s\s+(?:fine|good|ok|okay))\b",
    re.I,
)
_NO_PAT = re.compile(
    r"^(?:no\b|nope\b|nah\b|not\s+(?:really|quite|yet|sure)\b|"
    r"i\s+(?:disagree|can'?t|won'?t)\b|still\s+not\b|don'?t\s+agree\b|"
    r"not\s+(?:on\s+board|sold|convinced)\b|i'?m\s+not\b)",
    re.I,
)

# Acknowledgement phrases -- the "valid point / fair point / I agree" family
# that produces low-value loop patterns.

# Fact-chasing questions ask the group to resolve unavailable external facts
# (availability, waitlists, booking status, exact schedules). Those questions
# pushed chats into fake planning: "will check again today", "call ahead", etc.
_FACT_CHASING_QUESTION = re.compile(
    r"\b(?:"
    # Live updates / availability / booking-state checks
    r"what['’]?s\s+the\s+latest\s+on|any\s+update\s+on|availability|available\s+rooms?|"
    r"fully\s+booked|booked\s+out|reservation\s+status|wait\s*list|waitlist|call\s*-?\s*ahead|"
    # Asking people to look up or call for information
    r"do\s+we\s+know\s+(?:if|whether|what)|does\s+anyone\s+know\s+(?:if|whether|what)|"
    r"can\s+we\s+(?:check|call|ask|find\s+out|look\s+into)|"
    r"could\s+we\s+(?:check|call|ask|find\s+out|look\s+into)|"
    r"should\s+we\s+(?:check|call|ask|find\s+out|look\s+into)|"
    r"let['’]?s\s+(?:check|call|ask|find\s+out|look\s+into)|"
    r"look\s+into\s+that|check\s+that|find\s+that\s+out|"
    # Exact missing external values
    r"actual\s+(?:\w+\s+){0,3}(?:probability|probabilities|chance|cost|price|difference|fee|fees|duration|time)|"
    r"exact\s+(?:\w+\s+){0,3}(?:probability|probabilities|chance|cost|price|difference|fee|fees|duration|time|wait|schedule)|"
    r"compare\s+(?:across|between)\s+(?:these\s+)?options|"
    r"policy\s+on\s+(?:refunds?|changes?|cancellations?|baggage)|"
    r"refund\s+policy|change\s+policy|cancellation\s+policy|baggage\s+(?:fee|fees|policy)|"
    r"rough\s+estimate|transport(?:ation)?\s+costs?|max\s+budget|maximum\s+budget|"
    r"allergen\s+(?:protocols?|handling|safety)|food\s+safety\s+(?:protocols?|handling)|"
    r"does\s+(?:it|that|Option\s+[A-D]).{0,35}\b(?:have|offer|provide)\b.{0,35}\b(?:better|clear|explicit|safer)\b|"
    # Guarantees / external services
    r"guarantee\s+(?:a|an|the)?|partnerships?\s+with|equipment\s+rentals?"
    r")\b",
    re.I,
)


_ACK_PHRASES = re.compile(
    r"\b(?:"
    r"valid\s+(?:point|concern)|"
    r"fair\s+(?:point|enough)|"
    r"good\s+(?:point|call|question)|"
    r"great\s+point|"
    r"makes?\s+sense|"
    r"i\s+(?:totally\s+)?(?:agree|hear\s+you|see\s+(?:your\s+)?point)|"
    r"that(?:'s|\s+is)\s+(?:a\s+)?(?:fair|valid|good|true|right)|"
    r"that(?:'s|\s+is)\s+(?:a\s+)?concern|"
    r"that\s+concern\s+is\s+valid|"
    r"i\s+see\s+what\s+(?:you|\w+)\s+(?:mean|said|are\s+saying)|"
    r"\w+(?:'s|\s+is)\s+right(?:\s+about)?|"
    r"absolutely|noted|granted|true\s+enough|exactly"
    r")\b",
    re.I,
)



# ---------------------------------------------------------------------------
# Option attribute helpers
# ---------------------------------------------------------------------------

def _option_letter_from_text(option: str) -> Optional[str]:
    m = re.match(r"\s*Option\s+([A-D])\b", option, re.I)
    return m.group(1).upper() if m else None


def _option_attrs(options: list[str]) -> dict[str, dict[str, str]]:
    """Parse `attrs: key=value` pairs from generated option cards."""
    out: dict[str, dict[str, str]] = {}
    for opt in options:
        letter = _option_letter_from_text(opt)
        if not letter:
            continue
        attrs: dict[str, str] = {}
        if ": attrs:" in opt:
            raw = opt.split(": attrs:", 1)[1].split("; upside:", 1)[0]
            for item in raw.split(","):
                if "=" not in item:
                    continue
                key, value = [x.strip() for x in item.split("=", 1)]
                attrs[key] = value
        out[letter] = attrs
    return out


def _time_to_minutes(value: str) -> Optional[int]:
    value = value.strip().lower()
    m = re.match(r"^(\d{1,2}):(\d{2})$", value)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"^(\d{1,2})\s*(am|pm)$", value)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2) == "pm":
            h += 12
        return h * 60
    return None


def _mentioned_times(text: str) -> list[int]:
    times: list[int] = []
    for m in re.finditer(r"\b(\d{1,2}:\d{2})\b", text):
        val = _time_to_minutes(m.group(1))
        if val is not None:
            times.append(val)
    for m in re.finditer(r"\b(\d{1,2})\s*(am|pm)\b", text, re.I):
        val = _time_to_minutes(f"{m.group(1)}{m.group(2).lower()}")
        if val is not None:
            times.append(val)
    return times


def detect_option_attribute_mismatch(text: str, options: list[str], resolver: OptionResolver) -> Optional[VerificationIssue]:
    """Catch changing listed scenario values, currently mainly departure time.

    Example: option card says B departs 14:00, but the utterance says "3 pm
    departure for Option B". This is not external lookup; it is direct mutation
    of a scenario fact and should be repaired.
    """
    attrs_by_letter = _option_attrs(options)
    refs = resolver.options_in(text)
    if not refs:
        return None
    said_times = _mentioned_times(text)
    if said_times:
        for letter in refs:
            dep = attrs_by_letter.get(letter, {}).get("departure_time")
            dep_min = _time_to_minutes(dep) if dep else None
            if dep_min is not None and all(abs(t - dep_min) > 10 for t in said_times):
                return VerificationIssue(
                    code="OPTION_ATTRIBUTE_MISMATCH",
                    severity="repair",
                    message=f"Text changes Option {letter}'s listed departure time ({dep}).",
                )
    # Price mismatch near a referenced option. Conservative: only flag euro
    # amounts when the exact listed price exists and no mentioned amount matches.
    prices = [int(x) for x in re.findall(r"€\s*(\d{2,4})", text)]
    if prices:
        for letter in refs:
            price = attrs_by_letter.get(letter, {}).get("price_eur") or attrs_by_letter.get(letter, {}).get("price_per_night_eur") or attrs_by_letter.get(letter, {}).get("price_per_person_eur")
            if price and all(int(price) != p for p in prices):
                return VerificationIssue(
                    code="OPTION_ATTRIBUTE_MISMATCH",
                    severity="repair",
                    message=f"Text changes Option {letter}'s listed price (€{price}).",
                )
    return None


_CURRENCY_OR_PCT = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?%")
_PLAIN_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _fact_check_option_claims(turn_text: str, option_texts: list[str]) -> list[str]:
    """Flag numbers/quoted features that look like invented option attributes."""
    source = " ".join(option_texts).lower()
    suspicious: list[str] = []

    for match in _CURRENCY_OR_PCT.finditer(turn_text):
        token = match.group(0)
        if token.lower() not in source:
            suspicious.append(token)

    proximity = int(cfg.grounding.option_proximity_chars)
    option_spans = _option_attribute_spans(turn_text, option_texts, proximity)
    for match in _PLAIN_NUMBER.finditer(turn_text):
        token = match.group(0)
        if token.lower() in source or _CURRENCY_OR_PCT.match(token):
            continue
        if any(start <= match.start() <= end for start, end in option_spans):
            suspicious.append(token)

    for match in re.finditer(r'"([^"]{3,80})"', turn_text):
        phrase = match.group(1).strip()
        if phrase.lower() not in source:
            suspicious.append(f'"{phrase}"')

    for match in re.finditer(r"\(([^)]{4,60})\)", turn_text):
        phrase = match.group(1).strip()
        if re.search(r"\d", phrase) and phrase.lower() not in source:
            suspicious.append(f"({phrase})")

    return suspicious


def _option_attribute_spans(
    turn_text: str,
    option_texts: list[str],
    proximity: int,
) -> list[tuple[int, int]]:
    resolver = OptionResolver(option_texts)
    raw_spans = resolver.option_mention_spans(turn_text)
    if not raw_spans:
        return []
    spans = [
        (max(0, start - proximity), min(len(turn_text), end + proximity))
        for start, end in raw_spans
    ]
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

# ---------------------------------------------------------------------------
# Participant checks
# ---------------------------------------------------------------------------

def detect_empty(text: str) -> Optional[VerificationIssue]:
    if not text or not text.strip() or len(text.split()) < 2:
        return VerificationIssue(
            code="EMPTY_OR_SILENCE",
            severity="fatal",
            message="Turn is empty or too short.",
        )
    return None


def detect_name_prefix(text: str, speaker_name: str) -> Optional[VerificationIssue]:
    if text.strip().lower().startswith(f"{speaker_name.lower()}:"):
        return VerificationIssue(
            code="NAME_PREFIX",
            severity="warn",
            message="Turn starts with speaker's own name as a prefix.",
        )
    return None


def detect_invalid_option_reference(
    text: str, options: list[str], resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """Flag Option X where X is not among the valid letters A-D (or whatever exists)."""
    valid = {l.upper() for l in resolver.letters}
    for m in re.finditer(r"\bOption\s+([A-Za-z])\b", text, re.I):
        letter = m.group(1).upper()
        if letter not in valid:
            return VerificationIssue(
                code="INVALID_OPTION_REFERENCE",
                severity="repair",
                message=f"Referenced non-existent Option {letter} (valid: {sorted(valid)}).",
            )
    return None


def detect_option_denial(
    text: str, options: list[str], resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """Flag claims that a listed option is unavailable or doesn't exist."""
    if _OPTION_DENIAL.search(text):
        return VerificationIssue(
            code="VALID_OPTION_DENIED",
            severity="repair",
            message="Claimed a listed option is unavailable or doesn't exist.",
        )
    return None


def detect_invented_option_attribute(
    text: str, options: list[str], resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """Flag invented numbers or prices asserted close to an option mention."""
    flags = _fact_check_option_claims(text, options)
    if flags:
        return VerificationIssue(
            code="INVENTED_OPTION_FACT",
            severity="repair",
            message=f"Possible invented option attributes detected: {flags}",
        )
    return None


def detect_self_repetition(
    text: str,
    speaker_name: str,
    history: list[str],
    persona_state: Optional["ParticipantState"],
) -> Optional[VerificationIssue]:
    """Flag if this turn is too similar to the speaker's most recent own turn."""
    threshold_last = float(_vcfg("own_last_turn_jaccard", 0.55))
    threshold_points = float(_vcfg("own_points_similarity", 0.65))

    current_toks = _tokenize(text)
    if not current_toks:
        return None

    # Check against last own turn in raw history
    for line in reversed(history):
        if ":" not in line:
            continue
        spk, msg = line.split(":", 1)
        if spk.strip() == speaker_name:
            prev_toks = _tokenize(msg)
            if _jaccard(current_toks, prev_toks) >= threshold_last:
                return VerificationIssue(
                    code="SELF_REPETITION",
                    severity="repair",
                    message="Turn is too similar to speaker's immediately previous turn.",
                )
            break  # only check the most recent own turn for "repair" severity

    # Check against compact memory points (warn severity -- older material)
    if persona_state and len(getattr(persona_state, "points_made", [])) > 1:
        for point in persona_state.points_made[:-1]:  # skip the last one (covered above)
            point_toks = _tokenize(point)
            if _jaccard(current_toks, point_toks) >= threshold_points:
                return VerificationIssue(
                    code="SELF_REPETITION",
                    severity="warn",
                    message="Turn repeats a key point the speaker made earlier.",
                )

    return None


def _ack_in_head(text: str) -> bool:
    """True iff an acknowledgement phrase appears in the opening of the text.
    'Opening' is the first `ack_loop_head_chars` characters."""
    head_chars = int(_vcfg("ack_loop_head_chars", 60))
    head = text.strip()[:head_chars]
    return bool(_ACK_PHRASES.search(head))


def _is_ack_led(text: str) -> bool:
    """A turn is 'ack-led' if it opens with an acknowledgement phrase, OR if
    it's short enough that an ack phrase anywhere makes the whole turn pure
    acknowledgement.

    Both shapes feed the loop:
      "Fair point. I still think B."           -- ack-led with thin pivot
      "Yeah, that concern is valid."            -- short pure ack
    """
    if not text or not text.strip():
        return False
    short_cap = int(_vcfg("ack_loop_max_words_for_pure_ack", 14))
    words = text.split()
    if len(words) <= short_cap and _ACK_PHRASES.search(text):
        return True
    return _ack_in_head(text)


def detect_ack_loop(
    text: str,
    speaker_name: str,
    history: list[str],
) -> Optional[VerificationIssue]:
    """Group-level acknowledgement-loop detection.

    Fires when THIS turn is ack-led AND the recent non-self participant turns
    also contain acknowledgement language. The loop only matters as a pattern;
    a single ack turn is fine in isolation.
    """
    if not _vcfg("check_ack_loop", True):
        return None
    if not _is_ack_led(text):
        return None

    window = int(_vcfg("ack_loop_window", 3))
    needed = int(_vcfg("ack_loop_min_recent_with_ack", 2))

    scanned = 0
    recent_ack_count = 0
    for line in reversed(history):
        if scanned >= window:
            break
        if ":" not in line:
            continue
        spk, msg = line.split(":", 1)
        spk = spk.strip()
        if spk in {"Moderator"} or spk == speaker_name:
            continue
        scanned += 1
        if _ACK_PHRASES.search(msg):
            recent_ack_count += 1

    if recent_ack_count < needed:
        return None

    return VerificationIssue(
        code="ACK_LOOP",
        severity="repair",
        message=(
            f"Acknowledgement loop: this turn is ack-led and {recent_ack_count} "
            f"of the last {scanned} participant turns also acknowledged."
        ),
    )


# Stopwords stripped before semantic-repeat comparison so function words don't
# pad the union and hide real attribute-level overlap.
_SEMANTIC_STOPWORDS = frozenset({
    "the", "and", "but", "for", "with", "that", "this", "these", "those",
    "are", "was", "were", "been", "being", "have", "has", "had",
    "would", "could", "should", "will", "can", "may", "might",
    "you", "your", "our", "us", "we", "they", "them", "their",
    "really", "just", "very", "also", "too", "still", "yet",
    "from", "into", "onto", "about", "than", "then", "there", "here",
    "what", "when", "where", "which", "who", "how", "why",
    "option", "options",
})


def detect_semantic_point_repeat(
    text: str,
    persona_state: Optional["ParticipantState"],
    resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """Repair when this turn restates a
    point the same speaker already made.

    Two shapes count as a repeat:
      1. Same option mentioned + jaccard >= same_option threshold on
         content tokens (stopwords stripped). ("Reliability of A is worth
         the cost" vs. "Option A's reliability offsets cost".)
      2. No option mentioned in either turn, but jaccard >= the stricter
         no-option threshold (catches "cost is a concern" loops).
    """
    if not _vcfg("check_semantic_point_repeat", True):
        return None
    if persona_state is None:
        return None
    points = getattr(persona_state, "points_made", []) or []
    if not points:
        return None

    threshold_same = float(_vcfg("semantic_repeat_jaccard_same_option", 0.50))
    threshold_none = float(_vcfg("semantic_repeat_jaccard_no_option", 0.65))

    current_opts = set(resolver.options_in(text))
    current_toks = _tokenize(text) - _SEMANTIC_STOPWORDS
    if len(current_toks) < 2:
        return None

    # Scan prior points oldest-first so we report the first match; skip the most
    # recent (covered by SELF_REPETITION against last own turn).
    for prior in points[:-1] if len(points) >= 1 else []:
        prior_opts = set(resolver.options_in(prior))
        prior_toks = _tokenize(prior) - _SEMANTIC_STOPWORDS
        if len(prior_toks) < 2:
            continue
        overlap = _jaccard(current_toks, prior_toks)
        same_option = bool(current_opts & prior_opts)
        if same_option and overlap >= threshold_same:
            return VerificationIssue(
                code="SEMANTIC_POINT_REPEAT",
                severity="repair",
                message=f"Same option-attribute point as earlier: '{prior[:60]}'",
            )
        if not current_opts and not prior_opts and overlap >= threshold_none:
            return VerificationIssue(
                code="SEMANTIC_POINT_REPEAT",
                severity="repair",
                message=f"Same point (no specific option) as earlier: '{prior[:60]}'",
            )

    return None



def detect_fact_chasing_question(text: str, phase: str) -> Optional[VerificationIssue]:
    """Flag questions that try to resolve unavailable outside facts.

    The simulator has only the option cards. It should decide from listed
    trade-offs, not invent calls, booking checks, availability updates, or
    live waitlist information.
    """
    if phase not in ("negotiation", "narrowing"):
        return None
    # This covers both literal questions ("what is the refund policy?") and
    # planning/request moves ("let's look into that", "we should call ahead").
    # Both push the simulated chat toward unavailable external facts.
    if _FACT_CHASING_QUESTION.search(text):
        return VerificationIssue(
            code="FACT_CHASING_QUESTION",
            severity="repair",
            message="Question asks for unavailable outside facts instead of deciding from listed option trade-offs.",
        )
    return None

def detect_question_chain(text: str, phase: str, history: list[str]) -> Optional[VerificationIssue]:
    """Flag new questions when the local thread is already question-heavy.

    The simulator should not become Q -> Q -> Q. A new question is allowed
    occasionally, but after a recent participant question the next useful move
    is usually a short answer, reaction, or decision statement.
    """
    if phase != "negotiation" or "?" not in text:
        return None
    recent_participant_msgs: list[str] = []
    for line in reversed(history):
        if ":" not in line:
            continue
        spk, msg = line.split(":", 1)
        if spk.strip() == "Moderator":
            continue
        recent_participant_msgs.append(msg.strip())
        if len(recent_participant_msgs) >= 4:
            break
    if not recent_participant_msgs:
        return None
    q_count_last3 = sum("?" in m for m in recent_participant_msgs[:3])
    q_count_last4 = sum("?" in m for m in recent_participant_msgs[:4])
    if q_count_last3 >= 1 or q_count_last4 >= 2:
        return VerificationIssue(
            code="QUESTION_CHAIN",
            severity="repair",
            message="Another question would create a question-after-question chain; answer or make a decision move instead.",
        )
    return None


def detect_missing_vote(
    text: str, phase: str, resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """In narrowing phase, flag if turn has no explicit vote."""
    if phase != "narrowing":
        return None
    if resolver.vote_in(text) is None:
        return VerificationIssue(
            code="MISSING_EXPLICIT_VOTE",
            severity="repair",
            message="Narrowing turn has no explicit vote for a single option.",
        )
    return None


def detect_unclear_confirmation(
    text: str, phase: str, candidate: Optional[str],
) -> Optional[VerificationIssue]:
    """In confirmation phase, flag if turn is classifiable as neither yes nor no."""
    if phase != "confirmation":
        return None
    lower = text.strip().lower()
    is_yes = bool(_YES_PAT.search(lower))
    is_no = bool(_NO_PAT.match(lower))
    if not is_yes and not is_no:
        return VerificationIssue(
            code="UNCLEAR_CONFIRMATION",
            severity="repair",
            message="Confirmation turn is neither clear yes nor clear no.",
        )
    return None


def detect_weak_compromise_confirmation(
    text: str, phase: str, candidate: Optional[str], persona_state: Optional["ParticipantState"],
) -> Optional[VerificationIssue]:
    """Flag bare yes-lines when accepting a non-preferred compromise.

    A plain "that's fine" is mechanically clear, but it made compromises feel
    unearned. If the candidate is not the speaker's preferred option, ask for
    one short reason why they can live with it.
    """
    if phase != "confirmation" or not candidate or persona_state is None:
        return None
    persona = getattr(persona_state, "persona_ref", None)
    beliefs = getattr(persona, "beliefs", None)
    if not beliefs or candidate == getattr(beliefs, "preferred", None):
        return None
    lower = text.strip().lower()
    is_yes = bool(_YES_PAT.search(lower))
    is_no = bool(_NO_PAT.match(lower))
    if not is_yes or is_no:
        return None
    words = re.findall(r"\b\w+\b", lower)
    reason_markers = re.compile(
        r"\b(because|since|as|given|quiet|price|cost|wait|time|travel|noise|"
        r"menu|variety|safety|allergen|local|comfort|flexib|reliable|"
        r"works?\s+because|still\s+prefer|but)\b",
        re.I,
    )
    if len(words) <= 6 or not reason_markers.search(text):
        return VerificationIssue(
            code="WEAK_COMPROMISE_CONFIRMATION",
            severity="repair",
            message="Non-preferred compromise acceptance needs one short reason, not only a bare yes.",
        )
    return None


# ---------------------------------------------------------------------------
# Main verification entry points
# ---------------------------------------------------------------------------

def verify_participant_turn(
    text: str,
    speaker_name: str,
    phase: str,
    options: list[str],
    history: list[str],
    persona_state: Optional["ParticipantState"],
    resolver: OptionResolver,
    candidate: Optional[str] = None,
) -> VerificationResult:
    """Run all participant-turn checks. Returns a VerificationResult."""
    if not _vcfg("enabled", True):
        return VerificationResult(ok=True)

    issues: list[VerificationIssue] = []

    empty = detect_empty(text)
    if empty:
        return VerificationResult(ok=False, issues=[empty], needs_repair=True)

    prefix = detect_name_prefix(text, speaker_name)
    if prefix:
        issues.append(prefix)

    if _vcfg("check_option_validity", True):
        inv_ref = detect_invalid_option_reference(text, options, resolver)
        if inv_ref:
            issues.append(inv_ref)

        opt_denial = detect_option_denial(text, options, resolver)
        if opt_denial:
            issues.append(opt_denial)

        inv_fact = detect_invented_option_attribute(text, options, resolver)
        if inv_fact:
            issues.append(inv_fact)

        attr_mismatch = detect_option_attribute_mismatch(text, options, resolver)
        if attr_mismatch:
            issues.append(attr_mismatch)

        if _vcfg("check_fact_chasing_questions", True):
            fact_q = detect_fact_chasing_question(text, phase)
            if fact_q:
                issues.append(fact_q)

        if _vcfg("check_question_chains", True):
            q_chain = detect_question_chain(text, phase, history)
            if q_chain:
                issues.append(q_chain)

    if _vcfg("check_repetition", True):
        rep = detect_self_repetition(text, speaker_name, history, persona_state)
        if rep:
            issues.append(rep)

    # Group-level acknowledgement loop. Skip in phases where
    # a brief yes/no IS the expected shape (confirmation/closure/opening).
    if _vcfg("check_ack_loop", True) and phase in ("negotiation", "narrowing"):
        ack = detect_ack_loop(text, speaker_name, history)
        if ack:
            issues.append(ack)

    # Semantic point repeat (stronger than warn-only memory check).
    if _vcfg("check_semantic_point_repeat", True) and phase in ("negotiation", "narrowing"):
        sem = detect_semantic_point_repeat(text, persona_state, resolver)
        if sem:
            issues.append(sem)

    if _vcfg("check_votes", True):
        mv = detect_missing_vote(text, phase, resolver)
        if mv:
            issues.append(mv)

    if _vcfg("check_confirmation", True):
        uc = detect_unclear_confirmation(text, phase, candidate)
        if uc:
            issues.append(uc)
        wc = detect_weak_compromise_confirmation(text, phase, candidate, persona_state)
        if wc:
            issues.append(wc)

    needs_repair = any(i.severity == "repair" for i in issues)
    return VerificationResult(ok=not needs_repair, issues=issues, needs_repair=needs_repair)
