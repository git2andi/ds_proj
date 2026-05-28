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

Checks for moderator turns:
  - introducing a new option letter
  - suggesting a combined/mixed solution (when not allowed)
  - declaring fake consensus

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

# Moderator: mixed solution signals
_MIXED_SOLUTION = re.compile(
    r"\b(?:combine|merge|mix\s+together|blend|hybrid\s+of|"
    r"both\s+option|option\s+[a-d]\s+and\s+option\s+[a-d]|"
    r"combine\s+(?:option|both)|merge\s+(?:option|both))\b",
    re.I,
)

# Fake consensus (for moderator turns outside appropriate phases)
_FAKE_CONSENSUS = re.compile(
    r"\b(?:everyone\s+agrees?|we\s+all\s+agree|consensus\s+(?:is|(?:has\s+been\s+)?reached)|"
    r"unanimou(?:s|sly)|it'?s\s+settled|we'?(?:ve|'re)\s+all\s+agreed)\b",
    re.I,
)


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
    from reasoning import fact_check
    flags = fact_check(text, options, topic="")
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


# ---------------------------------------------------------------------------
# Moderator checks
# ---------------------------------------------------------------------------

def detect_moderator_new_option(
    text: str, options: list[str], resolver: OptionResolver,
) -> Optional[VerificationIssue]:
    """Flag if moderator text introduces an option letter beyond the existing set."""
    valid = {l.upper() for l in resolver.letters}
    for m in re.finditer(r"\bOption\s+([A-Za-z])\b", text, re.I):
        letter = m.group(1).upper()
        if letter not in valid:
            return VerificationIssue(
                code="MODERATOR_NEW_OPTION",
                severity="repair",
                message=f"Moderator introduced non-existent Option {letter}.",
            )
    return None


def detect_moderator_mixed_solution(
    text: str, allow_multi_option_solution: bool = False,
) -> Optional[VerificationIssue]:
    """Flag if moderator suggests combining multiple options (when not allowed)."""
    if allow_multi_option_solution:
        return None
    if _MIXED_SOLUTION.search(text):
        return VerificationIssue(
            code="MODERATOR_MIXED_SOLUTION",
            severity="warn",
            message="Moderator suggested combining multiple options.",
        )
    return None


def detect_fake_consensus(text: str, phase: str) -> Optional[VerificationIssue]:
    """Flag if moderator declares consensus outside appropriate phases."""
    if phase in ("confirmation", "closure", "emergence"):
        return None
    if _FAKE_CONSENSUS.search(text):
        return VerificationIssue(
            code="FAKE_CONSENSUS",
            severity="warn",
            message="Moderator may be declaring consensus prematurely.",
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

    if _vcfg("check_repetition", True):
        rep = detect_self_repetition(text, speaker_name, history, persona_state)
        if rep:
            issues.append(rep)

    if _vcfg("check_votes", True):
        mv = detect_missing_vote(text, phase, resolver)
        if mv:
            issues.append(mv)

    if _vcfg("check_confirmation", True):
        uc = detect_unclear_confirmation(text, phase, candidate)
        if uc:
            issues.append(uc)

    needs_repair = any(i.severity == "repair" for i in issues)
    return VerificationResult(ok=not needs_repair, issues=issues, needs_repair=needs_repair)


def verify_moderator_turn(
    text: str,
    options: list[str],
    resolver: OptionResolver,
    phase: str = "",
    allow_multi_option_solution: bool = False,
) -> VerificationResult:
    """Run moderator-specific checks. Returns a VerificationResult."""
    if not _vcfg("check_moderator_options", True):
        return VerificationResult(ok=True)

    issues: list[VerificationIssue] = []

    empty = detect_empty(text)
    if empty:
        return VerificationResult(ok=False, issues=[empty], needs_repair=True)

    new_opt = detect_moderator_new_option(text, options, resolver)
    if new_opt:
        issues.append(new_opt)

    mixed = detect_moderator_mixed_solution(text, allow_multi_option_solution)
    if mixed:
        issues.append(mixed)

    fake = detect_fake_consensus(text, phase)
    if fake:
        issues.append(fake)

    needs_repair = any(i.severity in ("repair", "fatal") for i in issues)
    return VerificationResult(ok=not needs_repair, issues=issues, needs_repair=needs_repair)
