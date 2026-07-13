"""Deterministic test stand-in for the validator LLM (todo_validation item 6).

Production has exactly one soft-semantic authority: the validator role,
verified deterministically in ``src/interpreter.py``. Offline tests still
need a validator stand-in to drive the full candidate pipeline, so the old
conservative regex recall lives HERE, as a test double — it has no production
consumer and no state authority of its own.

Its recall deliberately equals the legacy parser's: strict commitments and
blockers come from the retained critical parser in ``src/parsing.py``; the
soft categories (ordinary concerns, weak support, comparisons, softening,
compromise offers, conditional support) use the regex vocabulary moved out of
production. Natural-language variants beyond this recall need a scripted
validator payload (see tests/test_interpreter.py).
"""

from __future__ import annotations

import re

from interpreter import TurnInterpreter
from models import (
    ActType,
    AnswerEvidence,
    BlockerEvidence,
    CommitmentEvidence,
    ComparisonEvidence,
    ConcernEvidence,
    EvidenceSpan,
    ProposalEvidence,
    QuestionEvidence,
    SofteningEvidence,
    SupportEvidence,
    VisibleEvidence,
)
from parsing import (
    OptionResolver,
    _CONDITIONAL_AFTER_COMMIT,
    _HARD_CONDITIONAL,
    _HEDGE,
    _REJECT,
    _commitment_object,
    _nearest_option,
    active_blocker_option,
    blocker_resolution_option,
    visible_commitment,
    visible_question,
)

# ---------------------------------------------------------------------------
# Soft-semantic vocabulary (moved from src/parsing.py; test recall only).
# ---------------------------------------------------------------------------

_SOFT_OBJECT = re.compile(
    r"\b(?:concern(?:s|ed)?|worr(?:y|ies|ied)|bother(?:s|ed)?\s+me|problems?|issues?|downsides?|"
    r"too\s+expensive|too\s+far|too\s+late|too\s+(?:pricey|costly)|risky|"
    r"(?:a\s+bit|too|rather|quite|pretty)\s+steep|"
    r"(?:seems?|looks?|feels?)\s+(?:too\s+|quite\s+|pretty\s+|a\s+bit\s+)?(?:high|expensive|pricey|risky|steep)|"
    r"not\s+ideal|doesn'?t\s+fit|would\s+be\s+hard)\b",
    re.I,
)

_SOFTENING = re.compile(
    r"\b(?:starting\s+to\s+(?:make\s+(?:more|a\s+lot\s+of)\s+sense|look\s+(?:better|safer|stronger|smarter|more\s+\w+)|win\s+me\s+over)|"
    r"beginning\s+to\s+(?:make\s+sense|look\s+better)|"
    r"(?:coming|come)\s+around\s+(?:to|on)|warming\s+(?:up\s+)?to|"
    r"makes\s+more\s+sense\s+(?:to\s+me\s+)?(?:now|after)|"
    r"growing\s+on\s+me|is\s+winning\s+me\s+over|(?:has|is)\s+won\s+me\s+over|"
    r"clicks\s+with\s+me\s+now|speaks\s+to\s+me\s+now|sounds\s+better\s+(?:to\s+me\s+)?now|"
    r"i\s+(?:see|get)\s+the\s+appeal\s+(?:of|now)|makes\s+a\s+strong(?:er)?\s+case|"
    r"more\s+tempting\s+now|i'?m\s+starting\s+to\s+(?:see|like|favor|lean\s+toward))\b",
    re.I,
)

_COMPROMISE_OFFER = re.compile(
    r"\b(?:could|can|would)\s+(?:we|everyone|you\s+all|y'?all)\s+(?:all\s+)?live\s+with\b|"
    r"\bwhat\s+if\s+we\s+(?:went|go|all\s+went|all\s+go)\s+with\b|"
    r"\bmeet\s+in\s+the\s+middle\s+(?:on|with|at)\b|"
    r"\bwould\s+(?:that|this|it)\s+work\s+for\s+everyone\b|"
    r"\bas\s+a\s+(?:compromise|middle\s+ground)\b",
    re.I,
)

_COMPARATIVE = re.compile(
    r"\b(?:than|versus|vs\.?|compared?\s+(?:to|with)|instead\s+of|over\b|"
    r"rather\s+than|beats?|wins?\s+(?:over|against)|side\s+by\s+side|trade-?off)\b",
    re.I,
)

_PRO_CLAIM = re.compile(
    r"\b(?:solves|fixes|keeps|gives|covers|fits|means|delivers|works|saves|hits|"
    r"offers|helps|suits)\b|\bi\s+(?:really\s+)?(?:like|love|prefer)\b",
    re.I,
)

# Old parsing._COMMIT is still importable but conditional support needs it via
# the same guard set the legacy conditional_support_option used.
from parsing import _COMMIT  # noqa: E402


def softening_option(check_text: str, resolver: OptionResolver) -> str | None:
    match = _SOFTENING.search(check_text)
    if not match:
        return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


def conditional_support_option(check_text: str, resolver: OptionResolver) -> str | None:
    match = _COMMIT.search(check_text)
    if not match or _REJECT.search(check_text):
        return None
    if not (_HARD_CONDITIONAL.search(check_text) or _HEDGE.search(check_text) or _CONDITIONAL_AFTER_COMMIT.search(check_text)):
        return None
    ids = resolver.ids_in_text(check_text)
    if len(ids) > 1:
        near = _commitment_object(check_text, resolver)
        ids = [near] if near else ids
    return ids[0] if len(ids) == 1 else None


def compromise_offer_option(check_text: str, resolver: OptionResolver) -> str | None:
    match = _COMPROMISE_OFFER.search(check_text)
    if not match:
        return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


def realized_comparison(text: str, resolver: OptionResolver) -> bool:
    check = text.replace("’", "'").replace("‘", "'")
    return len(set(resolver.ids_in_text(check))) >= 2 and bool(_COMPARATIVE.search(check))


def has_support_claim(text: str) -> bool:
    return bool(_PRO_CLAIM.search(text.replace("’", "'")))


def support_claim_target(text: str, resolver: OptionResolver) -> str | None:
    check = text.replace("’", "'")
    match = _PRO_CLAIM.search(check)
    if not match:
        return None
    return _nearest_option(check, match.start(), match.end(), resolver)


# ---------------------------------------------------------------------------
# The adapter itself.
# ---------------------------------------------------------------------------


def derive_evidence(
    text: str,
    resolver: OptionResolver,
    *,
    speaker_id: str = "p1",
    participant_names: dict[str, str] | None = None,
    intent=None,
    previous_speaker_id: str | None = None,
) -> VisibleEvidence:
    """Use the production deterministic interpreter in offline tests."""
    interpreter = TurnInterpreter(
        None, resolver, None, participant_names or {}, mode="critical"
    )
    result = interpreter.interpret(
        text=text,
        speaker_id=speaker_id,
        intent=intent,
        previous_speaker_id=previous_speaker_id,
        target_turn_text="target" if intent is not None and getattr(intent, "respond_to_turn", None) is not None else None,
    )
    assert result.evidence is not None
    return result.evidence


# Contextual acts keep their routed label for the display precedence, exactly
# as the legacy realized-act classification did.
_CONTEXTUAL_ACTS = {
    ActType.OPENING,
    ActType.ANSWER,
    ActType.PROCESS,
    ActType.COMPROMISE,
    ActType.VOTE,
    ActType.CLOSING,
}


def _legacy_primary_act(
    *,
    commitment,
    soft_rejects,
    hard_rejects,
    question_scope,
    option_refs,
    comparative,
    check_text,
    intent,
) -> ActType:
    if commitment and commitment[0] in {"vote", "accept"}:
        return ActType.VOTE
    if commitment or hard_rejects or soft_rejects:
        return ActType.CONCERN
    if question_scope:
        return ActType.ASK
    if len(set(option_refs)) >= 2 and comparative:
        return ActType.COMPARE
    if intent is not None and intent.act in _CONTEXTUAL_ACTS:
        return intent.act
    if option_refs and _PRO_CLAIM.search(check_text):
        return ActType.SUPPORT
    return ActType.COMMENT
