"""Visible-text parsing for option references and public commitments.

Final outcomes must be based on what the transcript visibly says. This parser is
therefore conservative: it only records a vote/acceptance when one option is
mentioned unambiguously and the utterance contains a clear commitment phrase.
"""

from __future__ import annotations

import re

from aliases import _GENERIC, _STOPWORDS, short_alias_map
from models import ActType, DialogueAct, MoveIntent, OptionCard
from utils import normalise_ws

# Common words that must never become a standalone option alias: they appear in
# ordinary sentences and would cause false option matches (e.g. "with", "data").
_ALIAS_STOPWORDS = _STOPWORDS | _GENERIC | {
    "with", "data", "open", "core", "team", "service", "services", "system",
    "support", "framework", "extension", "edition", "version", "standard",
    "cloud", "online", "based", "free", "premium", "basic", "pro",
    "analytics", "warehouse", "database", "serverless", "managed", "hosted",
    "platform", "solution", "single", "plan", "suite", "tool", "tools", "package",
    "neighborhood", "community", "local", "event", "center", "food", "project",
    "class", "course", "workshop", "session", "activity", "group", "program",
    "assistance", "weekly", "monthly", "morning", "evening", "weekend",
}

_QUESTION = re.compile(r"\?")
_RHETORICAL_TAIL = re.compile(r",\s*(?:right|yeah|no|huh|eh|you know|don't you think)\s*\?\s*$", re.I)
_GENUINE_QUESTION = re.compile(
    r"\b(?:how\s+(?:many|much|long|far|about)|what(?:'s|\s+is|\s+are|\s+do|\s+does|\s+about|\s+if)"
    r"|where|when|which|who|can\s+(?:we|it|they|you|anyone)|do\s+(?:we|they|you|i)|does\s+(?:it|that|this|anyone)"
    r"|is\s+(?:it|there|that|anyone)|are\s+(?:there|they|we|you|any)|could\s+(?:we|it|you|anyone)|would\s+(?:it|that|we|you|anyone)"
    r"|should\s+(?:we|i|you|they)|shall\s+we|will\s+(?:we|it|that)|has\s+anyone|any\s+of\s+us)\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(?:maybe|perhaps|possibly|might|could|leaning|seems?|sounds\s+like|"
    r"not\s+sure|not\s+sold|not\s+convinced|"
    r"for\s+now|unless|if|as\s+long\s+as|provided\s+that|i\s+guess|i\s+suppose|"
    r"would\s+need|need\s+to\s+know|depends|still\s+unsure)\b",
    re.I,
)
_COMMIT = re.compile(
    r"\b(?:i\s+vote\s+for|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to)|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+pick\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"gets?\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|"
    r"let'?s\s+go\s+with|we\s+should\s+go\s+with|go\s+with|settle\s+on|pick|choose|"
    r"i\s+support|i\s+accept|i\s+can\s+support|i'?m\s+fine\s+with|fine\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|i'?m\s+okay\s+with|okay\s+with|agree\s+on|final\s+choice)\b",
    re.I,
)
_SOFT_COMMIT = re.compile(
    r"\b(?:i\s+can\s+support|i\s+support|i\s+accept|i'?m\s+fine\s+with|fine\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|i'?m\s+okay\s+with|okay\s+with|agree\s+on)\b",
    re.I,
)
_DIRECT_VOTE = re.compile(
    r"\b(?:i\s+vote\s+for|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to)|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+pick\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"gets?\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|"
    r"let'?s\s+go\s+with|we\s+should\s+go\s+with|settle\s+on|final\s+choice)\b",
    re.I,
)
_CONDITIONAL_AFTER_COMMIT = re.compile(
    r"\b(?:but|however|though|although|still|only\s+if|if\s+we|if\s+it|as\s+long\s+as|"
    r"provided\s+that|are\s+we\s+okay|concern|worry|problem|issue|not\s+sure|unless)\b",
    re.I,
)
_HARD_CONDITIONAL = re.compile(
    r"(?:\?|\bonly\s+if\b|\bif\s+(?:we|it|they|you)\b|\bas\s+long\s+as\b|"
    r"\bprovided\s+that\b|\bunless\b|\bwould\s+need\b|\bneed\s+to\s+know\b|"
    r"\bdepends\b|\bare\s+we\s+okay\b)",
    re.I,
)
# On sanctioned switch turns (the controller explicitly invited a vote change and
# asked for a bridge clause), only genuine prerequisites still block a commitment.
# Concessive riders ("as long as", "though", "but") are the requested bridge shape.
_SANCTION_BLOCK = re.compile(
    r"(?:\?|\bonly\s+if\b|\bunless\b|\bwould\s+need\b|\bneed\s+to\s+know\b|\bdepends\b)",
    re.I,
)
_REJECT = re.compile(
    r"\b(?:i\s+reject|i\s+can'?t\s+support|cannot\s+support|not\s+okay\s+with|not\s+fine\s+with|"
    r"dealbreaker|blocked|hard\s+no|no\s+for\s+me|i'?m\s+against)\b",
    re.I,
)
_SOFT_OBJECT = re.compile(
    r"\b(?:concern|worry|problem|issue|downside|too\s+expensive|too\s+far|too\s+late|risky|"
    r"not\s+ideal|doesn'?t\s+fit|would\s+be\s+hard)\b",
    re.I,
)


class OptionResolver:
    def __init__(self, options: list[OptionCard]) -> None:
        self.options = options
        self.by_id = {option.id: option for option in options}
        self.alias_to_id = self._build_aliases(options)

    @staticmethod
    def _build_aliases(options: list[OptionCard]) -> dict[str, str]:
        candidates: dict[str, set[str]] = {}
        safe_short_names = short_alias_map(options)
        for option in options:
            name = option.name.lower()
            aliases = {name}
            clean = re.sub(r"[^\wäöüÄÖÜß\s'-]", " ", name)
            words = [w for w in clean.split() if len(w) >= 4 and w not in _ALIAS_STOPWORDS]
            aliases.update(words)
            if len(words) >= 2:
                aliases.add(" ".join(words[:2]))
            # Distinctive proper nouns/brands are how people actually refer to an
            # option (e.g. "Gin", "Rails", "SAS", "FastAPI"); capture capitalized
            # tokens of length >= 3 from the original name even when < 4 chars.
            aliases.update(
                w.lower() for w in re.findall(r"[A-Z][A-Za-z]{2,}", option.name)
                if w.lower() not in _ALIAS_STOPWORDS
            )
            aliases.add(safe_short_names[option.id].lower())
            candidates[option.id] = {a.strip() for a in aliases if a.strip()}
        owners: dict[str, set[str]] = {}
        for option_id, aliases in candidates.items():
            for alias in aliases:
                owners.setdefault(alias, set()).add(option_id)
        return {alias: next(iter(ids)) for alias, ids in owners.items() if len(ids) == 1}

    def ids_in_text(self, text: str) -> list[str]:
        lower = text.lower()
        found: list[str] = []
        for option_id in self.by_id:
            patterns = [
                rf"\boption\s+{re.escape(option_id.lower())}\b",
                rf"\b{re.escape(option_id.lower())}\)\b",
            ]
            if any(re.search(pattern, lower) for pattern in patterns) and option_id not in found:
                found.append(option_id)
        for alias, option_id in sorted(self.alias_to_id.items(), key=lambda x: len(x[0]), reverse=True):
            if option_id in found:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                found.append(option_id)
        return found

    def invalid_option_refs(self, text: str) -> list[str]:
        valid = set(self.by_id)
        refs = re.findall(r"\bOption\s+([A-Za-z])\b", text)
        return sorted({r.upper() for r in refs if r.upper() not in valid})

    def option_text(self, option_id: str) -> str:
        return self.by_id[option_id].prompt_card()


def _option_positions(segment: str, resolver: OptionResolver) -> list[tuple[int, str]]:
    """Earliest match position per option in ``segment``, sorted by position."""
    lower = segment.lower()
    hits: dict[str, int] = {}
    for option_id in resolver.by_id:
        m = re.search(rf"\boption\s+{re.escape(option_id.lower())}\b", lower)
        if m:
            hits[option_id] = min(hits.get(option_id, 1 << 30), m.start())
    for alias, option_id in resolver.alias_to_id.items():
        m = re.search(rf"\b{re.escape(alias)}\b", lower)
        if m:
            hits[option_id] = min(hits.get(option_id, 1 << 30), m.start())
    return sorted((pos, oid) for oid, pos in hits.items())


# Canonical commitment-phrase families with display labels, used to stop one
# family ("Count me in for ...") dominating a vote round (issue #25).
_PHRASE_FAMILIES: list[tuple[str, re.Pattern]] = [
    ("count me in for", re.compile(r"\bcount\s+me\s+in\s+for\b", re.I)),
    ("I'm all in for", re.compile(r"\bi'?m\s+(?:all\s+)?in\s+for\b", re.I)),
    ("gets my vote", re.compile(r"\bgets?\s+my\s+vote\b", re.I)),
    ("my vote is", re.compile(r"\bmy\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to)\b", re.I)),
    ("I'd go with", re.compile(r"\bi'?(?:d|ll)\s+go\s+with\b", re.I)),
    ("I'm going with", re.compile(r"\bi'?m\s+going\s+with\b", re.I)),
    ("my pick is", re.compile(r"\bmy\s+pick\s+is\b|\bis\s+my\s+pick\b|\bthat'?s\s+my\s+pick\b", re.I)),
    ("my choice is", re.compile(r"\b(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+choice\b|\bmy\s+top\s+choice\s+is\b", re.I)),
    ("I vote for", re.compile(r"\bi\s+vote\s+for\b", re.I)),
    ("works for me", re.compile(r"\bworks\s+(?:best\s+)?for\s+me\b", re.I)),
    ("let's go with", re.compile(r"\blet'?s\s+go\s+with\b", re.I)),
    ("I'm sold on", re.compile(r"\bi'?m\s+sold\s+on\b", re.I)),
]


def used_commitment_phrases(texts: list[str]) -> list[str]:
    """Which commitment-phrase families appear in ``texts`` (display labels)."""
    return [label for label, pattern in _PHRASE_FAMILIES if any(pattern.search(t or "") for t in texts)]


def _commitment_object(check_text: str, resolver: OptionResolver) -> str | None:
    """Disambiguate a multi-option line by the words around the commitment phrase.

    "I'd go with Garden Beds to improve our neighborhood green space" resolves
    to two options on the full text (a generic token of another option's name
    appears in the reason clause), but the object of the commitment itself is
    unambiguous: the first option named after the phrase ("go with X ..."), or
    the nearest one before it ("X gets my vote"). A coordinated pair right at
    the object ("either X or Y", "X or Y") stays ambiguous.
    """
    match = _COMMIT.search(check_text)
    if not match:
        return None
    after = re.split(r"[.;!?]", check_text[match.end():])[0]
    before = re.split(r"[.;!?]", check_text[: match.start()])[-1]
    after_hits = _option_positions(after, resolver)
    before_hits = _option_positions(before, resolver)
    after_distance = after_hits[0][0] if after_hits else None
    before_distance = len(before) - before_hits[-1][0] if before_hits else None

    if after_distance is not None and (before_distance is None or after_distance <= before_distance):
        first_pos, first_id = after_hits[0]
        if re.search(r"\beither\b", after[:first_pos], re.I):
            return None
        others = [(pos, oid) for pos, oid in after_hits[1:] if oid != first_id]
        if others and others[0][0] - first_pos < 40 and re.search(
            r"\b(?:or|and|vs\.?|versus)\b", after[first_pos: others[0][0]], re.I
        ):
            return None
        return first_id
    if before_distance is not None:
        last_pos, last_id = before_hits[-1]
        others = [(pos, oid) for pos, oid in before_hits[:-1] if oid != last_id]
        if others and last_pos - others[-1][0] < 40 and re.search(
            r"\b(?:either|or|and|vs\.?|versus)\b", before[others[-1][0]: last_pos], re.I
        ):
            return None
        return last_id
    return None


def visible_commitment(
    text: str,
    resolver: OptionResolver,
    *,
    sanctioned_switch: bool = False,
) -> tuple[str, str] | None:
    """Return (stance, option_id) for clear visible commitments only.

    stance is one of ``vote``, ``accept``, or ``reject``. Ambiguous or hedged
    lines return None; false positives are worse than missed commitments here.

    ``sanctioned_switch`` marks a turn where the controller explicitly allowed a
    vote change and instructed a bridge clause ("commit AND say what makes it
    workable despite preferring X"). On those turns a concessive rider after the
    commitment ("as long as", "even though", "but") must not void the commitment
    (issue I2) — only questions and genuine prerequisites still block.
    """
    check_text = text.replace("’", "'").replace("‘", "'")
    ids = resolver.ids_in_text(check_text)
    if len(ids) > 1:
        near = _commitment_object(check_text, resolver)
        if near is None:
            return None
        ids = [near]
    if len(ids) != 1:
        return None
    option_id = ids[0]
    if _REJECT.search(check_text):
        return ("reject", option_id)
    if not _COMMIT.search(check_text):
        return None
    direct_vote = _DIRECT_VOTE.search(check_text)
    soft_commit = _SOFT_COMMIT.search(check_text)

    if sanctioned_switch:
        if _SANCTION_BLOCK.search(check_text):
            return None
        return ("vote" if direct_vote else "accept", option_id)

    # Conditional or question-like support is not a public final vote.
    # Example: "I can support A, but are we okay with the higher cost?"
    # remains unresolved until the speaker explicitly votes without conditions.
    if _HARD_CONDITIONAL.search(check_text):
        return None
    if soft_commit and (_HEDGE.search(check_text) or _CONDITIONAL_AFTER_COMMIT.search(check_text)):
        return None
    if _HEDGE.search(check_text) and not direct_vote:
        return None
    if _CONDITIONAL_AFTER_COMMIT.search(check_text) and not direct_vote:
        return None
    stance = "vote" if direct_vote else "accept"
    return (stance, option_id)


def parse_dialogue_act(
    *,
    speaker_id: str,
    speaker_name: str,
    text: str,
    resolver: OptionResolver,
    participant_names: dict[str, str],
    intent: MoveIntent | None = None,
    previous_speaker_id: str | None = None,
) -> DialogueAct:
    text = normalise_ws(text)
    check_text = text.replace("’", "'").replace("‘", "'")
    option_refs = resolver.ids_in_text(check_text)
    addressee_id = _extract_addressee(text, speaker_id, participant_names)
    if addressee_id is None and _QUESTION.search(text) and previous_speaker_id and previous_speaker_id != speaker_id:
        if re.search(r"\b(?:you|your|that|what\s+about|how\s+about)\b", text, re.I):
            addressee_id = previous_speaker_id
    question_target = addressee_id if _QUESTION.search(text) and addressee_id else None
    if question_target is None and _QUESTION.search(text) and _is_genuine_question(text):
        question_target = _best_respondent(speaker_id, previous_speaker_id, participant_names)

    act_type = intent.act if intent else (ActType.ASK if question_target else ActType.REACT)
    commitment = visible_commitment(
        text, resolver, sanctioned_switch=bool(intent and intent.allow_vote_change)
    )
    explicit_vote: str | None = None
    accepts: list[str] = []
    soft_rejects: dict[str, str] = {}
    hard_rejects: dict[str, str] = {}
    proposes_option: str | None = None

    if commitment:
        stance, option_id = commitment
        if stance in {"vote", "accept"}:
            explicit_vote = option_id
            if stance == "accept":
                accepts.append(option_id)
            act_type = ActType.VOTE if stance == "vote" else ActType.ACCEPT
        elif stance == "reject":
            soft_rejects[option_id] = text
            act_type = ActType.REJECT
    elif option_refs:
        if _REJECT.search(check_text) or _SOFT_OBJECT.search(check_text):
            soft_rejects[option_refs[0]] = text
        if act_type == ActType.PROPOSE_COMPROMISE and len(option_refs) == 1:
            proposes_option = option_refs[0]

    return DialogueAct(
        speaker_id=speaker_id,
        text=text,
        act_type=act_type,
        option_refs=option_refs,
        addressee_id=addressee_id,
        question_target_id=question_target,
        explicit_vote=explicit_vote,
        accepts=accepts,
        soft_rejects=soft_rejects,
        hard_rejects=hard_rejects,
        proposes_option=proposes_option,
    )


def has_commitment_hedge(text: str) -> bool:
    return bool(_HEDGE.search(text))


def _is_genuine_question(text: str) -> bool:
    if _RHETORICAL_TAIL.search(text):
        return False
    return bool(_GENUINE_QUESTION.search(text))


def _best_respondent(
    speaker_id: str,
    previous_speaker_id: str | None,
    participant_names: dict[str, str],
) -> str | None:
    others = [pid for pid in participant_names if pid != speaker_id]
    if previous_speaker_id and previous_speaker_id in others:
        return previous_speaker_id
    return others[0] if others else None


def _extract_addressee(text: str, speaker_id: str, participant_names: dict[str, str]) -> str | None:
    lower = text.lower()
    for pid, name in participant_names.items():
        if pid == speaker_id:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return pid
    return None
