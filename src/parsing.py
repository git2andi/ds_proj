"""Pure visible-semantics layer: text in, DialogueAct out (contract 4.4).

Final outcomes must be based on what the transcript visibly says. This parser
is therefore conservative: it only records a vote/acceptance when one option is
mentioned unambiguously and the utterance contains a clear commitment phrase.
It never mutates dialogue state and never makes controller decisions — a group
question gets a scope, not a respondent.
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
    "original", "plus", "select", "basic", "classic", "standard", "premium",
    "table", "room", "house", "place", "option", "choice", "pick",
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
    r"\b(?:i\s+vote\s+for|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to|\s+stays\s+(?:with|on))|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i'?m\s+choosing|i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+pick\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"i'?m\s+still\s+on\s+(?!the\s+fence)|i'?ll\s+stay\s+(?:with|on)|i'?ll\s+back\s+(?!down|off|out|up)|"
    r"gets?\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|"
    r"let'?s\s+go\s+with|we\s+should\s+go\s+with|go\s+with|settle\s+on|pick|choose|"
    r"i'?(?:d|ll)\s+switch\s+to|i\s+can\s+live\s+with|i'?d\s+be\s+happy\s+with|"
    r"i\s+support|i\s+accept|i\s+can\s+support|i'?m\s+fine\s+with|fine\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|i'?m\s+okay\s+with|okay\s+with|agree\s+on|final\s+choice)\b",
    re.I,
)
_SOFT_COMMIT = re.compile(
    r"\b(?:i\s+can\s+support|i\s+support|i\s+accept|i'?m\s+fine\s+with|fine\s+with|"
    r"i\s+can\s+live\s+with|i'?d\s+be\s+happy\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|i'?m\s+okay\s+with|okay\s+with|agree\s+on)\b",
    re.I,
)
_DIRECT_VOTE = re.compile(
    r"\b(?:i\s+vote\s+for|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to|\s+stays\s+(?:with|on))|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i'?m\s+choosing|i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+pick\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"i'?m\s+still\s+on\s+(?!the\s+fence)|i'?ll\s+stay\s+(?:with|on)|i'?ll\s+back\s+(?!down|off|out|up)|"
    r"gets?\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|"
    r"i'?(?:d|ll)\s+switch\s+to|"
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
    r"\b(?:concern(?:s|ed)?|worr(?:y|ies|ied)|bother(?:s|ed)?\s+me|problems?|issues?|downsides?|"
    r"too\s+expensive|too\s+far|too\s+late|too\s+(?:pricey|costly)|risky|"
    r"(?:a\s+bit|too|rather|quite|pretty)\s+steep|"
    r"(?:seems?|looks?|feels?)\s+(?:too\s+|quite\s+|pretty\s+|a\s+bit\s+)?(?:high|expensive|pricey|risky|steep)|"
    r"not\s+ideal|doesn'?t\s+fit|would\s+be\s+hard)\b",
    re.I,
)

# --- I3 vocabulary: active blockers, resolutions, compromise offers, reasons ---

# Strong option-tied vetoes. Stronger than _SOFT_OBJECT: while active, the sim
# cannot vote for the option without a visible resolution (consumed in I4).
_ACTIVE_BLOCKER = re.compile(
    r"\b(?:dealbreaker|deal\s+breaker|hard\s+no|hard\s+pass|"
    r"i\s+can'?t\s+support|i\s+cannot\s+support|i\s+won'?t\s+support|"
    r"not\s+okay\s+with|not\s+fine\s+with|doesn'?t\s+work\s+for\s+me|"
    r"can'?t\s+get\s+behind|i'?m\s+out\s+on|non-?starter)\b",
    re.I,
)
# Noun-shaped triggers get a negation guard ("that's not a dealbreaker").
_BLOCKER_NOUN = re.compile(r"\b(?:dealbreaker|deal\s+breaker|hard\s+no|hard\s+pass|non-?starter)\b", re.I)
_BLOCKER_NEGATION = re.compile(
    r"(?:\bnot\b|\bno\s+longer\b|\bisn'?t\b|\bwasn'?t\b|\bhardly\b|\bwouldn'?t\s+be\b|\bfar\s+from\b)"
    r"\s+(?:really\s+|quite\s+|exactly\s+|necessarily\s+)?(?:a\s+|the\s+)?$",
    re.I,
)
# A speculative or other-directed blocker mention is NOT the speaker's own veto:
# "the fixed height might be a dealbreaker for some teammates" raises a concern,
# it does not bind the speaker. Personal markers ("for me/us") override the
# speculation guard; false blockers are far worse than missed ones because a
# recorded blocker hard-binds later votes.
_BLOCKER_SPECULATIVE = re.compile(
    r"\b(?:might|may|could|would|whether|wonder(?:ing)?|worr(?:y|ied|ies))\b", re.I
)
_BLOCKER_OTHER_DIRECTED = re.compile(
    r"\bfor\s+(?:some(?:one|body)?|others?|any(?:one|body)\s+else|taller|shorter|new(?:er)?\s+\w+|"
    r"the\s+others|teammates?|the\s+team|the\s+group|folks|people|him|her|them)\b",
    re.I,
)
_BLOCKER_PERSONAL = re.compile(r"\bfor\s+(?:me|us)\b|\bi'?m\s+out\b|\bi\s+(?:can'?t|cannot|won'?t)\b", re.I)
# A blocker counts as resolved only when the same sim visibly says so, with no
# hard-conditional residue ("if we…", "only if…") left in the line.
# _RESOLUTION_HEAD phrases explicitly reference the resolved concern; when one is
# present, the mention of "concern/worry/issue" must not re-trip the conditional
# guard in visible_commitment ("That fixes my concern; I can live with X").
_RESOLUTION_HEAD = re.compile(
    r"\b(?:that\s+(?:addresses|fixes|solves|resolves|settles|covers|handles)\s+(?:my|the|it)|"
    r"my\s+(?:concern|worry|issue)\s+is\s+(?:addressed|resolved|gone|covered|handled)|"
    r"no\s+longer\s+a\s+(?:dealbreaker|problem|blocker|non-?starter)|"
    r"not\s+a\s+(?:dealbreaker|problem|blocker)\s+anymore|i'?m\s+okay\s+with\s+it\s+now)\b",
    re.I,
)
_RESOLUTION = re.compile(
    rf"{_RESOLUTION_HEAD.pattern}|\bi\s+can\s+live\s+with\b",
    re.I,
)
# Visible compromise proposals, including question forms ("could we all live with X?").
_COMPROMISE_OFFER = re.compile(
    r"\b(?:could|can|would)\s+(?:we|everyone|you\s+all|y'?all)\s+(?:all\s+)?live\s+with\b|"
    r"\bwhat\s+if\s+we\s+(?:went|go|all\s+went|all\s+go)\s+with\b|"
    r"\bmeet\s+in\s+the\s+middle\s+(?:on|with|at)\b|"
    r"\bwould\s+(?:that|this|it)\s+work\s+for\s+everyone\b|"
    r"\bas\s+a\s+(?:compromise|middle\s+ground)\b",
    re.I,
)
# Visible softening: the speaker says another option is winning them over
# without committing ("B is starting to make more sense to me"). Moves the
# latent lean (issue 3) but is hedged wording — it never parses as a vote.
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


def softening_option(check_text: str, resolver: OptionResolver) -> str | None:
    """Option the line visibly softens toward, if any (issue 3)."""
    match = _SOFTENING.search(check_text)
    if not match:
        return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


# Does a commitment carry a visible reason clause? (Consumed by I4: a switch
# away from the initial preference needs a stated reason.)
_REASON_MARKER = re.compile(
    r"\b(?:because|since|despite|even\s+though|"
    r"for\s+(?:its|their|the|a|an)\b|to\s+(?:keep|boost|get|make|support|showcase|save|avoid|stay)\b|"
    r"so\s+(?:we|it|everyone|the)\b|"
    r"it\s+(?:solves|fixes|keeps|gives|covers|fits|means|delivers|works|saves|hits))\b|"
    r"\bbut\b|—|;|:\s",
    re.I,
)


def _nearest_option(check_text: str, start: int, end: int, resolver: OptionResolver) -> str | None:
    """The option mentioned closest to a phrase match, within its sentence."""
    after = re.split(r"[.;!?]", check_text[end:])[0]
    before = re.split(r"[.;!?]", check_text[:start])[-1]
    after_hits = _option_positions(after, resolver)
    before_hits = _option_positions(before, resolver)
    after_distance = after_hits[0][0] if after_hits else None
    before_distance = len(before) - before_hits[-1][0] if before_hits else None
    if after_distance is not None and (before_distance is None or after_distance <= before_distance):
        return after_hits[0][1]
    if before_distance is not None:
        return before_hits[-1][1]
    # Fall back to a unique option mention anywhere in the line.
    ids = resolver.ids_in_text(check_text)
    return ids[0] if len(ids) == 1 else None


def active_blocker_option(check_text: str, resolver: OptionResolver) -> str | None:
    """Option the line visibly vetoes ('X is a dealbreaker for me'), if any."""
    match = _ACTIVE_BLOCKER.search(check_text)
    if not match:
        return None
    if _BLOCKER_NOUN.match(match.group(0)) and _BLOCKER_NEGATION.search(check_text[: match.start()]):
        return None
    # Only a personal, non-speculative veto binds the speaker. Judge within the
    # sentence containing the trigger so an unrelated hedge elsewhere is ignored.
    sentence_start = check_text.rfind(".", 0, match.start())
    sentence_end_match = re.search(r"[.;!?]", check_text[match.end():])
    sentence_end = match.end() + (sentence_end_match.start() if sentence_end_match else len(check_text))
    sentence = check_text[sentence_start + 1: sentence_end]
    if not _BLOCKER_PERSONAL.search(sentence):
        if _BLOCKER_SPECULATIVE.search(sentence) or _BLOCKER_OTHER_DIRECTED.search(sentence):
            return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


def blocker_resolution_option(check_text: str, resolver: OptionResolver) -> str | None:
    """Option whose earlier blocker this line visibly resolves, if any."""
    match = _RESOLUTION.search(check_text)
    if not match:
        return None
    if _HARD_CONDITIONAL.search(check_text):
        return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


def conditional_support_option(check_text: str, resolver: OptionResolver) -> str | None:
    """Option supported only conditionally ('I can support A, but only if…')."""
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
    """Option visibly proposed as common ground, including in question form."""
    match = _COMPROMISE_OFFER.search(check_text)
    if not match:
        return None
    return _nearest_option(check_text, match.start(), match.end(), resolver)


def commitment_has_reason(text: str) -> bool:
    """True when a commitment line carries a visible reason clause."""
    return bool(_REASON_MARKER.search(text.replace("’", "'")))


# A concession/movement marker: the line explicitly signals moving off an earlier
# stance ("I still like X, but…", "even though…", "coming around", "I can live
# with…"). Used with commitment_has_reason to enforce a visible bridge clause on
# preference switches (issue 5).
_CONCESSION = re.compile(
    r"\b(?:still|even\s+though|though|although|despite|"
    r"prefer(?:red|ring|s)?|i'?d\s+rather|rather\s+than|"
    r"used\s+to|originally|at\s+first|was\s+leaning|leaned|"
    r"i\s+liked|i\s+wanted|coming\s+around|come\s+around|won\s+over|"
    r"give\s+up|giving\s+up|gave\s+up|let\s+go\s+of|concede|conceding|"
    r"i\s+can\s+live\s+with|change[ds]?\s+my\s+mind|for\s+the\s+group|"
    r"easier\s+for\s+(?:the\s+group|everyone)|for\s+the\s+sake\s+of)\b",
    re.I,
)


def switch_bridge_ok(text: str, old_option_id: str, resolver: OptionResolver) -> bool:
    """True when a switch line visibly bridges the old stance to the new pick.

    A socially honest preference switch (issue 5) must carry both (a) a link to
    the old stance — the old option named, or an explicit concession marker —
    and (b) a reason for the movement. The new option is already present (it is
    the committed vote), so it is not re-checked here.
    """
    normalised = text.replace("’", "'").replace("‘", "'")
    if not commitment_has_reason(normalised):
        return False
    if _CONCESSION.search(normalised):
        return True
    return old_option_id in resolver.ids_in_text(normalised)


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
            # Full names, safe short names, and multi-word heads are reliable.
            # Single lowercase words from option names are often ordinary dialogue
            # vocabulary (e.g. "original pick" falsely matching "Senseo Original
            # Plus"), so we rely on distinctive capitalized tokens below instead.
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


# Pure connector between two option mentions — the "single plan out of two
# options" shape (P6). Comparisons and concessions put real words between the
# two mentions and do not match.
_BLEND_BETWEEN = re.compile(
    r"^\W*(?:and(?:\s+also)?(?:\s+(?:use|do|book|add|visit|try|include))?|plus|"
    r"combined?\s+with|mix(?:ed)?\s+with|together\s+with|along\s+with|with)"
    r"\s*(?:the\s+|some\s+of\s+)?\W*$",
    re.I,
)


def hybrid_blend_detected(text: str, resolver: OptionResolver) -> bool:
    """True when one utterance welds two different options into a single plan
    (P6): "Pine Ridge and also Willow Creek", "the class combined with the
    escape room". Used only on compromise-type turns, where a coordinated
    pair is exactly the implicit-new-option failure."""
    lower = text.replace("’", "'").lower()
    spans: list[tuple[int, int, str]] = []
    for option_id in resolver.by_id:
        for m in re.finditer(rf"\boption\s+{re.escape(option_id.lower())}\b", lower):
            spans.append((m.start(), m.end(), option_id))
    for alias, option_id in resolver.alias_to_id.items():
        for m in re.finditer(rf"\b{re.escape(alias)}\b", lower):
            spans.append((m.start(), m.end(), option_id))
    spans.sort()
    for (_s1, e1, id1), (s2, _e2, id2) in zip(spans, spans[1:]):
        if id1 == id2 or s2 <= e1:
            continue
        between = lower[e1:s2]
        if len(between) <= 30 and _BLEND_BETWEEN.match(between):
            return True
    return False


# Canonical commitment-phrase families with display labels, used to stop one
# family ("Count me in for ...") dominating a vote round (issue #25).
_PHRASE_FAMILIES: list[tuple[str, re.Pattern]] = [
    ("count me in for", re.compile(r"\bcount\s+me\s+in\s+for\b", re.I)),
    ("I'm all in for", re.compile(r"\bi'?m\s+(?:all\s+)?in\s+for\b", re.I)),
    ("gets my vote", re.compile(r"\bgets?\s+my\s+vote\b", re.I)),
    ("my vote is", re.compile(r"\bmy\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to)\b", re.I)),
    ("I'd go with", re.compile(r"\bi'?(?:d|ll)\s+go\s+with\b", re.I)),
    ("I'm going with", re.compile(r"\bi'?m\s+going\s+with\b", re.I)),
    ("I'm choosing", re.compile(r"\bi'?m\s+choosing\b|\bi\s+choose\b", re.I)),
    ("my pick is", re.compile(r"\bmy\s+pick\s+is\b|\bis\s+my\s+pick\b|\bthat'?s\s+my\s+pick\b", re.I)),
    ("my choice is", re.compile(r"\b(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+choice\b|\bmy\s+top\s+choice\s+is\b", re.I)),
    ("I vote for", re.compile(r"\bi\s+vote\s+for\b", re.I)),
    ("I'm still on", re.compile(r"\bi'?m\s+still\s+on\b|\bmy\s+vote\s+stays\b", re.I)),
    ("I'll stay with", re.compile(r"\bi'?ll\s+stay\s+(?:with|on)\b", re.I)),
    ("I'll back", re.compile(r"\bi'?ll\s+back\b", re.I)),
    ("I can live with", re.compile(r"\bi\s+can\s+live\s+with\b", re.I)),
    ("I'd be happy with", re.compile(r"\bi'?d\s+be\s+happy\s+with\b", re.I)),
    ("works for me", re.compile(r"\bworks\s+(?:best\s+)?for\s+me\b", re.I)),
    ("let's go with", re.compile(r"\blet'?s\s+go\s+with\b", re.I)),
    ("I'm sold on", re.compile(r"\bi'?m\s+sold\s+on\b", re.I)),
]


def used_commitment_phrases(texts: list[str]) -> list[str]:
    """Which commitment-phrase families appear in ``texts`` (display labels)."""
    return [label for label, pattern in _PHRASE_FAMILIES if any(pattern.search(t or "") for t in texts)]


def unused_commitment_phrases(avoid: list[str], limit: int = 3) -> list[str]:
    """Parser-recognized commitment phrasings not yet used this vote round (P9):
    suggesting these keeps LLM vote lines inside the vocabulary the observer
    can read, instead of pushing later voters into unparseable variety."""
    avoid_set = {a.lower() for a in avoid}
    return [label for label, _pattern in _PHRASE_FAMILIES if label.lower() not in avoid_set][:limit]


# Commitment phrases whose grammatical object comes before the phrase
# ("X gets my vote", "X works for me", "X is my pick").
_SUBJECT_FORM_COMMIT = re.compile(
    r"(?:gets?\s+my\s+vote|works\s+(?:best\s+)?for\s+me|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick))",
    re.I,
)

# Clause boundaries for commitment-object scoping: sentence punctuation plus
# em/en dashes and spaced hyphens, which introduce commentary clauses.
_CLAUSE_SPLIT = re.compile(r"[.;!?—–]|\s-{1,2}\s")

# Reason clause following a commitment ("… for its inclusive menu", "… because
# it solves the timing issue"). Used to stop voters echoing each other's
# justification word-for-word (issue I12).
_REASON_TAIL = re.compile(
    r"\b(?:because|since|for\s+(?:its|their|the|a|an)|"
    r"to\s+(?:keep|boost|get|make|support|avoid|stay|showcase|promote|improve))\b\s*.{8,70}",
    re.I,
)


def round_reason_snippets(texts: list[str]) -> list[str]:
    """Reason clauses already used in this round's commitments (max 10 words each)."""
    snippets: list[str] = []
    for text in texts:
        match = _REASON_TAIL.search(text or "")
        if match:
            snippet = match.group(0).strip().rstrip(".!?,;")
            snippets.append(" ".join(snippet.split()[:10]))
    return snippets


def _commitment_object(check_text: str, resolver: OptionResolver) -> str | None:
    """Disambiguate a multi-option line by the words around the commitment phrase.

    "I'd go with Garden Beds to improve our neighborhood green space" resolves
    to two options on the full text (a generic token of another option's name
    appears in the reason clause), but the object of the commitment itself is
    unambiguous: the first option named after the phrase ("go with X ..."), or
    the nearest one before it ("X gets my vote"). A coordinated pair right at
    the object ("either X or Y", "X or Y") stays ambiguous.

    Clauses split on dashes as well as sentence punctuation, and subject-form
    phrases ("X gets my vote", "X works for me") bind to the option BEFORE the
    phrase: in "Trello still gets my vote — ClickUp hasn't fixed my concern"
    the rival named after the dash is commentary, not the vote object.
    """
    match = _COMMIT.search(check_text)
    if not match:
        return None
    after = _CLAUSE_SPLIT.split(check_text[match.end():])[0]
    before = _CLAUSE_SPLIT.split(check_text[: match.start()])[-1]
    after_hits = _option_positions(after, resolver)
    before_hits = _option_positions(before, resolver)
    after_distance = after_hits[0][0] if after_hits else None
    before_distance = len(before) - before_hits[-1][0] if before_hits else None
    if before_hits and _SUBJECT_FORM_COMMIT.match(match.group(0)):
        after_distance = None

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
    # An explicit blocker-resolution head names the old concern; that mention
    # must not count as a fresh condition ("That fixes my concern; I can live
    # with X" is an acceptance).
    resolution_head = bool(_RESOLUTION_HEAD.search(check_text))
    if soft_commit and not resolution_head and (_HEDGE.search(check_text) or _CONDITIONAL_AFTER_COMMIT.search(check_text)):
        return None
    if _HEDGE.search(check_text) and not direct_vote and not resolution_head:
        return None
    if _CONDITIONAL_AFTER_COMMIT.search(check_text) and not direct_vote and not resolution_head:
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
    # Question scope from visible text only (4.4/5.5): a named or "you"-directed
    # question is direct; a genuine question without an addressee is a group
    # question with NO target — the controller assigns the respondent later.
    question_scope: str | None = None
    question_target = None
    if _QUESTION.search(text):
        if addressee_id:
            question_scope = "direct"
            question_target = addressee_id
        elif _is_genuine_question(text):
            question_scope = "group"

    commitment = visible_commitment(
        text, resolver, sanctioned_switch=bool(intent and intent.allow_vote_change)
    )
    explicit_vote: str | None = None
    accepts: list[str] = []
    soft_rejects: dict[str, str] = {}
    hard_rejects: dict[str, str] = {}

    if commitment:
        stance, option_id = commitment
        if stance in {"vote", "accept"}:
            explicit_vote = option_id
            if stance == "accept":
                accepts.append(option_id)
        elif stance == "reject":
            soft_rejects[option_id] = text
    elif option_refs:
        if _REJECT.search(check_text) or _SOFT_OBJECT.search(check_text):
            soft_rejects[option_refs[0]] = text

    blocker = active_blocker_option(check_text, resolver)
    if blocker and blocker != explicit_vote:
        hard_rejects[blocker] = text
    resolves_blocker = blocker_resolution_option(check_text, resolver)
    if resolves_blocker in hard_rejects:
        resolves_blocker = None  # one line cannot both raise and resolve a blocker
    conditional_support = None if commitment else conditional_support_option(check_text, resolver)
    offers_compromise = compromise_offer_option(check_text, resolver)
    softens_toward = None if commitment else softening_option(check_text, resolver)

    act_type = _realized_act_type(
        intent=intent,
        commitment=commitment,
        soft_rejects=soft_rejects,
        hard_rejects=hard_rejects,
        question_scope=question_scope,
        option_refs=option_refs,
        check_text=check_text,
    )

    return DialogueAct(
        speaker_id=speaker_id,
        text=text,
        act_type=act_type,
        option_refs=option_refs,
        addressee_id=addressee_id,
        question_scope=question_scope,  # type: ignore[arg-type]
        question_target_id=question_target,
        explicit_vote=explicit_vote,
        accepts=accepts,
        soft_rejects=soft_rejects,
        hard_rejects=hard_rejects,
        resolves_blocker=resolves_blocker,
        conditional_support=conditional_support,
        offers_compromise=offers_compromise,
        softens_toward=softens_toward,
    )


# Contextual acts keep their routed label: visible text alone cannot tell an
# answer, opening, procedure beat, compromise test, vote turn, or closing line
# apart from a plain statement — the surrounding controller context defines
# them. Ordinary acts must earn their label from text.
_CONTEXTUAL_ACTS = {
    ActType.OPENING,
    ActType.ANSWER,
    ActType.PROCESS,
    ActType.COMPROMISE,
    ActType.VOTE,
    ActType.CLOSING,
}

# Comparative wording that marks a genuine head-to-head, not two mentions.
_COMPARATIVE = re.compile(
    r"\b(?:than|versus|vs\.?|compared?\s+(?:to|with)|instead\s+of|over\b|"
    r"rather\s+than|beats?|wins?\s+(?:over|against)|side\s+by\s+side|trade-?off)\b",
    re.I,
)

# A statement realizes SUPPORT only when it visibly claims a benefit of a named
# option ("the Museum keeps the day easy"); a mere mention stays a comment.
_PRO_CLAIM = re.compile(
    r"\b(?:solves|fixes|keeps|gives|covers|fits|means|delivers|works|saves|hits|"
    r"offers|helps|suits)\b|\bi\s+(?:really\s+)?(?:like|love|prefer)\b",
    re.I,
)


def realized_comparison(text: str, resolver: OptionResolver) -> bool:
    """True when the line visibly contrasts two named options — the single
    textual authority for comparison threads, independent of the act label
    (a comparative question realizes ASK *and* a comparison)."""
    check = text.replace("’", "'").replace("‘", "'")
    return len(set(resolver.ids_in_text(check))) >= 2 and bool(_COMPARATIVE.search(check))


def _realized_act_type(
    *,
    intent: MoveIntent | None,
    commitment: tuple[str, str] | None,
    soft_rejects: dict[str, str],
    hard_rejects: dict[str, str],
    question_scope: str | None,
    option_refs: list[str],
    check_text: str,
) -> ActType:
    """Realized act from visible text: what the final line actually did.

    Precedence: commitment > objection > question > comparison > benefit
    claim. A routed ordinary act (support/concern/ask/compare/comment) that
    shows none of these signals degrades to comment — the routed intent never
    labels state on its own.
    """
    if commitment and commitment[0] in {"vote", "accept"}:
        return ActType.VOTE
    if commitment or hard_rejects or soft_rejects:
        return ActType.CONCERN
    if question_scope:
        return ActType.ASK
    if len(set(option_refs)) >= 2 and _COMPARATIVE.search(check_text):
        return ActType.COMPARE
    if intent is not None and intent.act in _CONTEXTUAL_ACTS:
        return intent.act
    if option_refs and _PRO_CLAIM.search(check_text):
        return ActType.SUPPORT
    return ActType.COMMENT


def _is_genuine_question(text: str) -> bool:
    if _RHETORICAL_TAIL.search(text):
        return False
    return bool(_GENUINE_QUESTION.search(text))


def _extract_addressee(text: str, speaker_id: str, participant_names: dict[str, str]) -> str | None:
    lower = text.lower()
    for pid, name in participant_names.items():
        if pid == speaker_id:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return pid
    return None
