"""Deterministic critical parser and resolver — never a semantic interpreter.

Retained responsibilities: option/alias/letter resolution with exact spans,
participant/addressee resolution, public-context reference resolution, strict
visible commitment/vote detection with post-checks, strict hard-blocker
detection, genuine-question detection, and the structural helpers grounding
relies on. Soft natural-language semantics (support, ordinary concerns,
comparisons, softening, proposals) are extracted conservatively by the
deterministic interpreter when needed for routing/state. This module never
mutates dialogue state and never makes controller
decisions — a group question gets a scope, not a respondent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from aliases import _GENERIC, _STOPWORDS, short_alias_map
from models import EvidenceSpan, OptionCard, OptionMention

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
    # Generic time-of-day / calendar words occur constantly in normal chat.
    # Treating a capitalized token from an option name such as "Home Cooking
    # Night" as the standalone alias "night" made "Saturday night" resolve
    # to that option and then poisoned grounding attribution.
    "day", "days", "night", "nights", "afternoon", "afternoons",
}

_QUESTION = re.compile(r"\?")
# Rhetorical / tag check-ins that end a statement without opening a real
# question thread: ", right?", ", isn't it?", ", don't we?", ", you know?".
_RHETORICAL_TAIL = re.compile(
    r",\s*(?:right|yeah|yep|no|huh|eh|ok|okay|you\s+know|don'?t\s+you\s+think"
    r"|(?:is|are|was|were|do|does|did|can|could|would|will|won|has|have|should)(?:n'?t)?"
    r"\s+(?:it|they|we|you|that|this|he|she))\s*\?\s*$",
    re.I,
)
# Small grammatical question detector (item 2): a WH word or an auxiliary verb
# opening a clause — at the string start or right after a clause boundary or
# short discourse lead-in. Combined with the `?` gate this recognises ordinary
# aux-led and WH questions with no option- or endpoint-specific phrasing.
_WH_LEAD = r"how|what'?s?|whats|where|when|which|who(?:m|se)?|why"
_AUX_LEAD = (
    r"do|does|did|is|are|am|was|were|can|could|would|will|won'?t|shall|should"
    r"|has|have|had|may|might|must|ain'?t"
    r"|isn'?t|aren'?t|wasn'?t|weren'?t|can'?t|couldn'?t|wouldn'?t|shouldn'?t"
    r"|don'?t|doesn'?t|didn'?t|hasn'?t|haven'?t|hadn'?t"
)
_QUESTION_CLAUSE = re.compile(
    r"(?:^|[.;:!?—–]\s*|,\s+|\b(?:so|and|but|or|well|hey|ok|okay|now|then|hmm|wait)\s+)"
    r"(?:" + _WH_LEAD + r"|(?:" + _AUX_LEAD + r"))\b",
    re.I,
)
# Short "A or B?" style option-choice question.
_CHOICE_QUESTION = re.compile(r"^[^?]{0,70}\bor\b[^?]{0,40}\?", re.I)
# Basic comparison signals (item 3). A relational connective explicitly ties two
# spans ("than", "versus", "compared to/with"); a contrast connective separates
# two clauses; a comparative adjective/adverb marks a comparative claim. These
# are grammatical, not option- or endpoint-specific.
_RELATIONAL = re.compile(r"\b(?:than|versus|vs\.?|compared\s+(?:to|with))\b", re.I)
_CONTRAST = re.compile(
    r"\b(?:while|whereas|but|however|though|although|on\s+the\s+other\s+hand)\b", re.I
)
_COMPARATIVE_ADJ = re.compile(
    r"\b(?:more|less|cheaper|pricier|costlier|dearer|faster|slower|bigger|larger|"
    r"smaller|closer|nearer|farther|further|longer|shorter|easier|harder|simpler|"
    r"higher|lower|stronger|weaker|safer|riskier|nicer|better|worse|"
    r"double|triple|twice|half)\b",
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
    r"\b(?:i\s+vote(?!\s+(?:against|no\b))(?:\s+for)?|i'?m\s+voting(?:\s+for)?|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to|\s+stays\s+(?:with|on))|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i(?:'?m|\s+am)\s+(?:choosing|picking)|i\s+(?:choose|pick)|i(?:'?m|\s+am)\s+(?:opting\s+for|settling\s+on|landing\s+on|committing\s+to|set\s+on)|i\s+(?:land\s+on|commit\s+to)|i'?ll\s+(?:take|pick|choose)\b|i'?ll\s+back\s+(?!down|off|out|up)|i'?ll\s+commit\s+to|i\s+back\s+(?!down|off|out|up)|i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+(?:final\s+)?(?:pick|choice)\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"i'?m\s+still\s+on\s+(?!the\s+fence)|i'?ll\s+stay\s+(?:with|on)|i'?ll\s+back\s+(?!down|off|out|up)|"
    r"(?:gets?|has)\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|is\s+where\s+i\s+land|"
    r"is\s+the\s+(?:right|clear|best|obvious)\s+choice|i'?m\s+(?:firmly|fully|totally)\s+with|"
    r"let'?s\s+go\s+with|we\s+should\s+go\s+with|go\s+with|settle\s+on|pick|choose|"
    r"i'?(?:d|ll|m)\s+switch(?:ing)?\s+to|switch(?:ing)?\s+from\s+[^.;,!?]{0,40}?\s+to\b|"
    r"i\s+can\s+live\s+with|i'?d\s+be\s+happy\s+with|"
    r"i\s+support|i\s+accept|i\s+can\s+support|i'?m\s+fine\s+with|fine\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|that\s+settles\s+it\s+for\s+me|i'?m\s+on\s+board\s+with|i'?m\s+okay\s+with|okay\s+with|agree\s+on|final\s+choice)\b",
    re.I,
)
_SOFT_COMMIT = re.compile(
    r"\b(?:i\s+can\s+support|i\s+support|i\s+accept|i'?m\s+fine\s+with|fine\s+with|"
    r"i\s+can\s+live\s+with|i'?d\s+be\s+happy\s+with|"
    r"works\s+(?:best\s+)?for\s+me|that\s+works|that\s+settles\s+it\s+for\s+me|i'?m\s+on\s+board\s+with|i'?m\s+okay\s+with|okay\s+with|agree\s+on)\b",
    re.I,
)
_DIRECT_VOTE = re.compile(
    r"\b(?:i\s+vote(?!\s+(?:against|no\b))(?:\s+for)?|i'?m\s+voting(?:\s+for)?|my\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to|\s+stays\s+(?:with|on))|i\s+choose|i'd\s+choose|i\s+would\s+choose|"
    r"i(?:'?m|\s+am)\s+(?:choosing|picking)|i\s+(?:choose|pick)|i(?:'?m|\s+am)\s+(?:opting\s+for|settling\s+on|landing\s+on|committing\s+to|set\s+on)|i\s+(?:land\s+on|commit\s+to)|i'?ll\s+(?:take|pick|choose)\b|i'?ll\s+back\s+(?!down|off|out|up)|i'?ll\s+commit\s+to|i\s+back\s+(?!down|off|out|up)|i'?d\s+go\s+with|i'?ll\s+go\s+with|i'?m\s+going\s+with|my\s+(?:final\s+)?(?:pick|choice)\s+is|"
    r"i'?m\s+(?:all\s+)?in\s+for|count\s+me\s+in\s+for|"
    r"i'?m\s+still\s+on\s+(?!the\s+fence)|i'?ll\s+stay\s+(?:with|on)|i'?ll\s+back\s+(?!down|off|out|up)|"
    r"(?:gets?|has)\s+my\s+vote|my\s+top\s+(?:choice|pick)\s+is|i'?m\s+sold\s+on|i'?m\s+(?:all\s+)?for\b|let'?s\s+(?:do|book|get)\b|"
    r"(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+(?:choice|pick)|is\s+where\s+i\s+land|"
    r"is\s+the\s+(?:right|clear|best|obvious)\s+choice|"
    r"i'?(?:d|ll|m)\s+switch(?:ing)?\s+to|switch(?:ing)?\s+from\s+[^.;,!?]{0,40}?\s+to\b|"
    r"let'?s\s+go\s+with|we\s+should\s+go\s+with|settle\s+on|final\s+choice)\b",
    re.I,
)
_CONDITIONAL_AFTER_COMMIT = re.compile(
    r"\b(?:but|however|though|although|still|only\s+if|if\s+we|if\s+it|as\s+long\s+as|"
    r"provided\s+that|are\s+we\s+okay|concern|worry|problem|issue|not\s+sure|unless)\b",
    re.I,
)
_HARD_CONDITIONAL = re.compile(
    r"(?:\?|\bonly\s+if\b|\bif\b|\bas\s+long\s+as\b|"
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


def has_commitment_phrase(text: str) -> bool:
    """Whether any commitment-family wording occurs at all — a cheap guard
    used by the selective fast paths (a line with commitment wording is never
    'non-state-changing')."""
    return bool(_COMMIT.search(text.replace("’", "'")))


def has_implicit_reference(text: str) -> bool:
    """Whether the line contains an implicit option reference shape ("it is",
    "that works", "the former") that could bind through public context."""
    return bool(_IMPLICIT_REF.search(text))


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

    def mentions(self, text: str) -> list[OptionMention]:
        """Every explicit option mention in ``text``, in visible text order.

        The one canonical resolver result (todo_validation item 5): each
        mention carries the option id, its exact span, the alias form used,
        and its textual order. Overlapping matches keep the longest span.
        Bare single-letter labels resolve case-sensitively and only for
        letters that are not ordinary English words ("B has my vote" resolves;
        the article in "A museum day" never does — "A" needs "Option A"/"A)").
        """
        lower = text.lower()
        raw: list[tuple[int, int, str, str]] = []  # start, end, option_id, alias_form
        for option_id in self.by_id:
            escaped = re.escape(option_id.lower())
            for pattern in (rf"\boption\s+{escaped}\b", rf"\b{escaped}\)"):
                for m in re.finditer(pattern, lower):
                    raw.append((m.start(), m.end(), option_id, text[m.start():m.end()]))
            if len(option_id) == 1 and option_id.upper() not in {"A", "I"}:
                for m in re.finditer(rf"\b{re.escape(option_id.upper())}\b", text):
                    raw.append((m.start(), m.end(), option_id, m.group(0)))
        for alias, option_id in self.alias_to_id.items():
            for m in re.finditer(rf"\b{re.escape(alias)}\b", lower):
                raw.append((m.start(), m.end(), option_id, text[m.start():m.end()]))
        raw.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        found: list[OptionMention] = []
        occupied_until = -1
        for start, end, option_id, form in raw:
            if start < occupied_until:
                continue
            occupied_until = end
            found.append(OptionMention(
                option_id=option_id,
                span=EvidenceSpan(text=form, start=start),
                order=len(found),
                alias_form=form,
                resolution="explicit",
            ))
        return found

    def ids_in_text(self, text: str) -> list[str]:
        """Unique option ids mentioned, in visible text order (item 5)."""
        found: list[str] = []
        for mention in self.mentions(text):
            if mention.option_id not in found:
                found.append(mention.option_id)
        return found

    def resolve_reference(self, text: str, *, context_candidates: Sequence[str] = ()) -> tuple[str | None, bool]:
        """Resolve an implicit option reference ("it", "that one", "the
        former/latter") to one unambiguous PUBLIC referent, or report ambiguity.

        Returns ``(option_id, ambiguous)``. Resolution sources, in order:
        an explicit mention earlier in the same utterance; "the former/latter"
        against an ordered two-candidate public context; exactly one public
        context candidate (targeted prior turn / active visible thread).
        Several plausible candidates stay unresolved with ``ambiguous=True``.
        Hidden intent is never a source: callers must pass only public context.
        """
        match = _IMPLICIT_REF.search(text)
        if match is None:
            return None, False
        preceding = [m for m in self.mentions(text) if m.span.start < match.start()]
        if preceding:
            return preceding[-1].option_id, False
        candidates = [c for c in dict.fromkeys(context_candidates) if c in self.by_id]
        phrase = " ".join(match.group(0).lower().split())
        if phrase.startswith("the former") and len(candidates) == 2:
            return candidates[0], False
        if phrase.startswith("the latter") and len(candidates) == 2:
            return candidates[1], False
        if len(candidates) == 1:
            return candidates[0], False
        return None, len(candidates) > 1

    def invalid_option_refs(self, text: str) -> list[str]:
        valid = set(self.by_id)
        refs = re.findall(r"\bOption\s+([A-Za-z])\b", text)
        return sorted({r.upper() for r in refs if r.upper() not in valid})


# Implicit-reference shapes eligible for context resolution: pronoun + verb
# ("it is", "that works", "it's"), demonstrative one-phrases ("that one"), and
# ordered references ("the former", "the latter"). A bare "that" as a
# conjunction ("I think that ...") never matches.
_IMPLICIT_REF = re.compile(
    r"\b(?:it|that|this)(?:'s|\s+(?:is|was|would|will|might|could|can|does|"
    r"seems?|feels?|looks?|sounds?|works?|costs?|takes?|has|gets?|keeps?|leaves?|beats?|fits?))\b"
    r"|\b(?:that|this)\s+one\b|\bthe\s+former\b|\bthe\s+latter\b",
    re.I,
)


def _option_positions(segment: str, resolver: OptionResolver) -> list[tuple[int, str]]:
    """Earliest match position per option in ``segment``, sorted by position."""
    hits: dict[str, int] = {}
    for mention in resolver.mentions(segment):
        if mention.option_id not in hits:
            hits[mention.option_id] = mention.span.start
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
    normalised = text.replace("’", "'")
    lower = normalised.lower()
    spans = [
        (m.span.start, m.span.start + len(m.span.text), m.option_id)
        for m in resolver.mentions(normalised)
    ]
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
    ("gets my vote", re.compile(r"\b(?:gets?|has)\s+my\s+vote\b", re.I)),
    ("my vote is", re.compile(r"\bmy\s+vote(?:\s+is|'?s\s+(?:on|for)|\s+goes\s+to)\b", re.I)),
    ("I'd go with", re.compile(r"\bi'?(?:d|ll)\s+go\s+with\b", re.I)),
    ("I'm going with", re.compile(r"\bi'?m\s+going\s+with\b", re.I)),
    ("I'm choosing", re.compile(r"\bi(?:'?m|\s+am)\s+(?:choosing|picking)\b|\bi\s+(?:choose|pick)\b", re.I)),
    ("I'm opting for", re.compile(r"\bi'?m\s+opting\s+for\b", re.I)),
    ("I'm settling on", re.compile(r"\bi'?m\s+(?:settling|landing)\s+on\b|\bi\s+land\s+on\b", re.I)),
    ("I'm committing to", re.compile(r"\bi(?:'?m|\s+am)\s+committing\s+to\b|\bi\s+commit\s+to\b|\bi'?ll\s+commit\s+to\b", re.I)),
    ("I'll take", re.compile(r"\bi'?ll\s+(?:take|pick|choose)\b|\bi'?ll\s+back\s+(?!down|off|out|up)|\bi\s+back\s+(?!down|off|out|up)", re.I)),
    ("my pick is", re.compile(r"\bmy\s+(?:final\s+)?(?:pick|choice)\s+is\b|\bis\s+my\s+pick\b|\bthat'?s\s+my\s+pick\b|\bis\s+where\s+i\s+land\b", re.I)),
    ("my choice is", re.compile(r"\b(?:is|makes\s+it)\s+(?:definitely\s+|clearly\s+|easily\s+)?my\s+choice\b|\bmy\s+top\s+choice\s+is\b", re.I)),
    ("I vote for", re.compile(r"\bi\s+vote(?!\s+(?:against|no\b))(?:\s+for)?\b|\bi'?m\s+voting\b", re.I)),
    ("I'm switching to", re.compile(r"\bswitch(?:ing)?\s+(?:from\s+[^.;,!?]{0,40}?\s+)?to\b", re.I)),
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


# Commitment phrases whose grammatical object comes before the phrase
# ("X gets my vote", "X works for me", "X is my pick").
_SUBJECT_FORM_COMMIT = re.compile(
    r"(?:(?:gets?|has)\s+my\s+vote|works\s+(?:best\s+)?for\s+me|is\s+where\s+i\s+land|"
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


# ---------------------------------------------------------------------------
# Critical-parser post-checks (todo_validation item 6)
#
# The deterministic commitment layer stays conservative and narrow: strict
# public votes/acceptances, blocker safety, and required-target protection.
# Natural menu-less vote wording is recognized by the deterministic commitment
# parser. Every claimed commitment must pass these checks before it can count.
# A hidden required_vote never counts
# without a visible, unambiguous public commitment.
# ---------------------------------------------------------------------------

# Prerequisite wording that voids a final commitment wherever it appears.
_PREREQUISITE = re.compile(
    r"\bonly\s+if\b|\bunless\b|\bwould\s+need\b|\bneed\s+to\s+know\b|\bdepends\b|"
    r"\bare\s+we\s+okay\b|\bas\s+long\s+as\b|\bprovided\s+that\b|\bif\b",
    re.I,
)
# On sanctioned switches, concessive riders ("as long as", "if we…") are the
# requested bridge shape; only genuine prerequisites still void.
_SANCTION_PREREQUISITE = re.compile(
    r"\bonly\s+if\b|\bunless\b|\bwould\s+need\b|\bneed\s+to\s+know\b|\bdepends\b|\bare\s+we\s+okay\b",
    re.I,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def commitment_post_checks(
    text: str,
    option_id: str,
    resolver: OptionResolver,
    *,
    kind: str = "vote",
    required_vote: str | None = None,
    rejected_options: Sequence[str] = (),
    resolves_blocker: str | None = None,
    sanctioned_switch: bool = False,
) -> list[str]:
    """Deterministic post-checks for one claimed public commitment.

    Returns issue codes (empty = pass). Enforced regardless of how the
    commitment was found:

    - the vote target must be visibly named (``COMMITMENT_TARGET_NOT_NAMED``);
    - no unresolved prerequisite anywhere (``CONDITIONAL_COMMITMENT``);
    - no question masquerading as the vote — the committing sentence itself
      must not be a question (``QUESTION_NOT_COMMITMENT``); a trailing
      check-in question in another sentence does not void the commitment;
    - no conflicting commitment to a different option (``CONFLICTING_COMMITMENT``);
    - required controller target alignment (``REQUIRED_VOTE_MISMATCH``);
    - rejected-option protection with the one same-line resolution exception
      (``BLOCKED_OPTION_ACCEPTED``).
    """
    issues: list[str] = []
    check = text.replace("’", "'").replace("‘", "'")
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(check) if s.strip()]
    commit_sentences = [s for s in sentences if option_id in resolver.ids_in_text(s)]
    if kind == "vote" and option_id not in resolver.ids_in_text(check):
        issues.append("COMMITMENT_TARGET_NOT_NAMED")
    prerequisite = _SANCTION_PREREQUISITE if sanctioned_switch else _PREREQUISITE
    if prerequisite.search(check):
        issues.append("CONDITIONAL_COMMITMENT")
    if commit_sentences and all(s.endswith("?") for s in commit_sentences):
        issues.append("QUESTION_NOT_COMMITMENT")
    for sentence in sentences:
        other = visible_commitment(sentence, resolver, sanctioned_switch=sanctioned_switch)
        if other and other[0] in {"vote", "accept"} and other[1] != option_id:
            issues.append("CONFLICTING_COMMITMENT")
            break
    if required_vote and option_id != required_vote:
        issues.append("REQUIRED_VOTE_MISMATCH")
    if option_id in set(rejected_options) and resolves_blocker != option_id:
        issues.append("BLOCKED_OPTION_ACCEPTED")
    return issues


def visible_question(
    text: str,
    *,
    speaker_id: str,
    participant_names: dict[str, str],
    previous_speaker_id: str | None = None,
) -> tuple[str, str | None] | None:
    """(scope, addressee) of a genuine visible question, else None.

    Question scope from visible text only (4.4/5.5): a named or "you"-directed
    question is direct; a genuine question without an addressee is a group
    question with NO target — the controller assigns the respondent later.
    The single deterministic owner of question detection (item 7): both the
    display parse and the evidence contract consume it.
    """
    if not _QUESTION.search(text):
        return None
    addressee = resolve_addressee(text, speaker_id, participant_names)
    if addressee is None and previous_speaker_id and previous_speaker_id != speaker_id:
        if re.search(r"\b(?:you|your|that|what\s+about|how\s+about)\b", text, re.I):
            addressee = previous_speaker_id
    if addressee:
        return "direct", addressee
    if _is_genuine_question(text):
        return "group", None
    return None


def visible_comparison(text: str, resolver: "OptionResolver") -> list[str] | None:
    """The two option ids in a basic visible comparison, or None (item 3).

    Deterministic when two distinct options are visibly present and connected by
    a comparative construction: a relational/contrast connective straddling two
    option spans, or a comparative adjective anywhere alongside two options.
    Subtle direction/dimension stays the validator's job; this only asserts that
    a two-option comparison is visibly present so COMPARISON_MISSES_OPTIONS is
    reserved for real failures.
    """
    distinct: list[tuple[str, int]] = []
    seen: set[str] = set()
    for m in resolver.mentions(text):
        if m.option_id not in seen:
            seen.add(m.option_id)
            distinct.append((m.option_id, m.span.start))
    if len(distinct) < 2:
        return None
    check = text.replace("’", "'").replace("‘", "'")
    # A connective that actually separates two different option spans.
    for rx in (_RELATIONAL, _CONTRAST):
        for m in rx.finditer(check):
            before = [oid for oid, p in distinct if p < m.start()]
            after = [oid for oid, p in distinct if p >= m.end() and oid not in before]
            if before and after:
                return [before[-1], after[0]]
    # A comparative adjective with two options present but no straddling
    # connective (e.g. possessive juxtaposition "A's cost is double B's").
    if _COMPARATIVE_ADJ.search(check):
        return [distinct[0][0], distinct[1][0]]
    return None


def _is_genuine_question(text: str) -> bool:
    """A group question with no visible addressee: genuine interrogative
    structure (WH/aux clause or a short A-or-B choice), never a tag check-in.
    The `?` gate is applied by the caller."""
    if _RHETORICAL_TAIL.search(text):
        return False
    if _QUESTION_CLAUSE.search(text):
        return True
    return bool(_CHOICE_QUESTION.search(text)) and len(text.split()) <= 12


def resolve_addressee(text: str, speaker_id: str, participant_names: dict[str, str]) -> str | None:
    """The participant visibly named in the text, if any — the single owner of
    addressee resolution (item 5). Never invents an addressee from intent:
    controller target vs. visible addressee is compared during validation."""
    lower = text.lower()
    for pid, name in participant_names.items():
        if pid == speaker_id:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return pid
    return None
