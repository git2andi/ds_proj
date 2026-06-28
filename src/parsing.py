"""Option resolution and dialogue-act extraction.

State is not guessed from free prose. Each generated turn ends with a small
machine-readable trailer, e.g. ``[act=accept; opt=C; stance=accept]``, which is
parsed and stripped here. The option resolver only matches unambiguous option
names/ids. There are deliberately no phrase "cue" lists: deciding whether a
message is a vote/accept/reject is the model's job, reported via the trailer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models import ActType, DialogueAct, MoveIntent, OptionCard
from utils import normalise_ws

# Stance values the model may report in a trailer.
_STANCES = {"vote", "accept", "object", "reject", "propose", "neutral"}
_STANCE_TO_ACT = {
    "vote": ActType.VOTE,
    "accept": ActType.ACCEPT,
    "reject": ActType.REJECT,
    "object": ActType.OBJECT,
    "propose": ActType.PROPOSE_COMPROMISE,
}
# Stance assumed when the model omits the trailer, keyed by the routed intent.
_INTENT_FALLBACK_STANCE = {
    ActType.VOTE: "vote",
    ActType.ACCEPT: "accept",
    ActType.REJECT: "object",  # ambiguous reject -> soft; hard blocks come from persona data
    ActType.PROPOSE_COMPROMISE: "propose",
}
# Routed moves on which a binding commitment (a final vote or accepting the group's pick)
# is actually meaningful — the narrowing vote and the confirmation round.
_DECISION_ACTS = {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}

# The trailer starts at a "[" immediately followed by act=/opt=/stance= and runs to the
# final "]". The greedy ".*" deliberately swallows any inner brackets the model emits
# (e.g. opt=[C, D]); without it such a trailer would leak into the chat unparsed.
_TRAILER = re.compile(r"\[\s*((?:act|stance|opt)\s*=.*)\]", re.I | re.S)
# Same trailer with its surrounding brackets dropped (some models omit them): a trailing
# run of key=value fields at the very end of the message, with an optional stray bracket.
_BARE_TRAILER = re.compile(
    r"\[?\s*((?:act|opt|stance)\s*=\s*[^\s;,\]]+(?:\s*[;,]\s*(?:act|opt|stance)\s*=\s*[^\s;,\]]+)*)\s*\]?\s*$",
    re.I,
)
_QUESTION = re.compile(r"\?")
# A tentative/conditional acceptance ("might be okay", "works if we leave early", "I guess
# that's fine") is not a firm commitment. Detected only to gate a confirmation ACCEPT so a
# hedged line can't be miscounted as a clean acceptance and close the decision prematurely.
_HEDGED_ACCEPT = re.compile(
    r"\b(?:might|maybe|perhaps|possibly|i guess|i suppose|"
    r"still\s+not\s+(?:sure|sold|convinced)|not\s+fully\s+(?:sure|sold|convinced)|"
    r"(?:okay|fine|work|works|good)\s+(?:for\s+\w+\s+)?if|as long as|provided that|only if)\b",
    re.I,
)


@dataclass(slots=True)
class TurnMove:
    """Parsed trailer: what the speaker reports they just did.

    ``present`` distinguishes "the model emitted a trailer" from "no trailer, so
    fall back to the routed intent" — the validator only flags contradictions
    when a trailer was actually present."""

    act: ActType | None = None
    option: str | None = None
    stance: str = "neutral"
    present: bool = False


class OptionResolver:
    def __init__(self, options: list[OptionCard]) -> None:
        self.options = options
        self.by_id = {option.id: option for option in options}
        self.alias_to_id = self._build_aliases(options)

    @staticmethod
    def _build_aliases(options: list[OptionCard]) -> dict[str, str]:
        # Collect candidate aliases per option, then drop any alias claimed by more
        # than one option. This makes shared words harmless: "Night" in both
        # "Bowling Night" and "Game Night" belongs to neither once collisions are
        # removed, so an option only matches on its own distinctive words.
        candidates: dict[str, set[str]] = {}
        for option in options:
            name = option.name.lower()
            aliases = {name}
            clean = re.sub(r"[^\wäöüÄÖÜß\s'-]", " ", name)
            words = [w for w in clean.split() if len(w) >= 4]
            aliases.update(words)
            if len(words) >= 2:
                aliases.add(" ".join(words[:2]))
            # Include LLM-generated short name so "Ride" resolves to "Ticket to Ride" etc.
            # Require len>=4 to skip generic one-word aliases like "The", "One".
            if option.short_name and len(option.short_name) >= 4:
                aliases.add(option.short_name.lower())
            candidates[option.id] = {a.strip() for a in aliases if a.strip()}
        owners: dict[str, set[str]] = {}
        for option_id, aliases in candidates.items():
            for alias in aliases:
                owners.setdefault(alias, set()).add(option_id)
        return {alias: next(iter(ids)) for alias, ids in owners.items() if len(ids) == 1}

    def ids_in_text(self, text: str) -> list[str]:
        lower = text.lower()
        found: list[str] = []
        # Explicit "Option X" references first (single letters only ever match this way).
        for option_id in self.by_id:
            if re.search(rf"\boption\s+{re.escape(option_id.lower())}\b", lower) and option_id not in found:
                found.append(option_id)
        # Then distinctive name aliases, longest first so multi-word names win.
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


def parse_trailer(raw: str, option_ids: list[str]) -> tuple[str, TurnMove]:
    """Split a raw generation into (clean message, TurnMove). Tolerant of a
    trailer that is bracketed, missing its brackets, malformed, or absent."""
    move = TurnMove()
    body: str | None = None
    message = raw
    bracketed = list(_TRAILER.finditer(raw))
    if bracketed:
        match = bracketed[-1]
        body = match.group(1)
        message = raw[: match.start()] + raw[match.end():]
    else:
        bare = _BARE_TRAILER.search(raw)
        if bare:
            body = bare.group(1)
            message = raw[: bare.start()]
    if body is None:
        # No tag at all: clean up a lone option letter the model may have left dangling.
        return normalise_ws(_strip_dangling_option_letter(raw, option_ids)), move
    move.present = True
    fields: dict[str, str] = {}
    for part in re.split(r"[;,]", body):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip().lower()] = value.strip().lower()
    act_raw = fields.get("act")
    if act_raw:
        try:
            move.act = ActType(act_raw)
        except ValueError:
            move.act = None
    opt_letters = re.sub(r"[^A-Z]", "", (fields.get("opt") or "").upper())
    if opt_letters[:1] in option_ids:
        move.option = opt_letters[:1]
    stance = fields.get("stance", "neutral")
    move.stance = stance if stance in _STANCES else "neutral"
    return normalise_ws(message), move


def _strip_dangling_option_letter(text: str, option_ids: list[str]) -> str:
    """Remove a trailing isolated option letter left over from a malformed tag,
    e.g. '...for in-depth analysis C' -> '...for in-depth analysis'."""
    stripped = text.rstrip()
    m = re.search(r"(?:^|\s)([A-Za-z])[.\s]*$", stripped)
    if m and m.group(1).upper() in option_ids:
        return stripped[: m.start(1)].rstrip(" .,;:-")
    return text


def parse_dialogue_act(
    *,
    speaker_id: str,
    speaker_name: str,
    text: str,
    resolver: OptionResolver,
    participant_names: dict[str, str],
    move: TurnMove | None = None,
    intent: MoveIntent | None = None,
    previous_speaker_id: str | None = None,
) -> DialogueAct:
    text = normalise_ws(text)
    option_refs = resolver.ids_in_text(text)
    addressee_id = _extract_addressee(text, speaker_id, participant_names)
    if addressee_id is None and _QUESTION.search(text) and previous_speaker_id and previous_speaker_id != speaker_id:
        if re.search(r"\b(?:you|your|that|what\s+about|how\s+about)\b", text, re.I):
            addressee_id = previous_speaker_id
    question_target = addressee_id if _QUESTION.search(text) and addressee_id else None
    # Questions to the room ("How many can it seat?", "What's the vibe like?") have
    # no explicit addressee but still deserve an answer. Pick the best respondent:
    # whoever champions the mentioned option, or fall back to the previous speaker.
    if question_target is None and _QUESTION.search(text) and _is_genuine_question(text):
        best = _best_respondent(speaker_id, option_refs, participant_names, previous_speaker_id)
        if best:
            question_target = best

    hedged = bool(_HEDGED_ACCEPT.search(text))
    stance, focus_opt, act_type = _resolve_move(move, intent, option_refs, question_target, hedged)
    if focus_opt and focus_opt not in option_refs:
        option_refs = [focus_opt, *option_refs]

    return DialogueAct(
        speaker_id=speaker_id,
        text=text,
        act_type=act_type,
        option_refs=option_refs,
        addressee_id=addressee_id,
        question_target_id=question_target,
        explicit_vote=focus_opt if stance == "vote" else None,
        accepts=[focus_opt] if stance == "accept" and focus_opt else [],
        soft_rejects={focus_opt: text} if stance == "object" and focus_opt else {},
        hard_rejects={focus_opt: text} if stance == "reject" and focus_opt else {},
        proposes_option=focus_opt if stance == "propose" else None,
    )


def _resolve_move(
    move: TurnMove | None,
    intent: MoveIntent | None,
    option_refs: list[str],
    question_target: str | None,
    hedged: bool = False,
) -> tuple[str, str | None, ActType]:
    focus_opt: str | None = None
    if move and move.option:
        focus_opt = move.option
    elif len(option_refs) == 1:
        focus_opt = option_refs[0]
    elif intent and len(intent.option_focus) == 1:
        # On no-trailer binding turns (VOTE/ACCEPT), the option must be visible in text.
        # Using intent.option_focus here would ghost-commit to an option the speaker
        # never mentioned, making invisible routing state look like social proof (KF03).
        trailer_present = move and move.present
        if trailer_present or intent.act not in {ActType.VOTE, ActType.ACCEPT}:
            focus_opt = intent.option_focus[0]

    # An opening turn only states an initial leaning — it can never be a vote, accept,
    # or compromise proposal, no matter what stance the model put in its trailer.
    # Clamp to a neutral opening so the first round can't be miscounted as commitments
    # (which used to inflate option acceptances and skew consensus).
    if intent and intent.act == ActType.OPENING:
        return "neutral", focus_opt, ActType.OPENING

    stance = move.stance if move and move.stance != "neutral" else "neutral"
    if stance == "neutral" and intent:
        stance = _INTENT_FALLBACK_STANCE.get(intent.act, "neutral")
    # A question is never a binding commitment, even when the intent was ACCEPT/VOTE and
    # the fallback stance would otherwise credit it. The model asked something instead of
    # committing; don't let the fallback silently register a false acceptance that triggers
    # premature consensus before the question is answered.
    if question_target and stance in {"accept", "vote"}:
        stance = "neutral"

    # A binding commitment is only real on a routed decision turn (the vote round or
    # confirmation). If the model tags an ordinary discussion line as a vote/accept, drop
    # it to neutral: a mid-discussion "accept" used to be counted as a real acceptance,
    # inflating the tally into a premature, often fake-unanimous outcome while the chat was
    # still openly arguing. We also discard the trailer's act so the turn falls back to its
    # routed move (e.g. SUPPORT), which still moves the lean — persuasion isn't lost, only
    # the spurious binding accept.
    clamped = bool(intent and intent.act not in _DECISION_ACTS and stance in {"vote", "accept"})
    if clamped:
        stance = "neutral"

    # A hedged/conditional acceptance ("might be okay", "works if we leave early") is not a
    # firm commitment, so it must not close the decision. Keep it neutral: the persona stays
    # an open holdout until the condition is actually resolved, instead of the chat declaring
    # consensus off a tentative line.
    if hedged and intent and intent.act == ActType.ACCEPT and stance == "accept":
        stance = "neutral"

    if stance in _STANCE_TO_ACT:
        act_type = _STANCE_TO_ACT[stance]
    elif move and move.act and not clamped:
        act_type = move.act
    elif intent and intent.act == ActType.OPENING:
        act_type = ActType.OPENING
    elif question_target or (intent and intent.act == ActType.ASK):
        act_type = ActType.ASK
    elif intent:
        act_type = intent.act
    else:
        act_type = ActType.REACT
    return stance, focus_opt, act_type


_RHETORICAL_TAIL = re.compile(r",\s*(?:right|yeah|no|huh|eh|you know|don't you think)\s*\?\s*$", re.I)
_GENUINE_QUESTION = re.compile(
    r"\b(?:how\s+(?:many|much|long|far|about)|what(?:'s|\s+is|\s+are|\s+do|\s+does|\s+about|\s+if)"
    r"|where|when|which|who|can\s+(?:we|it|they)|do\s+(?:we|they|you)|does\s+(?:it|that|this)"
    r"|is\s+(?:it|there|that)|are\s+(?:there|they|we)|could\s+(?:we|it)|would\s+(?:it|that))\b",
    re.I,
)


def _is_genuine_question(text: str) -> bool:
    if _RHETORICAL_TAIL.search(text):
        return False
    return bool(_GENUINE_QUESTION.search(text))


def _best_respondent(
    speaker_id: str,
    option_refs: list[str],
    participant_names: dict[str, str],
    previous_speaker_id: str | None,
) -> str | None:
    others = [pid for pid in participant_names if pid != speaker_id]
    if not others:
        return None
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
