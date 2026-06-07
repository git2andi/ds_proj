"""Canonical dialogue-act parser.

All vote/accept/reject/question interpretation lives here.  Other modules consume
DialogueAct instead of re-parsing natural-language messages.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import ActType, DialogueAct, MoveIntent
from utils import OptionResolver

_VOTE_CUES = re.compile(
    r"\b(?:i\s+(?:pick|choose|prefer|vote(?:\s+for)?|would\s+go\s+with|would\s+pick|want)|"
    r"i['’]?d\s+(?:pick|choose|go\s+with|take)|my\s+(?:pick|choice|vote)\s+(?:is|would\s+be)|"
    r"let['’]?s\s+(?:do|pick|go\s+with)|i['’]?m\s+(?:for|leaning\s+toward|going\s+with))\b",
    re.I,
)

_ACCEPT_CUES = re.compile(
    r"\b(?:can\s+live\s+with|could\s+live\s+with|works?\s+for\s+me|works?\s+well\s+enough|"
    r"works?\s+as\s+(?:a\s+)?compromise|i\s+can\s+accept|i['’]?d\s+accept|"
    r"i\s+could\s+accept|i\s+can\s+do|i\s+could\s+do|good\s+with|fine\s+with|okay\s+with|"
    r"sounds?\s+(?:good|fine|okay|ok)|that\s+works|that['’]?s\s+(?:fine|okay|ok)|"
    r"no\s+objection)\b",
    re.I,
)

_REJECT_CUES = re.compile(
    r"\b(?:can['’]?t\s+live\s+with|couldn['’]?t\s+live\s+with|doesn['’]?t\s+work\s+for\s+me|"
    r"not\s+(?:okay|ok|fine|good|sold|convinced)|i\s+wouldn['’]?t\s+do|i\s+can['’]?t\s+do|"
    r"no\s+to|rather\s+not|rule\s+out|hard\s+no)\b",
    re.I,
)

_YES_ONLY = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok|okay|fine|agreed|works|sounds\s+good)\b", re.I)
_NO_ONLY = re.compile(r"^\s*(?:no|nope|nah|not\s+really|still\s+not)\b", re.I)


def parse_dialogue_act(
    speaker_id: str,
    speaker_name: str,
    text: str,
    resolver: OptionResolver,
    participant_names: dict[str, str],
    intent: Optional[MoveIntent] = None,
) -> DialogueAct:
    option_refs = resolver.ids_in_text(text)
    addressee_id = _extract_addressee(text, speaker_id, participant_names)
    question_target = addressee_id if "?" in text and addressee_id else None

    vote = _extract_vote(text, option_refs)
    accepts = _extract_accepts(text, option_refs, intent)
    rejects = _extract_rejects(text, option_refs, intent)
    proposes = _extract_proposal(text, option_refs, intent)

    act_type = _infer_act_type(text, vote, accepts, rejects, proposes, question_target, intent)
    return DialogueAct(
        speaker_id=speaker_id,
        text=text,
        act_type=act_type,
        option_refs=option_refs,
        addressee_id=addressee_id,
        question_target_id=question_target,
        explicit_vote=vote,
        accepts=accepts,
        rejects=rejects,
        proposes_option=proposes,
    )


def _extract_addressee(text: str, speaker_id: str, participant_names: dict[str, str]) -> Optional[str]:
    lower = text.lower()
    for pid, name in participant_names.items():
        if pid == speaker_id:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return pid
    return None


def _extract_vote(text: str, option_refs: list[str]) -> Optional[str]:
    if option_refs and _VOTE_CUES.search(text):
        return option_refs[0]
    return None


def _extract_accepts(text: str, option_refs: list[str], intent: Optional[MoveIntent]) -> list[str]:
    candidate = intent.option_focus[0] if intent and intent.option_focus else None
    if option_refs and _ACCEPT_CUES.search(text):
        return option_refs
    if candidate and intent and intent.act in {ActType.ACCEPT, ActType.VOTE} and (_ACCEPT_CUES.search(text) or _YES_ONLY.search(text)):
        return [candidate]
    return []


def _extract_rejects(text: str, option_refs: list[str], intent: Optional[MoveIntent]) -> list[str]:
    candidate = intent.option_focus[0] if intent and intent.option_focus else None
    if option_refs and _REJECT_CUES.search(text):
        return option_refs
    if candidate and intent and intent.act == ActType.ACCEPT and (_REJECT_CUES.search(text) or _NO_ONLY.search(text)):
        return [candidate]
    return []


def _extract_proposal(text: str, option_refs: list[str], intent: Optional[MoveIntent]) -> Optional[str]:
    if option_refs and re.search(r"\b(?:middle\s+ground|compromise|maybe\s+we\s+could|what\s+if\s+we|could\s+we\s+do)\b", text, re.I):
        return option_refs[0]
    if intent and intent.act == ActType.PROPOSE_COMPROMISE and intent.option_focus:
        return intent.option_focus[0]
    return None


def _infer_act_type(
    text: str,
    vote: Optional[str],
    accepts: list[str],
    rejects: list[str],
    proposes: Optional[str],
    question_target: Optional[str],
    intent: Optional[MoveIntent],
) -> ActType:
    if vote:
        return ActType.VOTE
    if rejects:
        return ActType.REJECT
    if accepts:
        return ActType.ACCEPT
    if proposes:
        return ActType.PROPOSE_COMPROMISE
    if question_target or "?" in text:
        return ActType.ASK
    if intent:
        return intent.act
    return ActType.OTHER
