"""Deterministic thread engine (todo sections 5.4–5.7, 6).

Single owner of thread lifecycle state: creation/update, deterministic issue
keys, status aging, reactivation, and primary-thread selection. No other module
may assign ``thread.status`` directly. Everything here is deterministic and
LLM-free; probabilistic decisions (e.g. whether to continue a cooling thread)
belong to the router, which rolls its own dice and then calls in here.

All functions are pure over the passed state except the explicitly mutating
lifecycle entry points (open/touch/mark_response/resolve/reactivate/age),
which are called only from post-turn observation or flow code — never from
route selection.
"""

from __future__ import annotations

import re

from aliases import _GENERIC, _STOPWORDS, short_alias_map
from config_loader import cfg
from models import (
    BlockingStrength,
    DialogueState,
    Scenario,
    ThreadState,
    ThreadStatus,
    ThreadType,
)

# ---------------------------------------------------------------------------
# Issue-key normalization (5.6)
# ---------------------------------------------------------------------------

# Deterministic parser categories, tried after scenario attribute keys.
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("availability", re.compile(
        r"\b(?:availab\w+|book(?:ing|ed|s)?|reserv\w+|open\s+(?:on|that|late)|sold\s+out|slots?)\b", re.I)),
    ("timing", re.compile(
        r"\b(?:timing|schedule\w*|too\s+(?:late|early|long|short)|duration|hours?|deadline|when\b)\b", re.I)),
    ("accessibility", re.compile(
        r"\b(?:accessib\w+|wheelchair|stairs|walk(?:ing)?\s+distance|too\s+far|travel|commute|transit|parking)\b", re.I)),
    ("capacity", re.compile(
        r"\b(?:capacity|seats?|seating|fit\s+(?:all|everyone)|enough\s+(?:space|room)|group\s+size|crowd\w*)\b", re.I)),
    ("risk", re.compile(
        r"\b(?:risk\w*|unsafe|safety|danger\w*|injur\w*|allerg\w*|fail\w*|unreliab\w*)\b", re.I)),
    ("weather", re.compile(r"\b(?:weather|rain\w*|snow\w*|forecast\w*|outdoor\w*)\b", re.I)),
    ("cost", re.compile(
        r"\b(?:cost\w*|price\w*|budget|expensive|cheap\w*|afford\w*|euros?|dollars?|spend\w*)\b", re.I)),
]

# Generic evaluative words excluded from content signatures.
_EVALUATIVE = frozenset(
    "good bad better best worse worst nice great fine okay really think feel "
    "maybe quite pretty very just still even sure like want need makes seems "
    "sounds looks little much more most less thing things point issue concern "
    "worry problem option pick choice".split()
)

_WORD = re.compile(r"[a-zäöüß0-9'-]{3,}", re.I)


def _attr_key_in_text(key: str, text_lower: str) -> bool:
    words = [w for w in re.split(r"[_\s]+", key.lower()) if w]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}\w*", text_lower) for w in words)


def normalize_issue_key(
    text: str,
    scenario: Scenario,
    participant_names: list[str],
    *,
    focus_options: list[str] | None = None,
) -> str:
    """Deterministic issue key for a visible utterance (5.6).

    Priority: matching scenario option attribute key, then a deterministic
    parser category, then a normalized content signature. No LLM, no embeddings.
    """
    lower = text.lower()
    # 1. Scenario attribute keys, focus options first for stable preference.
    focus = focus_options or []
    ordered_options = [
        scenario.option(oid) for oid in focus if oid in scenario.option_ids
    ] + [o for o in scenario.options if o.id not in focus]
    seen_keys: list[str] = []
    for option in ordered_options:
        for key in option.attrs:
            if key not in seen_keys:
                seen_keys.append(key)
    for key in seen_keys:
        if _attr_key_in_text(key, lower):
            return key.lower().replace(" ", "_")
    # 2. Deterministic categories.
    for name, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return name
    # 3. Content signature from stable tokens.
    drop = set(_STOPWORDS) | set(_GENERIC) | set(_EVALUATIVE)
    for name in participant_names:
        drop.update(w.lower() for w in _WORD.findall(name))
    for option in scenario.options:
        drop.update(w.lower() for w in _WORD.findall(f"{option.name} {option.short_name}"))
        drop.add(option.id.lower())
    alias_map = short_alias_map(scenario.options)
    for alias in alias_map.values():
        drop.update(w.lower() for w in _WORD.findall(alias))
    tokens = sorted({w.lower() for w in _WORD.findall(text)} - drop)
    if not tokens:
        return "general"
    return "sig:" + "-".join(tokens[:3])


def normalize_pair(option_ids: list[str]) -> list[str]:
    """Deterministic option-pair normalization for comparison threads."""
    return sorted(dict.fromkeys(option_ids))


# ---------------------------------------------------------------------------
# Creation / update
# ---------------------------------------------------------------------------

def thread_identity(thread_type: ThreadType, focus_options: list[str], issue_key: str) -> tuple:
    return (thread_type, tuple(normalize_pair(focus_options)), issue_key)


def find_thread(
    state: DialogueState,
    thread_type: ThreadType,
    focus_options: list[str],
    issue_key: str,
) -> ThreadState | None:
    identity = thread_identity(thread_type, focus_options, issue_key)
    for thread in state.threads.values():
        if thread_identity(thread.thread_type, thread.focus_options, thread.issue_key) == identity:
            return thread
    return None


def open_thread(
    state: DialogueState,
    *,
    thread_type: ThreadType,
    focus_options: list[str],
    issue_key: str,
    started_by: str,
    source_turn_index: int,
    required_respondent: str | None = None,
    blocking_strength: BlockingStrength = BlockingStrength.NONE,
    question_scope: str | None = None,
    reopen_resolved: bool = False,
) -> ThreadState | None:
    """Create a thread, or update/reactivate the matching existing one.

    Identity is (thread_type, normalized focus options, issue_key). A stale
    match is reactivated when the config allows it; a resolved match is left
    untouched unless ``reopen_resolved`` marks an explicit visible reopen.
    Returns the affected thread, or None when the issue was suppressed as a
    repeat of a resolved thread.
    """
    existing = find_thread(state, thread_type, focus_options, issue_key)
    if existing is not None:
        if existing.status is ThreadStatus.RESOLVED and not reopen_resolved:
            return None  # repeated resolved issue is suppressed (6.2)
        if existing.status is ThreadStatus.RESOLVED:
            reactivate_thread(state, existing, turn_index=source_turn_index, by=started_by)
        elif existing.status is ThreadStatus.STALE:
            if not bool(cfg.threads.allow_reactivation):
                return None
            reactivate_thread(state, existing, turn_index=source_turn_index, by=started_by)
        else:
            touch_thread(state, existing, turn_index=source_turn_index, participant_id=started_by)
        if blocking_strength is BlockingStrength.HARD:
            existing.blocking_strength = BlockingStrength.HARD
        return existing

    state.thread_counter += 1
    thread = ThreadState(
        thread_id=f"t{state.thread_counter:03d}",
        thread_type=thread_type,
        status=ThreadStatus.HOT,
        focus_options=normalize_pair(focus_options),
        issue_key=issue_key,
        started_by=started_by,
        required_respondent=required_respondent,
        participants_involved=[started_by] if started_by else [],
        contribution_count=1,
        source_turn_index=source_turn_index,
        created_turn=source_turn_index,
        last_touched_turn=source_turn_index,
        blocking_strength=blocking_strength,
        question_scope=question_scope,  # type: ignore[arg-type]
    )
    state.threads[thread.thread_id] = thread
    return thread


def touch_thread(state: DialogueState, thread: ThreadState, *, turn_index: int, participant_id: str) -> None:
    # One accepted turn counts once, even when several observer paths fold it
    # into the same thread (per-thread caps count accepted contributions).
    if turn_index > thread.last_touched_turn:
        thread.contribution_count += 1
    thread.last_touched_turn = max(thread.last_touched_turn, turn_index)
    if participant_id and participant_id not in thread.participants_involved:
        thread.participants_involved.append(participant_id)


# ---------------------------------------------------------------------------
# Lifecycle transitions (single owner of thread.status)
# ---------------------------------------------------------------------------

def mark_response(state: DialogueState, thread: ThreadState, *, responder_id: str, turn_index: int) -> None:
    """A relevant response landed: hot -> cooling (6.1–6.4)."""
    touch_thread(state, thread, turn_index=turn_index, participant_id=responder_id)
    if thread.status is ThreadStatus.HOT:
        thread.status = ThreadStatus.COOLING


def resolve_thread(state: DialogueState, thread: ThreadState, *, reason: str) -> None:
    thread.status = ThreadStatus.RESOLVED
    thread.resolution_reason = reason


def reactivate_thread(state: DialogueState, thread: ThreadState, *, turn_index: int, by: str) -> None:
    """stale/resolved -> hot on a visible reopen."""
    thread.status = ThreadStatus.HOT
    thread.resolution_reason = None
    touch_thread(state, thread, turn_index=turn_index, participant_id=by)


def participant_turns_since(state: DialogueState, turn_index: int) -> int:
    return sum(1 for t in state.turns if t.speaker_id != "moderator" and t.index > turn_index)


def age_threads(state: DialogueState) -> None:
    """Post-turn aging: cooling questions resolve quietly, unanswered threads go
    stale after their timeout, blockers keep priority longer."""
    cooling_turns = int(cfg.threads.cooling_turns)
    stale_after = int(cfg.threads.stale_after_turns)
    blocker_stale_after = int(cfg.threads.hard_blocker_stale_after_turns)
    for thread in state.threads.values():
        if thread.status not in (ThreadStatus.HOT, ThreadStatus.COOLING):
            continue
        age = participant_turns_since(state, thread.last_touched_turn)
        if thread.status is ThreadStatus.COOLING and thread.thread_type is ThreadType.QUESTION:
            if age >= cooling_turns:
                resolve_thread(state, thread, reason="answered; no follow-up in cooling window")
                continue
        timeout = (
            blocker_stale_after
            if thread.thread_type is ThreadType.BLOCKER and thread.blocking_strength is BlockingStrength.HARD
            else stale_after
        )
        if age >= timeout:
            # Staleness only removes routing priority; underlying stance state
            # (rejections, concerns) remains untouched.
            thread.status = ThreadStatus.STALE


# ---------------------------------------------------------------------------
# Primary-thread selection (5.7)
# ---------------------------------------------------------------------------

_BLOCKING_ORDER = {BlockingStrength.HARD: 2, BlockingStrength.SOFT: 1, BlockingStrength.NONE: 0}


def _tie_break_key(thread: ThreadState, candidate_options: set[str]) -> tuple:
    relevance = 1 if set(thread.focus_options) & candidate_options else 0
    return (
        -_BLOCKING_ORDER[thread.blocking_strength],
        -relevance,
        -thread.last_touched_turn,
        thread.created_turn,
        thread.thread_id,
    )


def _pick(threads: list[ThreadState], candidate_options: set[str]) -> ThreadState | None:
    if not threads:
        return None
    return min(threads, key=lambda t: _tie_break_key(t, candidate_options))


def select_primary_thread(
    state: DialogueState,
    *,
    candidate_options: list[str] | None = None,
    include_cooling: bool = False,
) -> ThreadState | None:
    """Deterministic primary-thread selection in the 5.7 priority order.

    ``candidate_options`` is the current candidate/top pair. Cooling threads
    are only eligible when the caller passes ``include_cooling`` (the router
    rolls the continuation probability itself; this function stays pure).
    A thread that has already driven the hard cap of accepted contributions
    stops driving turns entirely.
    """
    candidates = set(candidate_options or [])
    hard_cap = int(cfg.threads.max_thread_turns_hard)
    hot = [
        t for t in state.threads.values()
        if t.status is ThreadStatus.HOT and t.contribution_count < hard_cap
    ]

    direct_q = [
        t for t in hot
        if t.thread_type is ThreadType.QUESTION and t.question_scope == "direct" and t.required_respondent
    ]
    if direct_q:
        return _pick(direct_q, candidates)
    group_q = [t for t in hot if t.thread_type is ThreadType.QUESTION and t.question_scope == "group"]
    if group_q:
        return _pick(group_q, candidates)
    hard_blockers = [
        t for t in hot
        if t.thread_type is ThreadType.BLOCKER
        and t.blocking_strength is BlockingStrength.HARD
        and set(t.focus_options) & candidates
    ]
    if hard_blockers:
        return _pick(hard_blockers, candidates)
    candidate_concerns = [
        t for t in hot
        if t.thread_type is ThreadType.CONCERN and set(t.focus_options) & candidates
    ]
    if candidate_concerns:
        return _pick(candidate_concerns, candidates)
    other_hot = [
        t for t in hot
        if t.thread_type in (ThreadType.BLOCKER, ThreadType.CONCERN, ThreadType.COMPARISON)
    ]
    if other_hot:
        return _pick(other_hot, candidates)
    if include_cooling:
        cooling = [
            t for t in state.threads.values()
            if t.status is ThreadStatus.COOLING and t.contribution_count < hard_cap
        ]
        if cooling:
            return _pick(cooling, candidates)
    return None


def resolve_comparison_threads(state: DialogueState, *, reason: str) -> None:
    """Resolve open comparison threads (6.4): a candidate/top pair was
    established or the controller left discussion for narrowing."""
    for thread in state.threads.values():
        if thread.thread_type is ThreadType.COMPARISON and thread.status in (
            ThreadStatus.HOT,
            ThreadStatus.COOLING,
        ):
            resolve_thread(state, thread, reason=reason)


def hot_threads(state: DialogueState) -> list[ThreadState]:
    return [t for t in state.threads.values() if t.status is ThreadStatus.HOT]


def hot_blocking_thread_against(state: DialogueState, candidate_options: list[str]) -> ThreadState | None:
    """A hot hard blocker against the candidate/top pair (narrowing/voting gate)."""
    candidates = set(candidate_options)
    matches = [
        t for t in hot_threads(state)
        if t.thread_type is ThreadType.BLOCKER
        and t.blocking_strength is BlockingStrength.HARD
        and set(t.focus_options) & candidates
    ]
    return _pick(matches, candidates)
