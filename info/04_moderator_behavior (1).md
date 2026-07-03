# Moderator behavior

## Purpose

The moderator is the environment facilitator. It should not dominate the discussion, but it should prevent common multi-user failures:

```text
ignored direct questions
premature voting
silent options
unclear commitments
endless split votes
unsupported factual drift
```

## MUCA-style what / when / who logic

The moderator should act according to three decisions:

```text
what:     what intervention is needed?
when:     is this the right moment to intervene?
who:      who should answer or be addressed?
```

Examples:

```text
what: clarify an unclear vote
when: after a conditional support statement
who: the participant who made that statement
```

or:

```text
what: request an answer
when: a direct question was ignored
who: the directly addressed participant
```

## Good moderator behavior

The moderator should be short and state-aware:

```text
Anton, Kenji asked about the no-checked-bag issue. Is that a deal-breaker for you?
```

```text
We have one vote for D and one for B. Quinn, before we close this, would C be an acceptable compromise or not?
```

```text
Before final votes, D has not been discussed. Can someone give one reason to keep or reject it?
```

## Bad moderator behavior

The moderator should avoid generic prompts that ignore the current state:

```text
Can everyone share what feels best?
```

It should also avoid asking a targeted question and then allowing another sim to answer instead.

## Direct response obligations

When the moderator addresses a participant by name, the next non-moderator turn should normally be that participant answering. Exceptions should be rare and explicit, for example if another participant repairs a misunderstanding before the target answers.

Required behavior:

```text
Moderator -> Anton question
Next participant turn -> Anton answer
```

not:

```text
Moderator -> Anton question
Next participant turn -> Kenji unrelated vote
```

## Finalization behavior

The moderator should move to voting only when the discussion has enough public evidence:

```text
major options have been socially processed
direct questions are answered or abandoned
participants have visible stances or unresolved concerns
no required clarification is pending
```

If votes split, the moderator may allow one bounded compromise attempt. After that, unresolved is acceptable.

## Implementation status (2026-07-03)

Implemented in `dialogue.py`: stall nudges pick a concrete visible issue in
priority order (uncovered option → pending question → unresolved blocker on the
candidate, probed once → visible split, weighed head-to-head → single holdout →
generic concern). Vote calls are option-neutral — the candidate is never named
inside the question. Moderator-addressed participants get a response obligation
consumed before normal speaker selection. Voting requires visible support
(cluster or visible compromise proposal); a split gets one bounded compromise
pass and a majority gets one minority-check beat before closure.
