# 05 — Discussion and decision flow

The discussion moves through explicit controller phases with a validated transition graph:

```text
opening -> discussion -> narrowing -> voting -> closing
narrowing -> discussion            (at most once, when the tested candidate collapses)
voting -> compromise_repair -> voting | closing
```

The opening round gives every participant a visible initial stance. It may include a very short chat-like greeting, but its main job is to state the current favorite and one grounded reason. Opening leans are public stance, never final votes.

## Stance movement

Private stance movement is represented by option ranks:

```text
5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 rejected
```

The controller chooses an intended move (`MoveIntent`); the LLM renders the utterance; validation checks that the line visibly matches the move; only then does the observer update the rank table, thread state, and coverage. Softening is not a routed move: when a final accepted line visibly warms to another option ("B is starting to make more sense to me"), the parser detects it and the observer moves the lean.

## Narrowing readiness

Normal `discussion -> narrowing` requires all mandatory conditions: no owed answer, no active repair, no hot hard blocker against the candidate, minimum discussion turns, coverage complete or validly attempted, at least one option with visible support from discussion turns (not just opening), and one realized head-to-head comparison. Then at least one trigger must fire: a visible support cluster, a stable top pair over the configured window, target length or the no-progress threshold with a candidate present, or the approaching hard cap with a viable candidate. The hard-cap override relaxes minimum-turn/coverage/support evidence but never erases an answer obligation or fabricates support.

## Narrowing behavior

Narrowing is bounded: one summary beat (participant-led when possible; moderator-led when the discussion was circling or the target length forced it) plus one holdout reaction beat testing the candidate. If the candidate visibly collapses, the flow returns to discussion exactly once while budget remains; otherwise it proceeds to voting.

## Voting and the repair state machine

During `voting`, every participant produces one formal visible commitment. After tallying, at most one repair objective runs at a time, classified in priority order and each at most once per run:

```text
1. unclear_vote          -> bounded clarification round for unclear voters only
2. (hard blockers        -> handled inside the flows: blocked voters vote an
                            acceptable alternative; holdouts with valid blocks
                            are never pressured)
3. majority_holdout      -> one reservation exchange + visible stay/switch beats
4. split_vote            -> summary, bounded exchanges, visible re-vote (max 2 candidates)
5. two_person_deadlock   -> each side names blocker + condition, then final commitments
```

`switch_resistance` governs all final movement (switching, compromise acceptance, holdout concession); `stubbornness` only governs discussion-phase defense. If no majority remains after repair, the run closes `unresolved` — honestly earned, never abrupt.

## Stubbornness, switch_resistance, and rejection

```text
stubbornness high       = defends hard during discussion, but final movement is separate
switch_resistance high  = very hard to move in narrowing/voting/repair, yet theoretically movable
rejection true          = hard blocker / cannot accept the rejected option (rank 1)
```

A hard blocker never comes from traits alone — only `rejection` and option-rank 1 are binding, and blocker-thread staleness never erases the underlying rejection.

## Speech-style boundary

Age/speech_style may change how a participant says a point. It must not change what the controller asks them to do, which option they prefer, or whether a switch is plausible.

## What must not happen

- no hidden consensus from private ranks;
- no discussion-phase commitment silently becoming a final vote;
- no invented blended option;
- no forced successful outcome when a rank-1 blocker remains;
- no invalid line printed as transcript evidence;
- no endless negotiation loop (every protocol is attempt- or turn-bounded);
- no speech-style feature overriding parameter-driven behavior.
