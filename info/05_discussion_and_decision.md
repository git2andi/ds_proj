# 05 — Discussion and decision flow

The discussion moves through explicit controller phases with a validated transition graph:

```text
opening -> discussion -> narrowing -> voting -> closing
narrowing -> discussion            (at most once, when the tested candidate collapses)
voting -> compromise_repair -> voting | closing
```

The opening round gives every participant a visible initial stance. It may include a very short chat-like greeting, but its main job is to state the current favorite and one grounded reason. Opening leans are public stance, never final votes.

## Stance movement

Private stance movement is represented by option ranks — the ONLY persistent private stance state; there is no hidden commitment/confidence float:

```text
5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 rejected
```

Only accepted visible text moves ranks:

- visible softening ("B is starting to make more sense to me") promotes the softened-to option;
- a visible acceptance ("B works for me too") makes that option acceptable (rank 4) without moving the lean;
- a visible bridged vote or sanctioned switch promotes the committed option; the former preferred option drops to acceptable;
- an explicit visible lean statement (for example, "I’m warming to B") may move the public and private lean; an ordinary support, concern, proposal, or conditional remark does not silently change ranks;
- a hard rejection (rank 1) is never silently removed — only the speaker's own visible resolution reopens it;
- controller intent alone never changes a rank, and repaired-away, fallback-blocked, or dropped text moves nothing.

`discussion_lean_shifts` counts every accepted turn that changed the speaker's top option during the discussion phase, whatever visible path caused it.

Other participants' turns never rewrite someone else's private ranks. Public support, criticism, or pressure works only through routing and opportunity: the controller may make an option more relevant, select it as a candidate, or route the pressured participant to respond — and only that participant's own accepted visible utterance may then update their ranks.

## One participant turn

```text
1. controller selects the speaker and a MoveIntent (act, objective, focus, target)
2. the dialogue LLM realizes exactly that move inside an <utterance> envelope
3. conservative extraction removes only response wrappers/labels
4. the deterministic critical parser extracts option mentions, questions,
   commitments, switches, blockers, and explicit lean movement
5. deterministic validation checks only correctness-critical properties:
   structure, aliases, required focus/questions, formal votes, switches,
   blocked-option acceptance, existing-option compromise, transferred exact
   values, unlisted exact quantities, and explicit unlisted feature/location claims
6. at most one targeted repair runs for a blocking critical failure
7. a minimal truthful fallback is available only for formal votes/switches and
   known-unknown answers; otherwise the attempt is dropped
8. only the final accepted visible text updates ranks, votes, threads, coverage,
   commitments, and blockers
```

Normal support, concern, comparison, opinion, implication, and answer quality are not sent to a validator LLM. The default critical runtime constructs no validator client and normally makes zero validator calls. A failed route is recorded with original/repair/fallback candidate text; repeating the same failed route first changes speaker and then simplifies or retires that route instead of issuing the same request indefinitely.

Concrete example: the controller routes Jonas a concern about the Museum. If the accepted line says that the listed 24-euro cost feels high, the observer opens a concern thread but leaves his rank unchanged. If he explicitly says “I’m warming to the Bike Ride,” the parser records a visible lean shift and both his public lean and private top rank follow his own words. Neither controller intent nor another participant’s pressure can move his stance silently.

## Narrowing readiness

Normal `discussion -> narrowing` requires all mandatory conditions: no owed answer, no active repair, no hot hard blocker against the candidate, minimum discussion turns, coverage complete or validly attempted, at least one option with visible support from discussion turns (not just opening), and one realized head-to-head comparison. Then at least one trigger must fire: a visible support cluster, a stable top pair over the configured window, target length or the no-progress threshold with a candidate present, or the approaching hard cap with a viable candidate. The hard-cap override relaxes minimum-turn/coverage/support evidence but never erases an answer obligation or fabricates support.

## Narrowing behavior

Narrowing is bounded: one summary beat (participant-led when possible; moderator-led when the discussion was circling or the target length forced it) plus one holdout reaction beat testing the candidate. If the candidate visibly collapses, the flow returns to discussion exactly once while budget remains; otherwise it proceeds to voting.

## Voting and the repair state machine

During `voting`, every participant produces one clear first formal commitment. First-round votes reveal the participant’s current accepted/public/private stance; they do not probabilistically jump to the group candidate merely because others support it.

After the complete first round:

```text
1. unclear_vote
   -> one bounded clarification/fallback for voters without a valid commitment
2. unanimous
   -> close successful after the active-blocker invariant passes
3. clear majority (more than a one-vote margin over all dissenters combined)
   -> close majority immediately
4. bare majority (winner has exactly one more vote than all dissenters combined)
   -> one moderator concern/willingness question to all dissenters,
      one or two relevant answers in total, one final switch-or-stay commitment
      per dissenter when needed, one retally, then close
5. no-majority split
   -> test one existing option once, target only the minimum number of legally
      movable participants needed for a majority, allow one or two answers,
      collect only candidate-or-current-vote commitments from those movers,
      retally once, close
6. two-person deadlock
   -> one bounded movement opportunity only when movement is plausible;
      otherwise close unresolved
```

For a tied split such as `1-1-1`, formal vote counts are equal, so the single compromise candidate is the tied option with the most positive visible discussion mentions. Further ties are resolved deterministically. For a `2-1-1` split, the plurality option remains primary and only one mover is targeted because one switch is enough for a majority.

A moderator narrowing question is never followed directly by the formal vote call: one relevant participant answers first. During split repair, a targeted participant may only switch to the tested candidate or retain the current vote; selecting an unrelated third option would create a new split and is not a valid repair response.

A hard blocker remains internal. In a bare-majority concern round the moderator treats that participant like any other dissenter and does not reveal the fixed blocker value; the internal rank-1 rule simply prevents an illegal switch. In split-candidate selection, participants who legally cannot move to the candidate are not counted as plausible movers.

The controller never runs a second candidate test or another repair round without a meaningful visible state change. Repeated identical votes are idempotent. Majority is a valid outcome and is not prolonged merely to force unanimity.

## Stubbornness, switch_resistance, and rejection

```text
stubbornness high       = defends hard during discussion, but final movement is separate
switch_resistance high  = very hard to move in narrowing/voting/repair, yet theoretically movable
rank-1 rejection        = hard blocker / cannot accept that option (manual `rejection`,
                          or every non-preferred option for a sampled exclusive blocker)
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
