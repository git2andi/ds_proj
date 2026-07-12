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
- a visible compromise offer or conditional support by the speaker may move their own lean, gated by stubbornness (an already-acceptable target moves more easily);
- a hard rejection (rank 1) is never silently removed — only the speaker's own visible resolution reopens it;
- controller intent alone never changes a rank, and repaired-away, fallback-blocked, or dropped text moves nothing.

`discussion_lean_shifts` counts every accepted turn that changed the speaker's top option during the discussion phase, whatever visible path caused it.

Other participants' turns never rewrite someone else's private ranks. Public support, criticism, or pressure works only through routing and opportunity: the controller may make an option more relevant, select it as a candidate, or route the pressured participant to respond — and only that participant's own accepted visible utterance may then update their ranks.

## One participant turn

```text
1. controller selects the speaker and a MoveIntent (act, objective, focus, target)
2. the dialogue LLM realizes exactly that move inside an <utterance> envelope
3. conservative extraction (structural only — never cut, never de-tailed)
4. deterministic critical layer: option/alias/addressee/pronoun resolution,
   strict commitments with post-checks, explicit blockers, genuine questions
5. selective validation (validation.mode, default selective): AT MOST one
   validator-LLM call, made only when soft natural-language meaning can change
   state, requesting just the semantic categories the intended move needs plus
   grounding claims. Simple fully-verifiable turns (direct votes, sanctioned
   switches, blocker restatements, process/closing lines, mention-free light
   comments) skip the call via explicit fast paths, each traced with a reason.
   Deterministic verification checks every span, id, critical commitment, and
   grounding claim (fact table) before anything counts.
6. assessment decides: ACCEPT / ACCEPT_WITH_METRIC / REPAIR / FALLBACK / DROP;
   repair triggers only on blocking failures — an unrealized soft function is
   telemetry, not a repair
7. at most one targeted repair, then a TRUTHFUL deterministic fallback (vote/
   switch, blocker restatement, coverage request, factual comparison, listed or
   does-not-say answer) or a dropped turn — every candidate goes through the
   same complete path
8. only the final accepted evidence object updates ranks, votes, threads,
   coverage — and consensus/public support read the SAME object
```

Concrete example: the controller routes Jonas a CONCERN about the Museum ("name the one concrete thing that still blocks Museum for you"). The dialogue LLM returns `<utterance>The Museum's 24 euros is steep for a quiet afternoon, honestly.</utterance>`. Extraction unwraps the envelope; the validator reports one ordinary concern bound to the Museum with span "24 euros is steep for a quiet afternoon" and one listed-fact claim ("24 euros", verified against `A.cost`); assessment accepts; the observer opens a `concern` thread on the Museum with issue key `cost`, counts an objection for coverage, and leaves every rank unchanged — Jonas objected, he didn't move.

Multi-function turns keep all their evidence: "I still dislike the Museum's price, but I'm switching to the Bike Ride because it's cheaper. Would that work for everyone?" carries a concern about A, a vote commitment to B, a switch with a visible reason, and a group question — none erases another, and the trailing check-in question does not void the commitment.

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
