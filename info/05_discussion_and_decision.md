# 05 — Discussion and decision flow

The discussion moves through:

```text
opening -> discussion -> narrowing/voting -> closure
```

The opening round gives every participant a visible initial stance. It may include a very short chat-like greeting, but its main job is to state the current favorite and one grounded reason.

## Stance movement

Private stance movement is represented by option ranks:

```text
5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 rejected
```

The controller chooses an intended move and effect, for example:

```text
act = soften_toward
option = B
effect = B +1
reason = B answers a concern or has visible group support
```

The LLM renders the utterance. Validation checks that the utterance visibly matches the move. Only then does the observer update the rank table.

## Discussion agenda

During discussion, the controller considers the chat-level agenda, open questions, open concerns, coverage gaps, visible support, and resistance. The agenda is a global checklist, not a per-person plan.

## Split-vote handling

If final votes are split with no majority, the controller may run bounded narrowing:

1. summarize the split;
2. test a plausible candidate;
3. ask holdouts for reservations;
4. let supporters answer once;
5. let holdouts visibly switch, stay, or name one alternative;
6. close as successful / majority / unresolved.

A future extension may test a second compromise candidate, but it should stay bounded and should not create an endless loop.

## Stubbornness vs rejection

Resistance to switching, compromise pacing, and holdout behavior are all controlled by `stubbornness`:

```text
stubbornness high = very resistant, but theoretically movable
rejection true    = hard blocker / cannot accept the rejected option (rank 1)
```

High-stubbornness sims defend longer and switch less often; low-stubbornness sims compromise and soften more easily. A hard blocker never comes from stubbornness alone — only `rejection` and option-rank 1 are binding.

## Speech-style boundary

Age/speech_style may change how a participant says a point. It must not change what the controller asks them to do, which option they prefer, or whether a switch is plausible.

## What must not happen

- no hidden consensus from private ranks;
- no invented blended option;
- no forced successful outcome when a rank-1 blocker remains;
- no invalid line printed as transcript evidence;
- no endless negotiation loop;
- no speech-style feature overriding parameter-driven behavior.
