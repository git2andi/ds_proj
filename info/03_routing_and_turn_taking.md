# 03 — Routing and turn-taking

The router decides who speaks next, what act they perform, who they address, and which option or thread they focus on. This is the central simulator control layer.

## Speaker choice

Speaker choice should combine:

```text
trait-derived turn target
+ local conversation obligations
+ unresolved questions/concerns
+ minority/holdout relevance
+ anti-monopoly damping
+ minimum visibility
```

It should not equalize everyone mechanically. It should also not rely only on traits.

## Response obligations

Direct questions should create bounded response obligations. The addressed sim should usually answer soon, but the system should avoid turning one sim into an interview loop.

## Same-speaker continuations

Rare same-speaker continuations are allowed by design, including chains up to three messages. They are valid only when they are addendums, corrections, clarifications, afterthoughts, or self-resolutions.

Invalid consecutive turns include:

- re-asking the same addressee the same question;
- repeating the same proposal;
- paraphrasing the previous line without new content;
- accidental monologues caused by routing.

## Participant-owned procedure

Participants can perform procedural acts, especially when the moderator is reduced or disabled:

- call for final picks;
- summarize a split;
- probe a holdout;
- suggest narrowing;
- test a compromise candidate.

These should be explicit enough to count in metrics.

## Current open issues

Participant-owned procedure is visible, but still crude. It can call votes and summarize splits, but post-split negotiation remains too shallow. Split-vote candidate selection and post-reservation decision routing are the current priorities.
