# 03 — Routing and turn-taking

The router decides who speaks next, what act they perform, who they address, and which option or thread they focus on. This is the central simulator control layer.

## Speaker choice

Speaker choice should combine:

```text
trait-derived turn target
+ local conversation obligations
+ unresolved questions/concerns
+ minority/holdout relevance
+ anti-monologue damping
+ minimum visibility
```

It should not equalize everyone mechanically. Dominant/high-engagement/high-initiative sims may speak more. Quiet sims should not disappear.

## Response obligations

Direct questions should create bounded response obligations. If Sim A asks Sim B a concrete question, Sim B should usually answer soon. This Q→A adjacency is desired and should be preserved.

The problem to avoid is question churn: an answer to topic A should not routinely open topic B before topic A has been developed through agreement, challenge, comparison, or elaboration.

## Same-speaker continuations

Same-speaker continuations are allowed by design. They are valid when they are addendums, corrections, clarifications, afterthoughts, or self-resolutions.

Example acceptable shape:

```text
A: Ben, what do you think about the cooking class?
A: Also, I like it because prep and cleanup are shared.
```

Invalid consecutive turns include:

- re-asking the same addressee the same question;
- repeating the same proposal;
- paraphrasing the previous line without new content;
- accidental monologues caused by routing.

## Direct addressing

Direct addressing is useful but should be sparse. Names should appear when they do real interactional work: asking someone, inviting someone, answering a specific person in a multi-party context, or challenging a prior speaker.

In n=2 discussions, repeatedly opening turns with the other person's name is especially unnatural and should be rare.

## Participant-owned procedure

Participants can perform procedural acts, especially when the moderator is reduced or disabled:

- call for final picks;
- summarize a split;
- probe a holdout;
- suggest narrowing;
- test a compromise candidate.

These moves should be explicit enough to count in metrics, but still short.

## Current validation focus

Check question rate, answer adjacency, direct-name frequency, same-speaker novelty, and trait-shaped dominance on free discussion turns.
