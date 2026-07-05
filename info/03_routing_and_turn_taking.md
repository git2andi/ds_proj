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

Split summaries in no-moderator mode are now deterministic participant-owned procedure: the controller appends a visible participant line with the exact vote counts and candidate. This prevents LLM paraphrases from changing the candidate and saves one utterance/grounding call. Other participant procedure still uses normal routed utterances.

These moves should be explicit enough to count in metrics.

## Current validation focus

Participant-owned procedure is visible, but post-split negotiation still needs live validation. Check that no-/light-moderator cases test the visible leader or best tied candidate and that holdouts produce explicit switch/stay/alternative decisions.
