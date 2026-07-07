# 03 — Routing and turn-taking

The router decides who speaks next, which dialogue act they perform, who they address, and which option/thread they focus on.

## Speaker choice

Speaker choice combines:

```text
trait-derived participation pressure
+ response obligations
+ unresolved questions / concerns
+ relevance to current option camps
+ anti-monologue damping
+ minimum visibility for quiet sims
```

The router should not force perfect balance. High-engagement/high-initiative sims may speak more, but quieter sims must still show visible stances.

## Local thread priority

v3 weakens private agenda use. A pending agenda item is only used when no local thread is hot. Direct questions, recent answers, and unresolved concerns should usually be developed before a new issue is introduced.

This keeps discussions from jumping through topics too quickly.

## Trait influence

v3 keeps trait influence small and explainable:

- higher directness increases challenge probability;
- higher compromise tendency increases `propose_compromise` and `soften` probability;
- stubbornness still raises resistance and challenge behavior;
- responsiveness helps with answers and invitations.

## Direct addressing

Names should appear when they do interactional work: asking, answering, inviting, or challenging a specific participant. In n=2, repeated name prefixes are usually unnatural and should remain rare.
