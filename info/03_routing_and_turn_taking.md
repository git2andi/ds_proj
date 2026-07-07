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

These moves should be explicit enough to count in metrics, but still short. Participant-owned procedure must sound like a group member, not the controller: no exact vote-count dumps, no candidate-testing vocabulary ("test the least-blocked candidate"), and never addressing the speaker themself. Exact procedural summaries with counts belong to moderator turns only.

Participant split summaries draw from trait-colored variant pools keyed to the
caller (P5): a direct caller says "So it's A and B and nobody's moving. Would X
actually work?", a compromising one "Feels like X is the one fewest of us
mind", a high-initiative one "Let's stop circling. If we said X, …". The
functional structure (name the plausible common ground, ask the holdouts) is
constant; only the social wording varies. The caller of a split summary is
never the first holdout to answer it (M2) — the addressed holdouts respond
first, and the caller voices their own reservation only afterwards.

## Agenda priority (P6)

The private agenda is a weak hint consulted only in quiet moments. The
effective hierarchy is: answer a pending direct question, react to a fresh
challenge/concern, develop the current thread, trait-shaped free acts — and
only then, when the local thread is cold (no question on the floor, no
unreacted answer, no unaddressed open concern), a probability-gated agenda
item (base 0.15 + 0.25·initiative). Observed agenda-driven turns are roughly
10-15% of participant turns; the agenda must never create checklist rhythm.

## Micro-reactions (P4)

The discussion loop occasionally (probability-gated, per-run cap of
`max(2, n//2 + 1)`) inserts a deterministic tiny reaction beat after an
answer, challenge, agreement, build, compromise, or softening turn: "Fair.",
"Same here.", "Not convinced.", "That's my worry too.". The polarity comes
from visible state — a sim whose current pick was just attacked resists, one
whose pick got support agrees, a challenge on a rival draws shared concern —
so even a two-word turn contributes socially. Every pool line is option-free,
so the parser can never read a commitment into one; the beats never fire
after a question, over a pending direct-answer obligation, or on top of
another micro turn. They cost no LLM call.

## Trait-colored delivery (P2)

Traits shape acts and phrasing during the discussion, not only at vote time:

- Act mix: `challenge` scales with stubbornness and directness, `propose_compromise`
  with compromise tendency, `soften` inversely with stubbornness. A compromising
  sim bridges early; a stubborn one does not drift into narrowing moves by chance.
- `MoveIntent.trait_color` carries a compact delivery label rendered as one
  prompt line: `challenge_directly` (direct sims: one sharp objection, no
  concession preamble), `soften_and_bridge` (gentle sims: acknowledge first,
  then the worry), `bridge_condition` (compromisers: name the condition that
  would make the option acceptable), `restate_concern` (see below).
- Stubborn restate: once per run, a high-stubbornness sim's ordinary turn
  becomes a challenge that brings their core concern back in fresh words —
  against a recorded objection target if one exists, else the largest rival
  camp — and holds their own pick.
- Concern attribution: a challenge registers its concern against the rival, not
  against the speaker's own pick when that name merely appears first in the line.
- Downhill switches are blocked: a narrowing beat never moves a sim to a camp
  with fewer visible votes than their own, so ultra-flexible sims cannot
  ping-pong between tested candidates.

## Current validation focus

Check question rate, answer adjacency, direct-name frequency, same-speaker novelty, and trait-shaped dominance on free discussion turns.
