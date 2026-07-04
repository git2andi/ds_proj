# Routing and turn-taking (the controller)

**Code:** `src/policy.py` (`PolicyMixin`), `src/dialogue.py` (the loops that call it),
`src/style.py`, `src/simulator.py` (agenda), `src/models.py` (`MoveIntent`).

This is the heart of the simulator. Every participant turn is produced by a small
control step — the LLM never chooses *who* speaks or *what move* to make, only how to
phrase the move it is handed.

## The per-turn loop

```text
read the visible state
choose speaker            (who talks next)
choose dialogue act       (build / agree / challenge / ask / answer / compare /
                           invite / propose_compromise / vote / …)
choose target + addressee (which earlier turn / person this responds to)
choose option focus       (which options the move is about)
build a MoveIntent        (a compact instruction object)
-> LLM realizes ONLY that one message   (see 05 for the generation+repair pipeline)
-> observer parses the result           (see 06)
-> visible state is updated
```

The `MoveIntent` is the contract between controller and LLM: it carries the speaker,
act, a plain-language `reason`, option focus, optional addressee, and style flags. It
never contains hidden metadata that could leak into the text.

## Choosing the act — reactive first, agenda last

`_route_discussion_turn` decides the move in a fixed priority order, so local context
drives the conversation and the private agenda only fills silence:

1. **Response obligation** — if someone was directly asked a question, the addressed
   sim answers next (see "Direct questions" below).
2. **Option coverage** — each option gets at most one light "has this been discussed?"
   nudge before voting (`_coverage_gap_option`; bounded by `coverage_attempts`).
3. **Reactive intents** (`_reactive_intent`) — adjacency-pair moves driven by what
   just happened, each behind a probability gate so runs don't become a script:
   - a **challenged** option gets **defended** by an advocate,
   - an **answer** gets a **follow-up**,
   - an unresolved **blocker** on the leading option gets **probed once**,
   - a visible **split** triggers a head-to-head **comparison**,
   - a **circling** thread (several turns, no question/movement, two camps) gets one
     bounded compromise/ask beat (`stagnation_break_done`).
4. **Agenda / free choice** — only now does the speaker maybe consume a private
   agenda item (probability rises with initiative), otherwise a trait-weighted act is
   picked (`_choose_discussion_act`).

Challenge reasons are stance-aware: a sim is never routed to argue against its own
current pick.

## Choosing the speaker — trait-weighted participation

Each sim has a **target turn share** derived from its parameters
(`simulator.expected_turn_share`: engagement dominates, initiative and
responsiveness tilt it, plus a floor so nobody's target is zero). `_choose_speaker`
weights each candidate by `engagement`/`initiative` and then pulls the *actual* turn
share toward that target (`exp(trait_share_adaptation * (target − actual))`): a sim
behind its target gets boosted, a sim ahead gets damped. Guard rails:

- **no immediate self-repeat** (the last speaker is skipped when others exist),
- **anti-monopoly**: a share more than `max_share_overshoot` above target is damped
  hard — high engagement may lead the room, never monologue,
- **minimum visibility**: a sim silent for `max_silence_rounds` full rounds is
  pushed back in regardless of traits — low engagement means quieter, not absent,
- a penalty on the second-to-last speaker to stop two people ping-ponging.

Structural turns (the opening round and vote rounds give everyone exactly one turn)
compress the realized spread toward equality, so trait differences show most in the
free discussion turns. In a controlled n=3 run with engagement pinned to
0.9/0.5/0.15, turn counts came out 9/8/7 with average words 29/19/14 and an
engagement–behavior correlation of ≈ +1.0.

**Responsiveness** shows up in answer latency: when a sim owes an answer to a direct
question, it replies immediately with probability `0.45 + 0.55 * responsiveness`,
and may otherwise sit out exactly one beat before the router forces the answer —
hesitation never lets the question lapse.

With a **corpus preset** active (`08`), this switches to share-aware weighting: one
sim is allowed to dominate within configured bounds instead of the trait targets.

## Choosing the target — thread-scored, not just "the last line"

`_choose_target_turn` scores the last few participant turns instead of always
replying to the most recent one. Open questions, objections/blockers, minority
voices, and turns about the leading or under-discussed options outrank plain
recency — so earlier unresolved points get revisited instead of dying after one
reply. An `answer` act deterministically targets the pending question.

Not every turn names an addressee. Names are used when they are functional
(answering, challenging a specific person, inviting a quiet participant), and
suppressed otherwise so the transcript doesn't read as "Name, … / Name, …".

## Direct questions create response obligations

When a turn visibly asks a direct question (detected from text by the parser, not
from the act label), the observer records a `ResponseObligation` on the addressed
sim. The router consumes it **before** normal speaker selection in both the
discussion and decision loops, so the addressed participant answers within the next
turn or two. Obligations expire after a bounded window and are counted as
`unanswered_direct_questions` (a metric, `08`).

```text
Kenji -> Anton: "Anton, is the no-checked-bag issue a deal-breaker?"
next relevant turn: Anton answers   (not: someone else casts an unrelated vote)
```

## Word budgets are trait-driven

`_word_bounds` sets each turn's soft length target from the speaker's parameters
(`0.45 + 0.70*verbosity + 0.15*engagement`, plus per-turn jitter), so terse sims stay
short and chatty ones run longer. Decision/switch turns get extra room for a bridge
clause. Budgets are style targets, not hard cuts — `utils.clean_generated` keeps a
complete sentence within a soft cap so no printed line ends mid-thought.

## Surface-style control (deterministic, no LLM call)

`src/style.py` tracks recent participant turns and sets compact prompt flags on the
`MoveIntent` to keep dialogue varied:

- suppress a leading name prefix when name density is high,
- suppress opening on an option name / "I …" / "We …" when those openings cluster,
- vary the opening word when recent turns all start the same way,
- steer act selection away from a repeated concession/worry/trade-off template.

For decision turns it also builds `avoid_phrases` (commitment-phrase families already
used this round or by this speaker earlier) and `avoid_reasons`, so re-asked voters
don't repeat themselves verbatim across vote rounds.


