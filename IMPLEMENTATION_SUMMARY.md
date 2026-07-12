# Validation / consensus simplification implementation

Implemented from `todo_validation_consensus_logging_simplification.md`.

Key results:

- deterministic `validation.mode: critical` is the only runtime validation mode;
- normal participant turns make zero validator-LLM calls;
- broad claim-level semantic validation and grounding models/prompts were removed;
- repairs are limited to one correctness-critical attempt;
- comparison fallbacks were removed; only narrow safe fallbacks remain;
- active acceptance, public lean, and formal vote state are separated;
- visible switches replace stale active backing;
- explicit visible soft movement updates public/private stance, while generic support does not;
- clear majorities close immediately;
- bare majorities receive one bounded concern/response/switch round;
- unresolved splits test one candidate once and only required movers revote;
- opposing two-person hard blockers close unresolved without a pointless loop;
- metrics and suite CSV were reduced to a grouped, denominator-safe schema;
- the ten-case suite was updated for critical mode and includes a controlled compromise-heavy case.

Current verification performed:

- `python -m unittest discover -s tests -v` -> 272 passed
- `python -m compileall -q src eval tests main.py` -> passed
- evaluation-suite CLI/import -> 10 valid critical-mode cases
- suite cache/fingerprint version -> `critical-validation-consensus-v8`

The live LLM evaluation suite was not executed in this environment because it requires the configured external endpoints. Run it locally with:

```powershell
py .\eval\run_eval_suite.py
```

## Regression correction after first live suite run

The first packaged revision exposed three integration faults that deterministic
unit coverage had missed:

- `eval/eval.py` referenced `ActType` without importing it, causing the c02
  post-run metrics crash.
- A formal-phase definite acceptance (for example, “Museum works for me”) was
  counted by the transcript tally but stored only as `current_acceptance`,
  causing `vote_state_consistency_failures`. Formal commitment turns now share
  one predicate and update runtime/tally consistently.
- `run_eval_suite.py` printed removed flat metric names, producing misleading
  `None` summary values. It now reads the grouped schema and the suite version
  was bumped so affected v3 rows are not reused.

Regression coverage was added for all three boundaries.

## Chat-quality and grounding closeout

Implemented after reviewing the first complete critical-mode suite logs:

- equal formal-vote ties such as `1-1-1` now test the tied option with the most
  positive accepted discussion mentions;
- split repair targets only the minimum number of legally movable participants
  required to create a majority;
- failed speaker/act/focus routes are recorded, retried with a different speaker
  once, then simplified or retired instead of being issued repeatedly;
- dropped-turn traces retain initial, repaired, fallback, and final rejected
  candidate text;
- participant prompts use a stronger closed-world rule for exact times,
  locations, amenities, capacities, availability, guarantees, capabilities,
  and outside conditions;
- deterministic grounding now catches unlisted exact quantities, explicit
  unlisted feature/location claims, clear attribute contradictions, and
  cross-option value transfers while leaving opinions and reasonable
  implications untouched;
- every fact available to participant generation is now also printed on the
  public option board; hidden/clipped option attributes were removed;
- natural formal-vote forms were expanded without accepting phrases such as
  “back down” as votes;
- compromise success requires a visible split-repair switch that causally
  resolves the no-majority state;
- verbosity metrics now compare configured verbosity, assigned word budgets,
  visible word length, and budget adherence;
- `unsupported_fact_flags` was replaced by the accurately scoped
  `critical_grounding_interventions` metric;
- moderator closure is deterministic and status-aware, preventing a majority
  from being described as unanimity;
- the live suite cache version is `critical-validation-consensus-v8`.

Retrospective application of the new deterministic grounding checks to the
previous ten accepted transcripts flags four clear violations: two unsupported
Senseo claims (counter fit and a “quick brew”), an invented airport lounge, and
a layover-time contradiction. The checks leave grounded compactness implications,
reasonable budget consequences, and explicit uncertainty statements untouched.

## Dialogue evidence and narrowing closeout

Implemented after reviewing the next live suite logs:

- a moderator narrowing question always receives one participant reply before the vote call, including a private hard blocker when that participant is the relevant holdout;
- natural formal commitments such as “I vote X”, “I commit to X”, and “I’m picking X” are parsed without repair, while negative constructions such as “vote against X” remain non-commitments;
- explicit visible movement such as “that settles it for me”, “I’m on board with X”, or “I’m warming to X” updates acceptance/lean state, while conditional wording remains non-final;
- concern detection ignores acknowledgement/resolution language and attributes a negative claim only to the locally mentioned option instead of every intent-focus option;
- split-repair final commitments are candidate-or-current-vote only; no unrelated third option may be introduced;
- mover ranking uses accepted visible openness toward the candidate as a plausibility signal while keeping private ranks out of public support;
- exact values such as `500€` or `2h` are allowed when present on the option card and blocked when altered or unlisted;
- deterministic feature checks include explicit unlisted seating claims, while repair prompting removes unsupported details or states uncertainty naturally rather than appending robotic disclaimers;
- question-sampling/suppression policy was deliberately left unchanged.


## Clause binding, exact-value normalization, comparison focus, and verbosity closeout

Implemented after reviewing the v7 live-suite logs:

- explicit stance phrases bind to the option inside their own clause, so wording such as “I’m leaning toward Park Picnic instead” cannot be reassigned to another option mentioned elsewhere in the turn;
- positive phrases such as “low risk” no longer create concern evidence, and negative predicates attach to the local subject option rather than a later comparison target;
- listed card values accept natural equivalent forms such as `12 euro`/`12 euros`, `45-minute`/`45 minutes`, and `5-hour`/`5 hours`, while altered or transferred values remain blocked;
- travel-time values and total-duration values are matched to their own attributes;
- ordinary comparison intents contain exactly two required options, eliminating three-option focus failures;
- verbosity now produces a wider monotonic word-budget spread, makes short beats much more common for terse participants than verbose participants, and gives range-specific generation instructions: low-verbosity turns make one brief point, while high-verbosity turns develop a reason with a consequence, contrast, or qualification;
- accepted utterances are still never clipped or padded after generation, so visible length remains natural rather than mechanically rewritten;
- suite cache/fingerprint version is `critical-validation-consensus-v8`.
