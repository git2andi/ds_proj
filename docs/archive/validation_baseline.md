# Validation-migration baseline (todo_validation item 1)

Captured 2026-07-12, before any migration change. Later items compare against
these numbers. Bounded live runs used `dialogue=gpt`, `validator=gpt`
(gpt-4.1-mini), 3 participants, default config.

## Deterministic state

- Test suite: `py -m unittest discover -s tests` — **464 tests OK** before the
  baseline reproductions; **478 OK (7 expected failures)** with them.
- `py -m compileall -q main.py src eval tests` — clean.

## Production LOC (affected subsystem)

| file | LOC |
|---|---|
| src/parsing.py | 920 |
| src/prompts.py | 851 |
| src/models.py | 744 |
| src/interpreter.py | 744 |
| src/dialogue.py | 694 |
| src/observer.py | 646 |
| src/validation.py | 487 |
| src/consensus.py | 181 |
| main.py | 85 |
| src/aliases.py | 74 |
| **subsystem total** | **5426** |

(builders.py 1102 and controller/* are touched but not migration targets.)

## Cost baseline (bounded live runs)

Run 1 — "Book a flight from Miami to Stockholm" (`logs/20260712_024041_245430`),
27 accepted participant turns, outcome majority:

| metric | value |
|---|---|
| validator calls | 33 (**1.22 per accepted turn**) |
| validator tokens in/out | 49,353 / 9,107 |
| avg validator input per call | **1,496** |
| dialogue (utterance+repair+moderator) in | 21,172 |
| validator share of total input | **0.672** |
| repair calls / rate | 7 / 0.259 |
| fallback turns / dropped turns | 1 / 0 |
| intended_function_realized_rate | 0.909 |
| intended_focus_agreement_rate | 1.0 |
| act_mismatch_rate (diagnostic) | 0.185 |
| unsupported_printed_turns | 0 |

Run 2 — "Choose a project management tool for a five-person startup"
(`logs/20260712_024339_473682`), 28 accepted turns, outcome majority:
35 validator calls (**1.25 per accepted turn**), validator 52,204/7,750
tokens (avg **1,492** in per call, **0.691** share of total input), repair
rate 0.107, 2 fallbacks, 1 dropped turn, realized rate 1.0, focus agreement
0.964, unsupported printed turns 0.

**Aggregate baseline: ~1.22–1.25 validator calls per accepted turn,
~1.49k validator input tokens per call, validator ≈ 67–69% of total input
tokens (≈ 2.4× dialogue input).**

Repair triggers (run 1): COMPARISON_MISSES_OPTIONS ×2,
ANSWER_DOES_NOT_ADDRESS_QUESTION ×2, UNSUPPORTED_CLAIM:cross_option_transfer,
UNSUPPORTED_CLAIM:invented_detail, UNBRIDGED_SWITCH. Two accepted turns still
printed with `ANSWER_DOES_NOT_ADDRESS_QUESTION` (non-blocking) — item 11's
"no blocking issue printed after repair" applies.

## Reproduced failures (regression coverage)

All reproductions live in the test suite as `expectedFailure` markers to be
flipped by their fixing item:

- `tests/test_cli.py` — explicit/piped CLI topic silently discarded when
  `environment.mode=manual` (item 2). File/pipe parsing behavior itself works
  and is pinned by passing tests.
- `tests/test_alias_repair.py` — an invalid generated `short_name`
  ("London Stop" for "Lufthansa Flight via London") discards the whole
  substantively valid scenario and exhausts all setup attempts (item 3).
- `tests/test_evidence_authority.py` — dual semantic authority:
  `consensus.public_support()` reads legacy `act.accepts` / `act.explicit_vote`
  / `act_type is SUPPORT`, so evidence-only support/acceptance is invisible to
  public support while a legacy SUPPORT label without validated evidence still
  creates support; proposal counts read `act.offers_compromise` (item 4).

## Live dual-authority example (run 1, turn 15)

> "That settles it for me—American's timing is a workable tradeoff."

Validated evidence: commitment `accept A`. Legacy act: `comment`, no
`explicit_vote`, no accepts — `public_support()` misses the acceptance the
observer recorded. 14 act/evidence divergences in 27 turns overall
(mostly harmless label differences; the commitment case above is the
state-relevant one).

## Post-migration LOC accounting (item 13)

After the migration the same subsystem measures 5,526 LOC (was 5,426). The
semantic-authority code shrank as intended — parsing.py 920→788 (soft-semantic
regex catalogs moved to the test-only stub validator in
`tests/evidence_adapter.py`), observer.py 646→625 (all reparsing removed),
DialogueAct lost its 8 evidence-duplicating fields — but the TODO's own new
features added offsetting lines: selective-mode fast paths and the
deterministic critical layer in interpreter.py (744→911), CLI precedence in
main.py (85→97), the alias-repair prompt in prompts.py, and richer validation
telemetry in dialogue.py (694→733). There is now ONE semantic path and no
production compatibility adapter; the cost reduction shows in tokens (below),
not in raw line count.

## Targets set by the TODO — final results (2026-07-12, item 15 gate)

Measured on the two focused gpt/gpt gate runs with the completed migration
(`logs/20260712_035642_125853`, `logs/20260712_035852_830187`):

| target | baseline | final | status |
|---|---|---|---|
| validator calls per accepted turn < 1.0 | 1.22–1.25 | **0.885** | PASS |
| avg validator input per call ≪ 1.4–1.5k | ~1,490 | **~740** | PASS |
| validator input ≤ dialogue input | 2.3–2.4× | **0.965×** | PASS |
| repair rate materially below 0.26 | 0.11–0.26 | **0.00–0.04** | PASS |
| dual-authority discrepancies | present | **0** (`vote_state_consistency_failures`) | PASS |
| blocking issue printed after repair | 2/run | **0** | PASS |
| unsupported printed claims | 0 | **0** | PASS |
| one semantic authority, no compat adapters | no | **yes** | PASS |
| subsystem LOC materially lower | 5,426 | 5,526 (see accounting above) | tokens, not lines |

Validator share of total input dropped from ~67% to ~45–50%. The focused
sample set also covered: explicit/file/piped CLI topics, the configured
manual environment with manual participants (hard blocker binding, 0
violations), live alias-only repairs (2 of 5 setup builds), and
validator-outage fail-closed behavior (stubbed tests). A cross-provider run
(dialogue=gpt, validator=gemini) was attempted but the gemini endpoint was
unreachable, so cross-provider behavior remains verified only by the
provider-agnostic unit tests; both roles are configured to `gpt`. Full
deterministic suite: 512 tests OK.
