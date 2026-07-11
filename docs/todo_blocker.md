# Final Prompt/Persona Corrections

## Scope

This is the final small correction block before moving to the dedicated parsing, validation, and fallback review.

Use the current repository as the source of truth. Make only the changes listed below. Do not reopen the completed prompt simplification or redesign the controller.

## [x] 1. Enforce the intended automatic hard-blocker model

> **Done (2026-07-11):** The builder records the sampled blocker
> (`SetupBuilder._hard_blocker_id`, group-level draw in `_trait_rows`, at most one per run)
> and threads it through persona construction. `_normalise_initial_stances(exclusive_blocker=
> True)` forces the assigned preferred option to rank 5 and EVERY other option to rank 1 with
> a grounded reason (LLM reason_against > rejection_reason > card concern > generic exclusive-
> requirement line) — a sampled blocker can no longer remain silently neutral/acceptable
> toward an alternative (the old clamp made the exclusive pattern impossible). Unsampled
> groups keep the movable clamp: LLM-given hard rejections on normal participants are still
> lifted to rank 2. `Persona.hard_blocker: bool` marks the sampled blocker (setup metadata,
> serialized automatically). Tests: exclusive normalization, movable-clamp, group-level
> sampling (exactly one agreeableness-1 row iff sampled). Forced-probability live runs:
> Kira 1×rank-5 + 3×rank-1 (`logs/20260711_225050_061354`), Diego same pattern
> (`logs/20260711_225145_412486`).

When the configured hard-blocker event is sampled:

- exactly one participant in the group must become the hard blocker;
- that participant must have exactly one preferred option;
- every other option must be hard-rejected;
- the preferred option must have rank `5`;
- every other option must have rank `1`;
- the generated participant must not silently remain neutral or acceptable toward another option.

The configured probability remains a low **group-level probability**. If no hard blocker is sampled, normal participants must remain capable of compromise according to their traits and ranks.

Do not make every low-agreeableness or high-stubbornness participant a hard blocker.

## [x] 2. Make the hard-blocker persona internally coherent

> **Done (2026-07-11):** `prompts.setup_personas` takes `hard_blocker_id`; when sampled, the
> rules name that participant as "this group's ONE exclusive hard blocker" and require: the
> assigned option at rank 5, every other option rank 1 with a short grounded reason_against
> each (or one clear exclusive requirement explaining all), a background/private_goal stating
> that non-negotiable requirement as one plausible, politely-held story, and every OTHER
> participant movable (rejection null, no rank-1s). Unsampled runs keep the old
> agreeableness-1 rules. Only agreeableness is pinned by the blocker trait ranges —
> engagement/verbosity stay free; high stubbornness/switch_resistance follow naturally from
> the OCEAN derivation. Representation: multi-option rejection is expressed by the rank table
> itself (already serialized/validated/logged everywhere); the singular `rejection`/
> `rejection_reason` stays as the manual single-rejection input — no schema replacement
> needed. The utterance prompt's blocked line generalizes: an all-but-one-rejected runtime
> renders "Hard constraint: only X is acceptable to them (reason). They reject every other
> option..." instead of naming a single rejection. Live blockers read coherent (vegetarian+
> beverages caterer; privacy/encryption platform) and stayed polite while never accepting an
> alternative.

When a hard blocker is sampled, ensure that the generated persona fields fit the exclusive preference:

- background;
- private goal;
- option stances;
- rejection reasons;
- stubbornness;
- switch resistance.

The preferred option and the rejection of all alternatives must form one plausible story.

Do not force unrelated traits such as engagement or verbosity to extreme values.

A hard blocker may still speak politely, but their decision behavior must consistently reject every non-preferred option.

If the current singular `rejection` / `rejection_reason` representation cannot express several rejected options clearly, simplify or replace it with the smallest coherent representation. Update all directly affected serialization, validation, prompting, and logging code.

## [x] 3. Validate and repair hard-blocker generation

> **Done (2026-07-11):** `_validate_world` (already inside the existing persona retry loop —
> a raise re-tries the batch, never silently downgrades) now enforces the contract: a sampled
> blocker must have exactly one rank-5 option equal to its single preferred option, all
> remaining options at rank 1, and a non-empty grounded reason for every rejection; a
> NON-blocker persona must never carry the exclusive pattern — more rank-1s than its single
> manual/LLM `rejection` allows raises (manual single-rejection profiles still pass).
> Age/profile contradiction checks unchanged and still apply to blockers. Tests: correct
> blocker passes, acceptable-alternative violation raises, missing rejection reason raises,
> accidental exclusive pattern on a normal persona raises, manual single rejection passes.

After persona generation, validate that an automatically sampled hard blocker has:

- exactly one rank-`5` option;
- all remaining options at rank `1`;
- a grounded reason for rejecting each alternative, or one clearly applicable exclusive requirement that explains all rejections;
- no contradiction between background, goal, ranks, and rejection text.

If generation violates the contract, use the existing setup repair/retry path. Do not silently downgrade the participant into a normal persona.

Also validate that a group without a sampled hard blocker is not accidentally given this exclusive rank pattern unless it was explicitly supplied in manual input.

## [x] 4. Preserve visible-evidence-only stance updates

> **Done (2026-07-11):** Focused tests in `tests/test_hard_blocker.py`
> (VisibleEvidenceOnlyStanceUpdates): another participant's visible support for a rival
> option and a direct attack on someone's favorite leave that participant's private ranks
> byte-identical, while the speakers' own visible commitment/objection still updates their
> own state; a participant's own visible acceptance still raises their own rank. This holds
> by construction since the todo_prompt block (`_apply_semantics` touches only the speaker's
> runtime; the former cross-participant commitment erosion was deleted), and is now pinned
> by tests. `tests/test_stance_movement.py` already covers the no-intent-only-movement and
> no-movement-from-blocked/fallback-text sides.

Verify with focused tests that another participant's support or criticism does not directly change someone else's private option ranks.

The intended sequence is:

1. Other participants create public support, criticism, or social pressure.
2. The controller may make the option more relevant, select it as a candidate, or route the participant to respond.
3. The participant visibly accepts, softens, resists, or switches in their own utterance.
4. Only that accepted visible utterance may update the participant's private ranks.

Public pressure may influence the probability or opportunity to switch, but it must not silently rewrite another participant's stance.

## [x] 5. Final verification and documentation

> **Done (2026-07-11):** Tests: 14 new in `tests/test_hard_blocker.py` (+2 alias tests) —
> unsampled group has no blocker, sampled event marks exactly one, exclusive
> normalization/contract validation, coherent-prompt content, manual single-rejection input,
> cross-participant rank isolation, own-acceptance rank change; 301/301 deterministic tests
> pass. Forced-probability live runs (hard_blocker_probability=1.0, config swapped and
> restored): `logs/20260711_225050_061354` — blocker Kira (1×rank-5 vegetarian+beverages
> caterer, 3×rank-1 with grounded reasons) held her exclusive option under a moderator probe
> while the group chose another → **majority** with 0 blocker violations;
> `logs/20260711_225145_412486` — blocker Diego (privacy/encryption story) whose option won.
> Normal eval suite with the committed 0.06 probability: **12/12 pass** (3 successful /
> 8 majority / 1 designed unresolved; 0 unsupported, 0 invalid, 0 dropped turns, 0 blocker
> violations; no blocker sampled at the low probability, as expected). One reliability fix
> the suite forced: the setup LLM systematically pluralizes short aliases ("Board Games" for
> "Board Game ..."), which rejected case q05's scenario three-of-three attempts twice in a
> row — `aliases.validated_short_alias` now accepts trivial singular/plural inflection of the
> name's own words (invented aliases still rejected; tests added). Docs updated: README
> (group-level probability, exclusive semantics, rank-1-only hard blockers), `info/02` (new
> "Hard blockers" section incl. representation and validation/retry), `info/05` (rank-1
> blocker wording + pressure-works-only-through-routing paragraph), and the config.yaml
> comment. `config.yaml` restored to committed values after the forced runs (only the
> intended comment fix remains in the diff).

Add or update focused tests for:

- no hard blocker sampled;
- hard blocker sampled in automatic generation;
- exactly one preferred option and all alternatives rejected;
- coherent hard-blocker persona fields;
- manual hard-blocker input;
- no silent cross-participant rank changes;
- visible acceptance causing a rank change;
- a hard blocker correctly producing majority or unresolved outcomes when the group selects another option.

Run:

1. targeted setup/persona tests;
2. targeted stance-update and switching tests;
3. the full deterministic test suite;
4. several live generations with forced hard-blocker probability for verification;
5. the normal evaluation suite with the configured low probability restored.

Update the existing README and relevant `info/*.md` files so they state clearly:

- the hard-blocker probability is group-level;
- a sampled hard blocker accepts only one option;
- all other options are hard-rejected;
- normal participants may compromise according to ranks and traits;
- other participants create pressure, but private ranks change only after the participant's own visible accepted utterance.

Restore `config.yaml` to its normal committed values after forced verification runs.

## Completion criteria

This correction block is complete when:

- a sampled hard blocker always has one preferred option and rejects every alternative;
- persona traits, background, goal, and rejection reasons are coherent;
- normal groups are not accidentally made uncompromising;
- public pressure never silently changes another participant's ranks;
- visible participant movement remains the only source of private stance updates;
- all tests and evaluation runs pass with the normal configuration restored.