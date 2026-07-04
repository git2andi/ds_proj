# Scenario generation (the environment)

**Code:** `src/builders.py` (`SetupBuilder`), `src/prompts.py` (`setup_scenario`),
`src/aliases.py`, `src/models.py` (`Scenario`, `OptionCard`).

Scenario generation turns a one-line topic into a concrete decision environment: a
set of factual option cards plus shared context. A topic like *"Choose a coffee
machine for the office"* has nothing to reason about on its own; the scenario gives
the world a fixed fact base. The options are invented, but **once generated they are
the hard facts of the run**.

## Two ways to build the environment

Set `environment.mode` in `config.yaml`:

- **`auto` (default)** — the setup LLM turns the topic (CLI arg / stdin) into the
  scenario.
- **`manual`** — you author the whole environment under `environment.manual`; no
  scenario LLM call runs, and any CLI topic is ignored. Use this to pin a fixed
  world and reuse it across runs for controlled comparisons.

`run.json` records which mode was used (`environment_mode`).

## What a scenario contains

```text
topic              the decision being made
decision_kind      a coarse label (restaurant_choice, tool_choice, …)
opening_question   one casual question about priorities/trade-offs
shared_context     2–3 facts everyone knows (budget, timing, group size, constraints)
options            exactly len(scenario.option_labels) cards (default 4: A–D)
```

Each **option card** (`OptionCard`):

```text
id            A / B / C / D
name          specific, realistic, a complete noun phrase
short_name    1–2 word nickname used in dialogue (validated alias, see below)
attrs         3–5 concrete attributes with stable values (cost/time/effort/…)
upside        one specific benefit
tradeoff      one specific downside
concern       a stable objection someone could raise
best_for      the priority this option serves
```

Only `name` + at least one attribute are strictly required in manual mode; the rest
are optional there.

## Auto mode: two small LLM calls

`SetupBuilder.build(n)` runs setup in two steps (kept separate so each call is small
enough to be reliable on slower endpoints):

1. **Scenario call** (`_generate_scenario`) → the option board + shared context.
2. **Persona call** (`_generate_personas`) → the participants (see `02`).

Both steps retry up to `simulation.setup_generation_attempts` times on malformed
output; if a valid world can't be produced, `build()` **raises** rather than
fabricating one. Cosmetic issues (e.g. an over-long option name) are cleaned
(`_clean_name`), not treated as failures.

## The source-of-truth rule

After generation, the option cards and shared context are the only facts that exist.
Sims may compare, reason, and express uncertainty from them — but must not add new
concrete facts.

```text
Allowed:
  "D is cheapest, but a red-eye sounds uncomfortable."
  "We don't know if baggage is included, so I wouldn't assume it."

Not allowed (unless the card/context says so):
  "The direct flight includes checked bags."
  "The venue already has catering."
```

Enforcement of this rule at discussion time is the grounding check in `05`/`06`.

## Hard numeric caps are enforced

If the shared context states a hard cap ("budget capped at $300", "within 20
minutes", "under 2 hours"), every option must satisfy it — an invalid option must
never be able to win the discussion. This lives in `builders.py`:

- `shared_context_caps` extracts hard caps, ignoring soft phrasings ("around $200")
  and normalizing units within a family (miles↔km, hours↔minutes). A cap that names
  an activity ("within 15 minutes *walking*") only binds attributes about that
  activity.
- `enforce_shared_caps` compares a cap to an attribute only when their per-unit
  basis matches (a "$500 total" budget never clamps a "cost per person" value).

In **auto** mode, early attempts *regenerate* on a violation (rewriting a number can
fabricate a false fact about a real-world named option); only the final attempt
clamps in place, recording the repair in `Scenario.setup_notes`. In **manual** mode,
a violating card is a **config error** — the run fails with the violation list; the
author's numbers are never rewritten.

## Group-size consistency (auto only)

If the topic or shared context explicitly names a group size ("a team of four"), it
must match `simulation.num_participants`, or setup fails fast with a clear message
(`_validate_topic_participant_count`, `_validate_participant_references`). These
guards are **not** applied to manual environments: there, a fact like "25 colleagues
will attend" describes the scenario, not the deciding group.

## Option nicknames (aliases)

`src/aliases.py` owns the single alias contract (`short_alias_map`): the 1–2 word
nickname each option is called by in dialogue and how the parser recognizes it.
Aliases deliberately exclude stopwords and generic words ("with", "data",
"analytics", "warehouse", …) so they never match by accident, and include
distinctive proper nouns/brands ("Gin", "Rails", "FastAPI"). This matters because
the observer reads votes and objections by matching these aliases in the text (`06`).

## Moderator framing

When the moderator opening is enabled (`04`), it presents the board honestly as
generated assumptions that are binding for the run:

```text
For this simulated decision, I'll treat the following setup as the shared facts.
```

When the opening is disabled, the same board is still shown as plain setup
scaffolding (and always appears in the transcript's `## Options` section).
