# Sim generation (the participants)

**Code:** `src/builders.py` (`SetupBuilder`, persona path),
`src/simulator.py` (`derive_simulator_parameters`, `build_initial_agenda`),
`src/prompts.py` (`setup_personas`), `src/models.py` (`Persona`, `TraitProfile`,
`SimulatorParameters`, `AgendaItem`).

A participant is not just a name and a persona sentence. It is a small **user
simulator**: OCEAN traits → explicit behavior parameters, a private preference and
goal, optional hard rejection, and a weak private communicative-goal list.

## Three layers of a simulated user

```text
identity        name, one-sentence background, private goal
personality     OCEAN traits (openness, conscientiousness, extraversion,
                agreeableness, neuroticism), 1–5
operational     behavior parameters that actually drive routing and style:
                engagement, verbosity, initiative, responsiveness,
                stubbornness, directness, compromise_threshold  (each 0–1)
```

OCEAN traits produce plausible individual differences, but they are too abstract to
be the control interface. The **behavior parameters** are what the controller reads.

## OCEAN → behavior parameters

`simulator.derive_simulator_parameters` maps traits (each normalized to 0–1) into
parameters. The formulas are pragmatic, not psychometric — the goal is a stable,
*observable* spread across generated dialogue:

```text
engagement     = 0.25 + 0.60*extraversion + 0.15*conscientiousness
verbosity      = 0.20 + 0.55*extraversion + 0.25*openness
initiative     = 0.20 + 0.50*extraversion + 0.30*openness
responsiveness = 0.30 + 0.45*agreeableness + 0.25*conscientiousness
stubbornness   = 0.10 + 0.60*(1-agreeableness) + 0.30*neuroticism
directness     = 0.25 + 0.35*conscientiousness + 0.25*extraversion + 0.15*(1-agreeableness)
compromise_threshold = 1 - compromise_willingness   (low threshold = compromises easily)
```

These feed concrete behavior: verbosity/engagement set the per-turn word budget
(`03`), stubbornness/agreeableness bias act selection and whether a sim will move,
compromise_threshold gates stance shifts (`06`).

## Two ways to create the cast

Set `participants.mode` in `config.yaml`:

- **`auto` (default)** — sample everything: names from a pool, traits from
  `personas.trait_ranges`, initial preferences from a configured split distribution,
  and backgrounds/goals from the persona LLM call.
- **`manual`** — define the cast under `participants.profiles`; the group size equals
  the number of profiles (`simulation.num_participants` is ignored). Profiles may be
  **partial** — any missing field is filled by the auto path. A profile can pin a
  `name`, `description`, `private_goal`, `preferred_option`, `traits`, direct
  `parameters:` overrides, and a `rejection` (hard blocker).

If **every** profile is complete (name + description + private_goal +
preferred_option), the **persona LLM call is skipped entirely** — a fully
deterministic cast for controlled experiments. `run.json` records
`participants_mode`.

## Initial preferences are assigned by the controller

The controller decides each sim's initial primary preference *before* prompting
(`_preference_assignments`), so the group starts from a chosen split (e.g. `2-1` =
two sims prefer one option, one prefers another). In auto mode the split is sampled
from `personas.preference_distribution.shape_weights[n]`; a manually pinned
preference bypasses the distribution (unpinned profiles then get a uniformly random
option, never their own rejection). A persona row that drops or reorders the required
primary is *repaired* deterministically (`repair_preferred_options`), not retried.

## Private vs public state

The distinction is central to the whole project (see `06`, `07`):

```text
private / internal (guides behavior, never counts as agreement)
  initial preference, current lean, hard rejections, soft concerns,
  private goal, behavior parameters, agenda items

public / visible (the only thing the outcome may use)
  visible option mentions, objections, questions, answers, commitments/votes
```

## Hard blockers

A hard blocker is a sim with `agreeableness = 1` who **cannot accept** a specific
rejected option. In auto mode one may appear at random
(`personas.hard_blocker_probability`); in manual mode a profile with a `rejection`
becomes one (agreeableness pinned to 1, `rejection_reason` required). The blocker's
rejected option is never assigned as its own primary preference, and no state
mutation can make it vote for that option (`06`).

## The agenda — an honest caveat

Each sim gets a small private **communicative-goal list** (`build_initial_agenda`):
state a grounded reason for the initial pick, ask about a constraint, compare with a
rival, and (trait-dependent) raise a hard concern, look for compromise, or push back
on a rival. `refresh_agenda` marks items obsolete/blocked as the sim's stance moves.

**Status:** this is a *weak hint system*, not agenda-based user simulation. The
router consumes an item **only in quiet moments** — reactive rules and response
obligations always win first (`03`) — and observed runs leave most items pending at
the end. Do **not** describe the project as an "agenda-based user simulator". A real
goal stack (each turn consumes/defers/updates one item) is a possible future
direction, tracked in `docs/todo.md` issue 3.

## Voice differentiation

Sims should sound different because their parameters shape utterance style. The
per-turn prompt sends a compact "voice capsule" derived from the parameters
(`prompts._voice_guidance`), not a raw trait dump:

```text
high directness:  "Too pricey. Not worth it."
low directness:   "I see why D is attractive, but I'm a bit worried about the timing."
high stubbornness: keeps returning to its own priority, concedes little
high verbosity:   happily adds a second clause or a small aside
low engagement:   dry and minimal; speaks less and more briefly
```

The goal is not theatrical role-play — it is measurable variation in participation,
wording length, consistency, and compromise behavior (see the realization metrics in
`08`).
