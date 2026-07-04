# Info notes



## A. Current pipeline summary

The current project is already built around a reasonable high-level architecture:

`main.py` calls `DialogueRunner(topic).run()`.

The actual pipeline is:

1. `src/builders.py`
   Generates the scenario, option board, shared context, personas, initial preference assignments, and hard blockers.

2. `src/simulator.py`
   Converts OCEAN/persona traits into operational parameters such as engagement, verbosity, stubbornness, initiative, responsiveness, and compromise threshold. It also builds a small private communicative-goal list per sim — a weak hint system consulted only in quiet moments, not agenda-based user simulation (most items stay pending; see docs/todo.md issue 3).

3. `src/dialogue.py` (+ `src/policy.py`, `src/observer.py`, `src/validation.py`)
   Runs the whole conversation. `DialogueRunner` is the orchestration loop (opening, vote rounds, compromise, moderator turns, closing); it mixes in three concern modules that share the same state — `PolicyMixin` (who speaks / which act / which target, vote readiness, word budgets, style flags), `ObserverMixin` (parse a line, apply visible-state semantics, response obligations), and `ValidationMixin` (turn validation, grounding, deterministic fallback). The moderator voice is configurable via `moderator:` in `config.yaml`.

4. `src/prompts.py`
   Builds prompts for setup, moderator turns, participant turns, repair prompts, grounding checks, and closure.

5. `src/parsing.py`
   Extracts visible option references, addressees, questions, rejections, proposals, and commitments from generated text.

6. `src/consensus.py`
   Computes final outcome from explicit visible votes/acceptances.

7. `src/logger.py` and `src/evaluation.py`
   Save transcripts, JSON logs, token stats, and basic dialogue metrics.

The architecture is described in the `info/` notes (`00_overview.md` is the map; the
rest follow a run: `01` scenario → `02` sims → `03` routing → `04` moderator →
`05` discussion/decision → `06` consensus → `07` evaluation/logging → `08` config →
`09` topic examples). The LLM renders individual utterances; the controller manages
turn-taking, state, and consensus.

As of 2026-07-04 all eight planned `docs/todo.md` items are implemented (one commit
each): explicit `participants:` and `environment:` input modes (auto | manual), an
honest agenda framing, a real evaluation layer, bridge-clause enforcement on
preference switches, trustworthy phase history (no false "closure" markers,
`phase_history` in `run.json`), a configurable moderator, and the split of
`dialogue.py` into policy/observer/validation modules. Invalid generated turns are
replaced by deterministic fallbacks instead of being printed, public stance and vote
readiness come from visible parsed text only, hard shared-context caps are enforced at
setup, and grounding runs behind a regex tripwire.