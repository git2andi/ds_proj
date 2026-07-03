# Info notes



## A. Current pipeline summary

The current project is already built around a reasonable high-level architecture:

`main.py` calls `DialogueRunner(topic).run()`.

The actual pipeline is:

1. `src/builders.py`
   Generates the scenario, option board, shared context, personas, initial preference assignments, and hard blockers.

2. `src/simulator.py`
   Converts OCEAN/persona traits into operational parameters such as engagement, verbosity, stubbornness, initiative, responsiveness, and compromise threshold. It also builds a small private communicative-goal list per sim — a weak hint system consulted only in quiet moments, not agenda-based user simulation (most items stay pending; see docs/todo.md issue 3).

3. `src/dialogue.py`
   Runs the whole conversation. It creates the opening, controls participant turns, selects speakers, selects speech acts, chooses addressees/target turns, handles moderator interventions, starts vote rounds, attempts compromise, and closes the run.

4. `src/prompts.py`
   Builds prompts for setup, moderator turns, participant turns, repair prompts, grounding checks, and closure.

5. `src/parsing.py`
   Extracts visible option references, addressees, questions, rejections, proposals, and commitments from generated text.

6. `src/consensus.py`
   Computes final outcome from explicit visible votes/acceptances.

7. `src/logger.py` and `src/evaluation.py`
   Save transcripts, JSON logs, token stats, and basic dialogue metrics.

The intended architecture in `info/00_overview.md`, `info/03_agentic_behavior (1).md`, and `info/07_consensus_and_outcomes (1).md` is implemented: the LLM renders individual utterances, while the controller manages turn-taking, state, and consensus.

As of 2026-07-03 the transcript–state integrity refactor is complete (see `docs/todo.md`, section 4): invalid generated turns are replaced by deterministic fallbacks instead of being printed, public stance and vote readiness come from visible parsed text only, hard shared-context caps are enforced at setup, targeting/act selection are thread- and adjacency-driven, moderator vote calls are option-neutral, and grounding runs behind a regex tripwire (n=3 runs ≈ 15–20k input tokens). Deeper evaluation metrics are deliberately deferred until the discussion behavior is considered final.