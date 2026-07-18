# Evaluation and logging

## Run artifacts

Each run writes a human-readable transcript and a structured `run.json`. The JSON contains:

- scenario, aliases, and complete persona cards;
- provider/model provenance, seed, configuration hash, Python version, and Git revision when available;
- visible turns and optional structured action traces;
- generation attempts, repairs, dropped turns, fallbacks, and deterministic vote records;
- public preferences, acceptances, movements, point-use state, votes, outcome, and review flags;
- process and token counters.

Invalid text is not inserted into the visible transcript, but failed attempts remain available for debugging.

## Deterministic evaluation

`eval/summarize_runs.py` reads existing `run.json` files without calling an LLM. It produces:

- `deterministic_runs.csv`;
- `trait_participants.csv`;
- `trait_levels.csv`;
- `evaluation_summary.md`.

The summary covers completion, outcome distribution, protocol and vote consistency, participant and moderator turns, voluntary participation, visible movement, realization failures, LLM calls, token use, and trait realization. Engagement is compared with voluntary floor share relative to equal participation; verbosity is compared with words in generated non-vote turns; stubbornness is compared with visible flexibility; directness uses an optional lexical hedge-rate proxy.

## LLM transcript judging

`eval/judge_transcripts.py` is a separate post-hoc evaluator. Three role-conditioned referees receive the same scenario, option board, complete persona cards, visible transcript, votes, and outcome. They score naturalness, coherence, groundedness, persona consistency, and deliberation quality on a five-point scale. Moderator turns affect the interaction-level dimensions but are excluded from persona consistency.

The judge never runs during dialogue generation. Its CSV files are written incrementally. Complete panels matching the requested provider, model, judge count, and prompt version are skipped, so an interrupted judge run can safely resume without deleting previous results.
