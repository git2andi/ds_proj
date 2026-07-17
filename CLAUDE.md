# Development instructions

Read `README.md` and `info/00_overview.md` before changing runtime behavior.

## Architectural invariants

1. `UserAction` is authoritative and is created by `UserSimulator`.
2. The floor selects one intact bid and never rewrites its act, focus, reason, target, movement, or vote.
3. The LLM realizes one selected action as wording only.
4. Public protocol state changes only after accepted visible text. Questions, newly opened concerns, movement, and votes must be visibly realized.
5. There is no validator LLM. Deterministic validation is limited to concrete action invariants, grounding risks, direct-answer relevance, vote clarity, required movement visibility, hard-blocker contradictions, and near-duplicate wording.
6. Do not reintroduce expected-turn-share correction, urgency formulas, controller-selected concessions, candidate pressure scores, or unrestricted semantic parsing.
7. Engagement controls voluntary bidding; verbosity controls word budget; directness controls wording; stubbornness controls movement probability. Hard blockers never move.
8. The environment owns only opening order, direct-answer obligations, categorical floor selection, one active issue, broad pacing, narrowing, vote collection, and outcomes.
9. Direct questions name one addressee and create one required answer. Other issue responses remain voluntary and bounded.
10. A valid majority closes immediately. A second vote is allowed only after visible movement in one bounded re-narrowing round.
11. Voluntary movement that fails generation and one repair is dropped and logged. Only formal votes and mandatory movement statements may use concise deterministic fallbacks.
12. Grounding must be option-specific. A value belonging to another option cannot validate the focused option. Subjective personal judgments remain allowed.
13. Setup sampling must use the run-local RNG. Evaluation comparisons must reuse the same generated scenario/personas for every candidate in a replicate.
14. All behavior probabilities and language limits belong in `config.yaml` and must pass `Config._validate()`.
15. Keep the project proportional to an eight-page implementation report. Do not add coalitions, emotions, deception, status hierarchies, long-term memory, or research-scale evaluation infrastructure.

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

One bounded `VOTING → NARROWING → VOTING` return is permitted when no majority exists and re-narrowing produces visible movement.

## Testing

```powershell
py -m pytest -q
```

Tests should assert public behavior, ownership boundaries, and concrete correctness properties rather than exact LLM prose.

## Evaluation

Active scripts live in `eval2/`; `eval/` preserves historical outputs only. `src/eval.py` exposes the runtime metric schema consumed by post-hoc evaluation.

- `run_eval_suite.py`: focused pinned cases;
- `run_scenarios.py`: broader scenario batch;
- `evaluate_runs.py`: deterministic analysis;
- `judge_transcripts.py`: three-role LLM judge;
- `validate_judge.py`: corruption validation;
- `run_config_sweep.py`: four observed configuration areas;
- `run_config_confirmation.py`: matched multi-topic confirmation.

Experiments override configuration in memory and resolve paths relative to `eval2/`. Paired configuration runs reuse the same generated setup and record a setup fingerprint.

The `voluntary` metric means self-selected floor entry. Openings, required answers, required narrowing turns, votes, and liveness-forced turns are not used to evaluate engagement.
