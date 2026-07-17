# Evaluation and logging

## Runtime outputs

Each run writes:

- `transcript.md`: option board, visible dialogue, outcome, and compact metrics;
- `run.json`: scenario, complete personas, public/private runtime state, structured actions when enabled, issues, votes, failed generation records, and metrics;
- `metrics.csv`: one flat row for comparison;
- prompt logs when `output.write_prompts` is enabled.

`src/eval.py` defines the runtime metric schema. `eval2/evaluation_metrics.py` performs deterministic post-hoc analysis over completed `run.json` files.

## Deterministic tests

```powershell
py -m pytest -q
```

The offline suite checks setup validation, seeded sampling, simulator authority, floor arbitration, questions and concerns, stance movement, hard blockers, grounding, repair accounting, voting, logging, and evaluation script definitions.

## Active evaluation folder

All active scripts live in `eval2/`; paths are resolved relative to that folder and scripts work from either the repository root or `eval2/`. Experiments modify configuration in memory only and create required output folders automatically.

### Focused case suite

```powershell
py .\eval2\run_eval_suite.py
```

Runs 17 pinned cases over ten public scenarios and group sizes 2–7. It checks closure, openings, direct-answer ordering, valid votes, hard blockers, bounded re-voting, movement accounting, issues, repairs, and pacing. Two cases are explicit long-dialogue diagnostics rather than normal defaults.

### Scenario batch

```powershell
py .\eval2\run_scenarios.py --limit 40
```

Runs `eval2/scenarios.txt` (`participant_count | topic`) and preserves partial progress when interrupted. Setup or protocol failures become explicit error rows rather than aborting the batch.

### Deterministic post-hoc evaluation

```powershell
py .\eval2\evaluate_runs.py
```

Produces compact evidence for protocol completion, questions, votes, outcomes, hard blockers, movement, repairs, dropped turns, fallbacks, grounding flags, repetition, issues, option coverage, token use, and aggregate engagement/verbosity/stubbornness realization.

These metrics are diagnostics, not claims of complete semantic understanding. For example, unsupported-fact counts report validator signals and may miss qualitative hallucinations.

### LLM transcript judge

```powershell
py .\eval2\judge_transcripts.py
```

The judge sees the complete scenario, options, persona cards, moderator messages, participant messages, votes, and outcome. Three deterministically rotated specialist roles score:

- naturalness;
- coherence;
- groundedness;
- persona consistency;
- deliberation quality.

Scores use anchored integers from 1 to 5. Malformed structured responses are retried. Individual judge rows and aggregate run rows are stored separately. Consensus and successful outcomes are not rewarded automatically.

### Judge validation

```powershell
py .\eval2\validate_judge.py
```

Creates controlled turn-order, grounding, persona, and outcome corruptions. The target judge dimension should decrease for each corrupted version. This is a lightweight consistency check, not a replacement for human evaluation.

### Configuration sweep and confirmation

```powershell
py .\eval2\run_config_sweep.py
py .\eval2\run_config_confirmation.py
```

The sweep is limited to four observed problem areas:

- duplicate detection;
- issue follow-up depth;
- consecutive turns;
- small-group closure.

For each seed, all candidates reuse the exact same generated scenario and personas and record a setup fingerprint. The confirmation script uses the selected cumulative settings on several matched topics, again reusing one setup per topic across profiles. Setup tokens are therefore not repeatedly charged to every compared runtime.

## Historical results

`eval/` preserves the earlier scenario logs, judge outputs, and intentionally interrupted batch. It contains no active evaluation scripts. Current tools can still analyze those logs by passing an explicit `--logs` path.

## Turn terminology

A self-selected or `voluntary` turn means that the simulator chose to enter the ordinary floor. Openings, required answers, required narrowing positions, votes, and liveness-forced contributions are excluded from engagement evaluation.
