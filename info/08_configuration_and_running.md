# Configuration and running

`config.yaml` contains the provider, setup limits, direct trait ranges, simulator probabilities, conversation bounds, language limits, moderator switch, validation thresholds, and output settings. `src/config_loader.py` validates ranges, monotonic mappings, group-size boundaries, preference distributions, and cross-field constraints at startup.

Important sections:

- `llm`: dialogue provider, models, endpoints, timeouts, and setup/dialogue/repair sampling;
- `environment`: automatic or manual option board;
- `simulation`: participant count, supported range, run seed, and setup attempts;
- `participants`: automatic or manual persona profiles;
- `personas`: direct trait ranges, hard-blocker probability, and preference-shape distribution;
- `conversation`: voluntary pacing, issue bounds, narrowing bounds, prompt history, and consecutive turns;
- `simulator`: engagement bid probabilities, action probabilities, question modes, and stubbornness movement probabilities;
- `language`: verbosity budgets, action caps, directness instructions, and duplicate detection;
- `moderator`: one enabled/disabled switch;
- `validation`: grounding and repetition thresholds;
- `output`: log folder and optional debug/action/prompt output;
- `limits`: input-token warning threshold.

Normal stubbornness is sampled from 1–4. A hard blocker is a separate rare group-level condition and uses stubbornness 5. Manual profiles can specify direct traits and one explicit hard blocker.

## Commands

Run one dialogue:

```powershell
py .\main.py "Your topic"
```

Run deterministic tests:

```powershell
py -m pytest -q
```

Run evaluation:

```powershell
py .\eval2\run_eval_suite.py
py .\eval2\run_scenarios.py --limit 5
py .\eval2\evaluate_runs.py
py .\eval2\judge_transcripts.py
py .\eval2\validate_judge.py
py .\eval2\run_config_sweep.py
py .\eval2\run_config_confirmation.py
```

All evaluation paths are relative to `eval2/`. Configuration experiments apply overrides in memory and restore them after every run. Sweep and confirmation comparisons reuse paired scenario/persona setups.

`conversation.diagnostic_allow_reason_reuse` exists only for deliberately long diagnostic cases and should remain disabled for normal simulations.
