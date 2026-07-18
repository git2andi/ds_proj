# Configuration and running

`config.yaml` controls only the implemented compact runtime.

Main sections:

- `llm`: provider, model, endpoint, timeout, and setup/dialogue/repair sampling profiles;
- `environment`: automatic or manual scenario;
- `simulation`: participant count, bounds, seed, and scenario generation attempts;
- `participants`: automatic or manual profiles;
- `scenario`: labels, attribute bounds, context bounds, and maximum alias length;
- `personas`: trait ranges, hard-blocker probability, and preference shapes;
- `conversation`: voluntary-turn budgets, thread cap, stagnation threshold, compromise cap, prompt context, and consecutive-turn bound;
- `simulator`: willingness and movement probabilities;
- `language`: word budgets and directness instructions;
- `moderator`, `consensus`, `limits`, and `output`.

Scenario and persona generation each use the configured three-attempt limit. The separate alias-and-name metadata call uses the setup sampling profile. It does not consume an additional scenario-generation attempt and cannot invalidate a structurally valid board. Dialogue sampling applies to realized turns. The repair profile is used only for a failed opening; required answers are not repaired or replaced. Formal votes are deterministic.

Install and run:

```powershell
py -m pip install -r requirements.txt
py .\main.py
```

Deterministic tests:

```powershell
py -m pytest -q
```

Clean scenario batch:

```powershell
py .\eval\run_scenarios.py --limit 10 --seed 500 --clean
# Uses two isolated worker processes by default; pass --workers 1 to run sequentially.
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

The batch runner refuses to mix new results into a nonempty output directory unless `--clean` is explicitly supplied. It uses process-based concurrency because each dialogue temporarily overrides the module-level configuration. The parent process serializes updates to `scenario_runs.csv` and `scenario_summary.md`; individual run directories are written independently by the workers.
