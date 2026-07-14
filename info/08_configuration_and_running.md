# Configuration and running

The main runtime controls in `config.yaml` are:

```text
llm.dialogue and provider/model sampling
simulation participant count, seed, setup attempts, one-repair bound
personas direct trait ranges and hard-blocker probability
conversation group pacing, issue caps, recent prompt context, floor limits
moderator.enabled
consensus majority fraction
output paths
```

Removed controls include OCEAN, switch resistance, expected turn shares, validator endpoints, granular moderator switches, thread priorities, repair families, fallback families, and unanimity-repair tuning.

Manual personas assign traits directly:

```yaml
participants:
  mode: manual
  profiles:
    - name: Nora
      description: Works on a practical project.
      private_goal: Needs reliable equipment.
      preferred_option: C
      age: 29
      speech_style: relaxed practical wording
      traits:
        engagement: 4
        verbosity: 3
        directness: 4
        stubbornness: 2
```

A manual hard blocker additionally sets `hard_blocker: true` and a `rejection_reason`. At most one may appear.

Run a generated scenario:

```powershell
py .\main.py "Choose a study location"
```

Run the configured manual environment by invoking `main.py` without an explicit topic. Run deterministic verification first, then the live LLM-backed evaluation suite:

```powershell
$env:PYTHONPATH = "src"
py -m pytest -q
py -m compileall -q main.py src eval tests
py .\eval\run_eval_suite.py
```
