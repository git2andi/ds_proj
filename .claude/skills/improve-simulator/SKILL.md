# Skill: improve-simulator

Iterative improvement workflow for the group-discussion simulator.

## When to use

When the user asks to improve, fix, or evaluate the simulator's output quality —
naturalness, convergence, state integrity, moderator behaviour, or validation.

## Workflow

### 1. Identify the problem

- Read the most recent transcripts (`logs/<newest>/transcript.md`) and `run.json`.
- Check `docs/known_failures.md` — is this already tracked?
- Run `evals/run_eval.py --check-latest 4` to see if automated checks flag it.

### 2. Write a failing test (if deterministic)

If the fix belongs in `validation.py` or `parsing.py`, add a test in
`tests/test_validation.py` or `tests/test_parsing.py` first. Run:

```powershell
& .\dspro\Scripts\python.exe -m pytest tests/ -v
```

The new test should fail, confirming the bug exists.

### 3. Implement the fix

- **`config.yaml`** for tunable numbers.
- **`src/prompts.py`** for LLM prose and moderator text.
- **`src/validation.py`** for deterministic guardrails.
- **`src/parsing.py`** for trailer/option parsing.
- **`src/router.py`** for turn-taking logic.
- **`src/dialogue.py`** for orchestration and state tracking.

Design constraints (never violate):
- Topic-agnostic: fixes must work for any topic, any group size 2–7.
- No fabricated fallbacks: if a call fails, raise — don't paper over it.
- No offline/mock mode: all runs require a live provider. Error on connection failure.

### 4. Verify offline

```powershell
& .\dspro\Scripts\python.exe -m pytest tests/ -v
```

All tests green before any live run.

### 5. Verify live

Run at least one small and one large group:

```powershell
"Test topic" | & .\dspro\Scripts\python.exe .\main.py
```

Or use the eval runner:

```powershell
& .\dspro\Scripts\python.exe evals\run_eval.py --run --topic "relevant topic" --size 3
```

Read the transcript. Check `run_eval.py` output for FAIL/WARN lines.

### 6. Update tracking

- Update `docs/known_failures.md` (move to Fixed, add regression signal).
- If a new failure pattern is found, add it to Open section.
- If the fix changes evaluation thresholds, update `evals/run_eval.py`.

## Files involved

| File | Role |
|---|---|
| `tests/test_validation.py` | Unit tests for validation guardrails |
| `tests/test_parsing.py` | Unit tests for trailer parsing and commitment gating |
| `evals/run_eval.py` | Post-run regression checker |
| `evals/scenarios.yaml` | Topic/size spread for batch evaluation |
| `docs/known_failures.md` | Tracked failures and their fix status |
| `docs/evaluation.md` | Full evaluation workflow reference |

## Important

- Never change generation behaviour without a live run to verify.
- The `uni` provider requires the Bamberg VPN. No fallback.
- `max_repairs_per_turn: 1` — validation catches issues but only gets one repair shot.
- Prompt changes in `src/prompts.py` affect every turn — test broadly, not just the target case.
