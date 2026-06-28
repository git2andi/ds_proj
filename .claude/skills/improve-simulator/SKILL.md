# Skill: improve-simulator

Iterative improvement workflow for the group-discussion simulator.

## When to use

When the user asks to improve, fix, or evaluate the simulator's output quality —
naturalness, convergence, state integrity, moderator behaviour, or validation.

## Workflow

### Upgrade boundary

One upgrade is one backlog issue and one independently verifiable task unless the user explicitly groups issues. Finish its tests, required live evidence, transcript review, and information-file synchronization before starting another issue. Stop at that boundary unless automatic continuation was explicitly requested.

### 1. Identify the problem

- Read the most recent transcripts (`logs/<newest>/transcript.md`) and `run.json`.
- Check `docs/known_failures.md` — is this already tracked?
- Separate deterministic controller defects from provider-specific wording.
- For naturalness, assess plain-spoken conversation among friends: no slang-heavy Gen-Z voice, corporate or academic register, mini-essays, standalone pitches, or invisible persona traits.

### 2. Write a failing test (if deterministic)

If the fix belongs in `validation.py` or `parsing.py`, add a test in
`tests/test_validation.py` or `tests/test_parsing.py` first. Run:

```powershell
& .\ds_proj\Scripts\python.exe -m pytest tests/ -v
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
- Preserve visible commitment gating, trait-driven stubbornness, grounded claims, and configured outcome rules.
- Do not tune phrase regexes or prompts for one provider, seed quoted examples, or inject forced turns for naturalness.

### 4. Verify offline

```powershell
& .\ds_proj\Scripts\python.exe -m pytest tests/ -v
```

All tests green before any live run.

### 5. Verify live

One mandatory n=3 run, then 5–6 additional runs across n=2–7 with random topics:

```powershell
"Test topic" | & .\ds_proj\Scripts\python.exe .\main.py
```

Read every relevant transcript and `run.json`; metrics alone are insufficient. Compare visible behavior with the issue's acceptance criteria and check regressions before closing it.

### 6. Update tracking

- Update `docs/known_failures.md` with the evidence and resolution status.
- If a new failure pattern is found, add it to Open section.
- Consolidate overlapping symptoms so the backlog has no duplicate issues.
- Audit and update every applicable active source: `AGENTS.md`, `CLAUDE.md`, both repository skill copies, active memory/index files, `README.md`, and other affected workflow docs. Historical per-fix memories remain historical.

## Files involved

| File | Role |
|---|---|
| `tests/test_validation.py` | Unit tests for validation guardrails |
| `tests/test_parsing.py` | Unit tests for trailer parsing and commitment gating |
| `docs/known_failures.md` | Tracked failures and their fix status |

## Important

- Never change generation behaviour without a live run to verify.
- Use only the provider explicitly authorized for the current task. No silent fallback.
- `max_repairs_per_turn: 1` — validation catches issues but only gets one repair shot.
- Prompt changes in `src/prompts.py` affect every turn — test broadly, not just the target case.
