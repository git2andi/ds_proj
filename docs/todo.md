# TODO: v3 stabilization and simplification

This file is the active work queue. It should contain only open work.

The project remains an **option-grounded multi-user decision simulator**. Keep exactly three outcomes:

- `successful`
- `majority`
- `unresolved`

Do not add a fourth outcome and do not add broad new personality subsystems without evidence from logs.

## Current baseline

v3 is based on v1 plus selected v2 fixes:

- controller-selected, LLM-rendered holdout switch/stay turns after majority and split-vote reservations;
- no downhill compromise;
- bounded tie compromise for flexible sims;
- unresolved acknowledgement before closure;
- split-summary self-answer avoidance;
- active-thread priority over private agenda;
- small trait influence on act choice and vote phrase selection;
- required-vote validation so generated decision lines cannot drift to the wrong option;
- moderator-owned final vote call; participant self-closure was removed to avoid extra routing complexity;
- observer fixes for false hard blockers on a sim's own current favorite.

v3 intentionally excludes v2 micro-reactions, friendliness, personal anchors, and large trait-color wording logic.

## Required protocol

1. Work on one issue at a time.
2. Prefer controller/state simplification over prompt expansion.
3. Run static checks after every code change:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

4. If LLM access is available, run at least:

```powershell
py .\eval\run_eval_suite.py --quick
```

5. Before declaring the version stable, run:

```powershell
py .\eval\run_eval_suite.py --full
```

6. Inspect transcripts manually. Do not rely only on metrics.
7. Update `README.md`, `CLAUDE.md`, and relevant `info/*.md` files after behavior changes.

## Open issues

### O1 — Validate v3 behavior with the eval suite

Run the quick suite first, then the full suite. Compare against the intended v3 behavior:

- fewer abrupt unresolved endings;
- no fake successful outcomes from forced compromise;
- holdouts visibly switch or stay with reasons;
- no participant immediately answers their own split prompt;
- directness/compromise affect behavior without making everyone sound templated;
- final votes do not contradict the sim's own visible objections;
- majority outcomes remain possible when a holdout has a convincing visible reason to stay.

### O2 — File-by-file simplification

Simplify without changing behavior. Priority files:

1. `src/dialogue.py` — split orchestration helpers into clearer sections or smaller pure helpers.
2. `src/policy.py` — keep routing rules explainable; remove duplicate or stale branches.
3. `src/builders.py` — separate manual validation from auto generation if it improves readability.
4. `src/prompts.py` — keep only behavior-relevant prompt rules.

Do not reduce file count at all costs. The goal is clearer responsibility, not artificial minimalism.

### O3 — Remove obsolete code only after proving it is unused

Before deleting a helper, search all references and confirm it is not used by logs, JSON serialization, metrics, or eval cases. Prefer removing dead code over adding abstractions.

### O4 — Outcome plausibility review

After full-suite logs exist, inspect cases where `successful` appears after a split. Confirm the transcript actually earns unanimity. If closure looks forced, tune `_should_switch_after_reservation`, not the prompt.

## Non-goals

- More traits.
- Micro-reaction subsystem.
- Personal anchors.
- Full agenda simulation.
- Full memory/reflection.
- Open-domain group chat.
- Provider comparison before v3 is stable.
