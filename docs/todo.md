# TODO

This file is the active work queue. It should contain only open work.

The project remains an **option-grounded multi-user decision simulator** with exactly three outcomes:

- `successful`
- `majority`
- `unresolved`

## Required protocol

1. Work on one issue at a time.
2. Move all existing logs into logs/archive.
3. Prefer replacing old logic over adding parallel logic.
4. Run static checks before packaging:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

5. Pick the next todo from the list.
6. Once you're sure all todos are done and everything works, run `eval\run_eval_suite.py` locally.
7. Update `CLAUDE.md`, `README.md`, and `info/*.md` to reflect the current state of the code, then remove completed todos from this file.

## Open todos

None. The v10 stabilization/simplification round (simulator-parameter model, OCEAN derivation, speech_style rename, age bands, engagement-only turn share, verbosity-only word budgets, deterministic question obligations, dialogue-state move logic, stubbornness merge, prompt simplification, eval fixtures/labels, docs) was completed on 2026-07-09 and verified with a full eval suite pass (12/12 cases, rc=0, no unsupported printed turns, no blocker violations).
