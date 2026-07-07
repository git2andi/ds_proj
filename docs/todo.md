# TODO: v3 stabilization and simplification

This file is the active work queue. It should contain only open work.

The project remains an **option-grounded multi-user decision simulator** with exactly three outcomes:

- `successful`
- `majority`
- `unresolved`

## Current baseline

The current v3 version has a central per-sim/per-option stance table:

```text
4 preferred, 3 acceptable, 2 neutral, 1 disliked, 0 rejected/hard blocked
```

Runtime stance now uses this rank table directly. There are no separate runtime preference/rejection containers. The compact controller act vocabulary is:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

## Required protocol

1. Work on one issue at a time.
2. Prefer replacing old logic over adding parallel logic.
3. Run static checks before packaging:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

4. Run quick/full eval locally when provider access is available.
5. Inspect transcripts manually.

## Open checks after this refactor

- Run `py .\eval\run_eval_suite.py --quick` and inspect the transcripts.
- Verify that repair/fallback counts did not increase.
- Verify that successful, majority, and rare unresolved outcomes still appear naturally.
- Verify that opening turns are chat-like but not repetitive.
- Verify that rank movements explain final switches and do not force consensus.
- If repair cost is still high, next simplification should target decision/vote prompting and parser compatibility, not new features.
