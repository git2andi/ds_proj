# Project engineering guide

This repository has one active dialogue runtime. Do not reintroduce the deleted controller/parser/observer architecture or add compatibility wrappers for it.

## Architectural invariant

`UserAction` is authoritative. Each `UserSimulator` independently decides whether to bid, which action to perform, the option focus, addressee, reason, issue effect, stance update, and vote. `FloorManager` may select among intact bids but must not rewrite them. The dialogue LLM only realizes the selected action. Accepted state changes are applied from the action, never reconstructed from text.

## Runtime ownership

- `src/models.py`: public/private runtime state, actions, issues, votes, outcomes.
- `src/simulator.py`: seeded participant-local policy and open-floor arbitration.
- `src/dialogue.py`: phase loop, obligations, issue lifecycle, deterministic moderator, realization and one bounded repair.
- `src/prompts.py`: compact setup and action-realization prompts.
- `src/validation.py`: structured-action checks and minimal hard-failure text checks.
- `src/consensus.py`: public candidate standings and vote outcomes.
- `src/builders.py`: existing scenario and persona setup, direct traits, at most one hard blocker.
- `src/logger.py`: compact transcript, JSON, and CSV metrics.

## Non-negotiable boundaries

Do not add an LLM call for bidding or action selection. Do not add a validator LLM. Do not infer complete state from utterances. Do not add expected participation shares, per-person quotas, multiple active issues, deterministic participant fallback lines, hidden stance changes, majority-to-unanimity repair, or more than one re-vote.

Direct traits are integer scales: engagement, verbosity, and directness use 1–5; normal stubbornness uses 1–4. Stubbornness 5 is reserved for an explicit hard blocker. Engagement affects only voluntary bid probability and urgency. Verbosity controls soft action-scaled word targets; directness changes wording only. Stubbornness controls acceptance and switching, which also require distinct public evidence and switch hysteresis. Age and speech style affect lexical realization only.

Candidate standings and switching evidence must use distinct public participants, never raw repeated support counts. Question uniqueness is keyed by intent, option focus, and addressee. Concern responses must retain issue provenance and explicitly identify whether they mitigate the same issue, accept the trade-off, or agree the concern remains. Required direct answers take precedence over phase transitions.

## Verification

After runtime changes, run deterministic tests first. The evaluation command then uses the configured live dialogue LLM:

```powershell
$env:PYTHONPATH = "src"
py -m pytest -q
py -m compileall -q main.py src eval tests
py .\eval\run_eval_suite.py
```

Inspect representative `transcript.md` and `run.json` files, not only aggregate metrics. Confirm that actions remain autonomous, questions route correctly, stance changes are visible, concrete option facts remain grounded, hard blockers never switch, majority closes directly, and at most one re-vote occurs.
