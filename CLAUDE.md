# CLAUDE.md

This file gives working instructions for Claude Code / Codex-style coding agents in this repository.

## Role and project framing

Act as a senior Python developer and AI/dialogue-simulation engineer. This is a university project for an **option-grounded multi-user decision simulator**.

Do not treat this as a generic chatbot or a generic multi-agent demo. The system simulates 2-7 configurable users discussing a fixed option board. The option board and shared context are the factual source of truth. The goal is to produce analyzable group-decision traces in which configurable participant parameters visibly affect behavior.

The outcome must be based on visible text only:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement after bounded narrowing.

## Before editing

Read these files first:

1. `docs/todo.md` — authoritative current open issues. It is not a changelog.
2. `README.md` — current project framing and run instructions.
3. `info/00_overview.md` through `info/08_configuration_and_running.md` — workflow notes.
4. `config.yaml` — active behavior settings.
5. Latest `logs_eval_suite/` output if available — especially `eval_suite_runs.csv`, `run.json`, and transcripts from split-vote/no-moderator cases.

Do not assume old issues are still open if `docs/todo.md` says otherwise. Do not claim something is fixed unless code structure or fresh logs support it.

## Current highest priorities

The latest code round changed the controller, prompts, validation, and eval suite, but live LLM validation is still required. Current priorities are:

1. Verify deterministic split-vote ranking: a strict visible plurality should be tested before weaker one-vote candidates; tied leaders should be ranked by blockers/resistance/compromise fit.
2. Verify post-reservation decision beats: each holdout should visibly switch, stay, or name an alternative before closure.
3. Run the forced `f01_manual_manual_n2_stubborn_deadlock` case and inspect whether `two_person_deadlock_attempted = true` and the transcript shows the blocker/concession exchange.
4. Check that candidate-specific reservations no longer import tradeoffs from unrelated options.
5. Compare token diagnostics: compact utterance prompts, deterministic peer split summaries, and deterministic grounding tripwires should reduce `tokens_utterance_in` and `tokens_grounding_in`.
6. Ensure unsupported logistical workaround claims are repaired or phrased as uncertainty.
7. Monitor trait-weighted participation for regressions, but do not overfit it before narrowing/cost are validated.

Work on these one at a time.

## Implementation principles

- Prefer deterministic controller logic over adding more LLM calls.
- Keep prompts smaller and more act-specific; do not solve every issue with longer prompts.
- Do not turn the simulator into a rigid agenda checklist.
- Keep the option-grounded decision scope.
- Sims may propose uncertain mitigations, but must not state invented concrete facts as known.
- Low-engagement sims should be quieter, not invisible.
- High-engagement sims should be more active, not accidentally dominant.
- Same-speaker continuations are allowed as a design choice, including rare chains up to three messages, but only if they are genuine addendums, corrections, clarifications, afterthoughts, or self-resolutions. Prevent duplicate consecutive turns.
- Mid-discussion stance movement should be possible before final voting.
- Unresolved outcomes are allowed, but they should feel earned after real narrowing attempts.

## Development workflow

1. Update `docs/todo.md` first if the open issue list is stale.
2. Pick one issue.
3. Inspect the relevant code and logs.
4. Make the smallest coherent code change.
5. Run static checks:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
```

On shells that do not expand `src\*.py`, run the equivalent Python compile command manually.

6. Run targeted eval cases if LLM access is available. Use `py run_eval_suite.py --quick` for a quick pass and `py run_eval_suite.py --full` before claiming behavioral completion.
7. Inspect transcripts manually, not only metrics.
8. Update the relevant `info/*.md`, `README.md`, and this file only when behavior or workflow changed.
9. Do not remove an issue from `docs/todo.md` unless logs or deterministic code prove it is fixed.

## What to inspect in logs

For split/narrowing behavior, inspect:

- `q01_manual_manual_three_way_split`
- `q03_manual_manual_trait_spread`
- `q05_auto_env_manual_participants`
- `q06_auto_auto_baseline_n3`
- no-/light-moderator cases
- future forced `n=2` stubborn-deadlock case

Metrics to watch:

- `outcome_status`
- `visible_votes`
- `discussion_lean_shifts`
- `split_reservation_exchanges`
- `two_person_deadlock_attempted`
- `participant_procedural_moves`
- `unsupported_printed_turns`
- `engagement_behavior_correlation`
- token usage by call type

## Non-goals for the next round

Do not prioritize:

- adding more personality traits,
- integrating more papers,
- broad open-domain chat,
- cosmetic transcript polish,
- large architectural rewrites unrelated to the open issues,
- more LLM calls for negotiation.

The next round should make disagreement handling smarter and reduce cost.
