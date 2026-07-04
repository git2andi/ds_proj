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

The latest evaluation showed that the simulator improved, but these issues remain open:

1. Split-vote candidate selection is sometimes poor. If one option leads, it should usually be tested first. If all are tied, choose a candidate by compromise potential, not arbitrary order.
2. After a reservation response, holdouts need an explicit switch/stay/alternative step.
3. The `n=2` deadlock protocol needs a forced stubborn 1-1 validation case and may need adjustment.
4. Compromise prompts and reservations sometimes mix attributes from the wrong option.
5. Token cost is still too high. Utterance calls and grounding calls dominate.
6. Grounding still leaks occasional unsupported logistical facts or mitigation claims.
7. Trait-weighted participation is improved but must not regress.

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
