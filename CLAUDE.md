# CLAUDE.md

Working instructions for coding agents in this repository.

## Project framing

This is a university project for an **option-grounded multi-user decision simulator**. The system simulates 2-7 LLM-driven participants discussing a fixed set of options. The goal is a configurable simulator whose participant parameters visibly affect turn-taking, stance movement, disagreement, compromise, and final outcomes.

The architecture should stay explainable:

```text
state -> route speaker/macro-act/target/focus -> generate one utterance -> validate -> parse visible state -> continue/narrow/vote/close
```

The LLM renders utterances. The controller owns phase logic, routing, narrowing, and final outcome rules.

## Current stance model

Private stance is stored as one per-sim/per-option rank table:

```text
4 = preferred
3 = acceptable
2 = neutral / untested
1 = disliked but negotiable
0 = rejected / hard blocked
```

Use `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` as computed helpers over the rank table. Do not add a second independent preference model.

Participant setup may provide short `reason_for` / `reason_against` fields for each option. Keep these short. Leave neutral options neutral. Hard rejects should be rare and grounded.

## Compact act vocabulary

The controller should reason over macro acts:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Legacy act aliases have been removed. Do not grow the act list unless there is strong log evidence that a new act cannot be represented by the macro set.

## Implementation principles

- Prefer deterministic controller/state logic over new LLM calls.
- Keep prompts act-specific and compact.
- Use the option-rank table as the stance source of truth.
- Do not add more personality traits unless required.
- Do not turn the project into a full agenda simulator.
- Normal auto-generated sims should have movable preferences, not categorical blockers.
- Manual hard constraints must remain binding.
- Unresolved outcomes are allowed, but must be earned after narrowing attempts.
- Decision turns should normally parse without repair.
- Invalid lines must not become transcript evidence.

## Development workflow

1. Work on one issue at a time.
2. Inspect the current code and latest logs before editing.
3. Make the smallest coherent change.
4. Run static checks:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

5. Run targeted eval cases if LLM access is available:

```powershell
py .\eval\run_eval_suite.py --quick
```

6. Before claiming stable behavior, run:

```powershell
py .\eval\run_eval_suite.py --full
```

7. Inspect `transcript.md` and `run.json`, not only CSV metrics.
8. Update `README.md`, `docs/todo.md`, and relevant `info/*.md` files when behavior or workflow changes.

## What to inspect in logs

- Does the transcript read like a plausible group decision?
- Are turns grounded in the option board?
- Are direct questions answered by the addressed sim?
- Do stance switches have visible reasons?
- Do final votes respect the sim's rank/visible stance?
- Do rank movements make majority/success/unresolved outcomes plausible?
- Are blockers preventing false unanimity?
- Are repair/fallback/token counts reasonable?
