# CLAUDE.md

This file gives working instructions for Claude Code / Codex-style coding agents in this repository.

## Role and project framing

Act as a senior Python developer and AI/dialogue-simulation engineer. This is a university project for an **option-grounded multi-user decision simulator**.

Do not treat this as a generic chatbot or a generic multi-agent demo. The system simulates 2-7 configurable users discussing a fixed option board. The option board and shared context are the factual source of truth. The goal is to produce analyzable group-decision traces in which configurable participant parameters visibly affect behavior.

The outcome must be based on visible text only:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement after bounded narrowing.

Do not add a fourth outcome label. If a participant cannot accept an option because of a blocker or unresolved decisive concern, make the transcript show that refusal; the existing outcome logic should then produce `majority` or `unresolved`.

Use `gpt` as the dialogue provider for the next quality-improvement baseline unless the task is explicitly provider comparison.

## Before editing

Read these files first:

1. `docs/todo.md` — authoritative current open issues. It is not a changelog.
2. `README.md` — current project framing and run instructions.
3. `info/00_overview.md` through `info/09_topic_examples.md` — workflow notes.
4. `config.yaml` — active behavior settings. Confirm `llm.provider: "gpt"` for dialogue quality runs.
5. Latest `logs_eval_suite/` output if available — especially `eval_suite_runs.csv`, `run.json`, and transcripts.

Do not assume old issues are still open if `docs/todo.md` says otherwise. Do not claim something is fixed unless deterministic code or fresh logs support it.

## Current highest priorities

The current quality target is not more architecture. The project already has option grounding, controller routing, visible-state observation, repair, evaluation, trait-scaled word budgets, thread-aware routing, group-size-aware addressing, earned stance movement, an issue ledger for repeated unknowns, and free-discussion dominance metrics (behavioral round completed 2026-07-06, validated with the full 12-case suite).

`docs/todo.md` is the authoritative open list. Currently open:

1. Simplify accumulated controller/validation paths, one area at a time, protected by `--full` suite runs before and after (O1).
2. Grounding-judge false positives on short fully-grounded lines pollute the unsupported metrics (O2).
3. Keep monitoring split runs for unbounded option blends; escalate to a deterministic observer check only if they reappear (O3).
4. `switch_explanation_rate` under-counts em-dash reason clauses; fix the detector, not the phrasing menus (O4).

## Implementation principles

- Prefer deterministic controller/state logic over adding more LLM calls.
- Keep prompts smaller and more act-specific. Do not add broad social instructions as the first fix.
- Do not turn the simulator into a rigid agenda checklist.
- Keep the option-grounded decision scope.
- Sims may propose uncertain mitigations, but must not state invented concrete facts as known.
- Speaking should not be balanced by default. Dominant/high-engagement/high-initiative sims may speak more. Quiet sims should not disappear.
- Direct questions create response obligations. The addressed sim should usually answer soon.
- Avoid question churn: after a question is answered, prefer build/agree/challenge/compare on the same issue before opening a new issue.
- Same-speaker continuations are allowed when they are genuine addendums, corrections, clarifications, afterthoughts, or self-resolutions. Prevent duplicate consecutive turns and repeated questions.
- Direct addressing is useful, but leading names should be less frequent, especially in n=2 runs.
- Verbosity is an average behavior, not a per-turn template. Every sim may have short and longer turns.
- Hard blockers should not sabotage the chat. They should resist only according to configured traits/constraints and still participate in discussion.
- Unresolved outcomes are allowed, but they should feel earned after real narrowing attempts.

## Development workflow

1. Update `docs/todo.md` first if the open issue list is stale.
2. Pick one issue only.
3. Inspect the relevant code and latest logs.
4. Make the smallest coherent change.
5. Run static checks:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
```

On shells that do not expand `src\*.py`, run the equivalent Python compile command manually.

6. Run targeted eval cases if LLM access is available. Use `py run_eval_suite.py --quick` for a quick pass and `py run_eval_suite.py --full` before claiming behavioral completion.
7. Inspect transcripts manually, not only metrics.
8. Update the relevant `info/*.md`, `README.md`, and this file only when behavior or workflow changed.
9. Remove or narrow an item in `docs/todo.md` only after logs/code prove it is fixed.

## What to inspect in logs

Always inspect both text and structured state:

- `transcript.md`: Does the conversation feel like a real option-grounded group decision?
- `run.json`: Do visible commitments, switches, blockers, obligations, and option references match the transcript?
- `eval_suite_runs.csv`: Do outcomes and high-level metrics agree with manual inspection?

Priority checks:

- average words per participant and by act;
- whether short turns exist;
- question rate versus answer adjacency;
- direct-name/name-prefix frequency, especially in n=2;
- free-discussion turn share versus trait-derived expected share;
- whether same-speaker continuations add new content;
- whether stance switches have visible triggers;
- whether repeated unknowns such as parking/reservations loop;
- whether explicit blockers prevent false unanimity;
- repair/grounding token cost and unsupported printed turns.

## Non-goals for the next round

Do not prioritize:

- adding more personality traits,
- integrating more papers directly,
- full Generative Agents-style memory/reflection,
- a full agenda simulator,
- broad open-domain chat,
- cosmetic transcript polish before behavioral fixes,
- large architectural rewrites unrelated to the open issues,
- more LLM calls for negotiation.

The next round should make the simulator shorter, more causally coherent, more trait-shaped, and cheaper to run.
