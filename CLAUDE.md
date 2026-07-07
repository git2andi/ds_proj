# CLAUDE.md

Working instructions for coding agents in this repository.

## Project framing

This is a university project for an **option-grounded multi-user decision simulator**. The system simulates 2-7 LLM-driven participants discussing a fixed set of options. The goal is not arbitrary open-ended chat. The goal is a configurable simulator whose participant parameters visibly affect turn-taking, stance movement, disagreement, compromise, and final outcomes.

The architecture should stay explainable:

```text
state -> route speaker/act/target/focus -> generate one utterance -> parse visible state -> continue/narrow/vote/close
```

The LLM should render utterances. The controller should own phase logic, routing, narrowing, and final outcome rules.

## v3 baseline

v3 is intentionally a middle-ground version:

- Base: v1, because it is easier to explain.
- Ported from v2: outcome repair, split-vote/holdout switch-stay logic, unresolved acknowledgement, self-answer avoidance, small trait-routing improvements, parser-safe trait-shaped vote phrases, observer false-blocker fixes. Current v3 additionally uses required-vote validation for decision turns, parser-aligned LLM-rendered post-reservation lines, minimal last-resort fallback protection, and moderator-owned vote calls. Peer self-closure was removed to keep the process simpler.
- Not ported from v2: micro-reaction subsystem, friendliness parameter, personal anchors, larger trait-color prompt subsystem, and broad dynamic-pacing additions.

Do not re-add those omitted features unless there is clear log evidence that the simpler v3 behavior is insufficient.

## Implementation principles

- Prefer deterministic controller/state logic over new LLM calls, except where the controller has already selected the target and validation can block LLM drift.
- Keep prompts act-specific and compact.
- Do not add more personality traits unless required.
- Do not turn the project into a full agenda simulator.
- Do not treat balanced speaking as automatically good; dominance is valid when trait-supported and non-repetitive.
- Direct questions create bounded response obligations.
- After an answer, prefer developing the same thread before opening a fresh issue.
- Same-speaker continuations are allowed only when they add content.
- Direct names should do interactional work and should be rare in n=2.
- Normal auto-generated sims should have movable preferences, not categorical blockers.
- Manual hard constraints must remain binding even for agreeable sims.
- Unresolved outcomes are allowed, but must be earned after narrowing attempts.

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
- Does routing follow traits without becoming a monologue?
- Do quiet sims still show visible stances?
- Do stance switches have visible reasons?
- Do final votes respect the sim's visible stance rather than an old latent favorite?
- Do blockers prevent false unanimity?
- Do no-majority votes trigger bounded narrowing rather than abrupt closure?
- Are unresolved endings socially acknowledged?
- Are token counts, grounding calls, repair counts, and fallback counts still reasonable? Decision turns should normally parse without repair.

## Non-goals for the next round

Do not prioritize:

- full Generative Agents-style memory/reflection;
- new traits such as friendliness;
- micro-reactions as a separate subsystem;
- personal biographical anchors;
- open-domain roleplay;
- provider comparisons before the v3 `gpt` baseline is validated;
- large architectural rewrites before file-by-file simplification.

The next useful step is file-by-file simplification: reduce `dialogue.py` and `policy.py` complexity while preserving v3 behavior.
