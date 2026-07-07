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

The current quality target is not more architecture. The project already has option grounding, controller routing, visible-state observation, repair, evaluation, trait-scaled word budgets, thread-aware routing, group-size-aware addressing, earned stance movement, an issue ledger for repeated unknowns, free-discussion dominance metrics, trait-colored delivery and vote language, deterministic micro-reactions and unresolved social closure, personal anchors, derived friendliness, and conflict-adaptive pacing (behavioral + naturalness rounds completed 2026-07-06, discourse round completed 2026-07-07).

`docs/todo.md` is the authoritative open list. The 2026-07-07 discourse round (P1-P9: naturalness, trait visibility, defensible outcomes) is COMPLETE. What landed:

- P1 unresolved closure: after an `unresolved` finalization, one participant emits a deterministic acknowledgement beat naming the contested options in natural wording (two-way, three-way-tie, single-camp, and holdout-aware n=2 forms; appended without semantics so it can never parse as a vote), followed by the moderator closing or a varied peer wrap-up. A narrowing beat also never moves a sim to a camp with fewer visible votes than its own, so ultra-flexible sims cannot ping-pong between tested candidates and break a forming majority.
- P2 trait-colored delivery: `challenge` weight scales with directness (not only stubbornness), `propose_compromise` with compromise tendency, `soften` inversely with stubbornness. `MoveIntent.trait_color` carries a compact label rendered as one prompt line — `challenge_directly`, `soften_and_bridge`, `bridge_condition`, `restate_concern` (once per run, a high-stubbornness sim's ordinary turn becomes a fresh-words restate of its core concern against a recorded objection target or the largest rival camp). Concern attribution fix: a challenge registers against the rival, never the speaker's own pick just because its name appears first; a parsed hard rejection of the speaker's own current favorite is dropped as a misparse (old M1 class).
- P3 vote language: the parser gained "I'm choosing / I'm still on / I'll stay with / I'll back / I'd be happy with" (with lookahead guards — "I'll back down", "still on the fence" never parse); vote prompts order the unused-phrasings menu by trait fit (stubborn stayers "I'm still on", direct "I vote for", compromising/agreeable switchers "I can live with" / "I'd be happy with"); the deterministic post-reservation stay/switch beats use trait-flavored variant pools. Targeted runs: six distinct phrasings in one n=6 round, zero UNCLEAR_VISIBLE_COMMITMENT repairs.
- P4 micro-reactions: a deterministic, probability-gated tiny reaction beat (cap max(2, n//2+1) per run) after answers/challenges/agreement — "Fair.", "Same here.", "Not convinced.", "That's my worry too." — with polarity from visible state (own pick attacked -> resist; supported -> agree). Option-free texts, zero LLM cost, never after a question or over a pending obligation. tiny_turn_rate 0.07-0.19 in targeted runs (baseline 0.053).
- P5 narrowing speech: participant split summaries draw from trait-colored variant pools (direct / compromising / high-initiative callers), same functional structure, no controller vocabulary; the split-summary caller is never the first holdout to answer their own question (old M2 fixed). Salvage improvements: unspaced em-dash clause boundaries, longer trailing-subclause stubs stripped, hanging prepositions ("cut into.") and interrupted-style trailing dashes removed.
- P6 agenda priority: the agenda fires only when the local thread is cold (no question on the floor, no unreacted answer, no unaddressed concern) at 0.15 + 0.25*initiative; observed agenda-driven turns ~10-15% of participant turns.
- P7 personal anchors: each sim carries 1-2 compact trait-derived anchors (`simulator.derive_personal_anchors`, deterministic pool — never invented scenario facts; manual profiles may override via `anchors:`); the controller offers a sim's anchor to at most one prompt per run (opening 0.35, resistance/softening/compromise 0.22).
- P8 derived friendliness: `SimulatorParameters.friendliness` = 0.20 + 0.50*agree + 0.25*extra − 0.20*neuro*(1−agree), overridable per manual profile; rendered as voice guidance (warm >=0.72, dry-toned <=0.30, never hostile). Warm vs dry sims are detectably different; low friendliness stays cooperative.
- P9 flexible length: all-same-start casts get a lower minimum (−1.5 turns/participant, floor 3.0); cast drive (engagement + 0.5*initiative) shifts force/hard by ±0.5n; at the forced-narrowing point a still-multi-camp run with an unaddressed concern keeps discussing up to the hard cap. Observed: n=2 19, n=3 26/31, n=4 36, n=5 45 turns with structurally different runs.
- Grounding net widened during testing: dB/mpg/kHz/bit-style units added to the deterministic invented-measurement check (gpt-4.1-mini's judge repeatedly passed "45 dB"/"19 mpg"/"48kHz" claims).

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
