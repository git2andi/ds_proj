# TODO – Discourse Naturalness, Trait Visibility, and Defensible Outcomes

This file contains the current open implementation issues only. It is intended as the active handoff for the next coding session.

The project is an option-grounded multi-user decision simulator, not a generic chatbot and not a full human social simulator. The goal of these fixes is to improve naturalness, trait visibility, and outcome defensibility without adding broad new architecture, large prompt blocks, or unnecessary LLM calls.

Use `gpt` for dialogue generation unless a task explicitly compares providers.

---

## Required implementation workflow

Work on exactly one issue at a time unless a dependency is explicitly stated in the issue.

For each issue:

1. Move all current logs into `logs/archive/` so the active `logs/` folder is clean for this issue.
2. Read the issue carefully and understand the failure mode it targets.
3. Inspect the relevant code and documentation before changing anything.
4. Prefer deterministic controller/state/routing/template fixes over broad prompt expansion.
5. Keep token usage under control. Do not add large context blocks or extra LLM calls unless clearly necessary.
6. Implement the smallest fix that addresses the issue.
7. Run static validation, at minimum:
   - `py -m py_compile main.py run_eval_suite.py src/*.py`
8. Run targeted examples only:
   - Always use a different random topic.
   - one `n=3` run is mandatory;
   - then run 2–3 additional examples with varying group sizes from `n=2` to `n=7`.
   - Do not run the full suite after every issue.
9. Manually inspect the new logs. Verify the issue is solved across varying group sizes.
10. If the fix is not actually visible in the logs, revise the implementation and re-run targeted examples.
11. If a new serious issue appears while testing this issue, fix it directly before moving on.
12. Once the issue is solved, update the documentation:
    - `CLAUDE.md`
    - `README.md`
    - relevant `info/*.md` files
    - this `docs/todo.md`
13. Remove the solved issue from this file only after the targeted logs prove the fix works.
14. Before starting the next issue, again move current logs into `logs/archive/`.

Only after all issues in this file are resolved:

1. Move all remaining logs into `logs/archive/`.
2. Run the full evaluation suite once:
   - `py run_eval_suite.py --full`
3. Inspect every generated run.
4. Confirm that traits are visible, utterances are clean, voting is defensible, outcomes are correct, agenda is not dominating the dialogue, and no major hallucination/option-drift has appeared.
5. Update `CLAUDE.md`, `README.md`, `docs/todo.md`, and relevant `info/*.md` files to reflect the final current state.
6. Commit and push to git.
---

## Open issues

None. The 2026-07-07 discourse round (P1 unresolved closure, P2 trait-colored delivery, P3 vote language, P4 micro-reactions, P5 narrowing speech, P6 agenda priority, P7 personal anchors, P8 derived friendliness, P9 flexible pacing) is complete and validated with the full 12-case suite. See `CLAUDE.md` and `info/*.md` for what landed.

The two previous monitoring items were fixed in this round: M1-class cross-option mixups (concern misattribution to the speaker's own pick, self-blocker misparse, and dB/mpg/kHz-style invented measurements) and M2 (a split-summary caller answering their own holdout question).

## Monitoring notes (not blocking)

- M3: rare chopped fragments still slip through `utils.clean_generated` when a
  clause is cut at a noun phrase ("…but Remote Work's team building."). The
  salvage now handles dangling prepositions/dashes/subclauses; watch whether
  the remaining shapes justify another pass.
- M4: the LLM grounding judge (gpt-4.1-mini) repeatedly accepts invented
  unit-bearing quantities that the deterministic unit-class net now catches;
  if new unit families appear in transcripts, extend `validation._UNIT_NUMBER`
  rather than the judge prompt.

## General non-goals

Do not implement a full human-chat simulator.

Do not add rich long-term memory, full Generative Agents-style reflection, or detailed personal histories.

Do not make agenda the primary dialogue engine.

Do not solve repetition by simply adding large prompt instructions.

Do not add many new personality axes. Friendliness is acceptable because it directly improves surface tone and can be derived from existing traits.

Do not let personal anchors or micro-chatter override option-grounded decision making.

The intended target is not raw WhatsApp realism. The target is plausible controlled decision dialogue with visible configurable participant behavior.
