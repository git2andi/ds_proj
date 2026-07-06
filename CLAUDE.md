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

`docs/todo.md` is the authoritative open list. The 2026-07-06 naturalness round (P1-P11) is COMPLETE (validated with the full 12-case suite; only monitoring items M1/M2 remain in docs/todo.md). What landed:

- P1 participant procedural leakage: peer-owned split summaries now use natural group-member wording (no vote-count dumps, no test/candidate vocabulary, never self-addressing); exact procedural wording stays moderator-only. Alongside it, subject-form commitment parsing was fixed ("X still gets my vote — Y hasn't fixed my concern" used to record a vote for Y and produce false unanimity).
- P2 thread depth: the answer-follow-up rule fires more often (0.8) and base act sampling damps ask/invite right after an answer turn, so an answered point usually gets one build/agree/challenge on the same thread before a fresh issue opens (same-thread development 55% -> ~74% on the full suite).
- P3 continuations: a continuation inherits its own previous focus (from the turn's option refs, falling back to the routed intent), a pending direct question to another sim damps continuation probability, and a deterministic CONTINUATION_TOPIC_JUMP check repairs drafts that name only disjoint options.
- P4 short reactions: the short-beat draw is more frequent and deeper (0.22+0.28*(1-verbosity), factor 0.42-0.62), reactive intents' length_hint="short" now actually shrinks the budget, and a budget-aware one-beat prompt note replaces the compact-argument style when max_words <= 8. New tiny_turn_rate (<=5 words) metric; evaluation's expected-words mirror updated to match.
- P5 openings: leading names are functional-only (invite; addressed ask/challenge; addressed answer at n>=4) with proactive group-size-scaled suppression on ordinary turns, and option-name openings are damped when the previous turn already discussed the same option. Name-prefix rate at n=4/5 dropped from ~0.25 to <=0.09 with direct-response rate still 1.0.
- P6 bounded compromises: a deterministic HYBRID_COMPROMISE tripwire (parsing.hybrid_blend_detected) blocks compromise turns that weld two options into one plan ("X and also Y", "combined with"); coordinated pairs in vote lines already parse to None and get repaired. Compromise turns pin one option plus a condition.
- P7 malformed turns: three salvage root causes fixed in utils.clean_generated (thousands-separator commas are no longer clause boundaries — "…given the $40." bug; title abbreviations no longer count as sentence ends — "…but Dr."; verbless coordinated tails are stripped), and validation blocks bare marker heads ("Just to be clear.") and lone subordinate clauses (non-answer acts) as MALFORMED_UTTERANCE with a matching repair instruction.
- P8 implied unsupported claims: a free deterministic pre-judge check flags quantities whose unit class exists nowhere on the board ("25-51 inches" with no length attributes) with word-boundary number matching, the asserted-workaround regex covers "can just book", and the LLM judge prompt gained a STRICT specifics rule (any off-board number/menu item/feature is unsupported; only arithmetic on listed numbers is allowed). Fresh runs: unsupported_printed_turns 0, no off-board numbers in printed text.
- P9 vote reliability: vote/accept prompts now suggest a rotating menu of parser-recognized commitment phrasings not yet used in the round (parsing.unused_commitment_phrases), and "my vote stays with X" was added to the commitment regexes. UNCLEAR_VISIBLE_COMMITMENT repairs went from 2-4 per run to 0 across n=3/4/6 targeted runs while vote phrasing stayed varied.
- P10 peer closing: when moderator closing is off, one deterministic participant-owned wrap-up line closes the run (dialogue._emit_peer_closing) — "Okay, X it is", "So X wins for most of us, with N still not sold", "Looks like we're not landing this one today" — natural wording, no vote counts, holdout-aware (a holdout caller says they're still on their pick).
- P11 dominance recheck: free-discussion shares track trait targets within ±0.11 (usually ±0.05), top shares 0.28-0.53; damping unchanged.

Round-level metric movement (full suite, before -> after): repair rate 0.113 -> 0.077, unsupported printed turns 0.25 -> 0.08 per run, tiny-turn rate 0 -> 0.053, name-prefix rate 0.115 -> 0.084, switch explanation 0.73 -> 0.82, avg words/turn 14.4 -> 13.5, tokens/run slightly down. Only monitoring items M1 (rare cross-option attribute mixups) and M2 (split-reservation addressee ordering) remain open in `docs/todo.md`. P4 more short reaction turns, P5 fewer name/option openings, P6 bounded hybrid compromises, P7 malformed compressed utterances, P8 implied unsupported constraints, P9 vote turns less repair-dependent, P10 participant closing without moderator, P11 dominance recheck. See `docs/todo.md` for details.

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
