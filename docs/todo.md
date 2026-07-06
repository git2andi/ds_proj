# TODO: latest active fixes

This file is the only active work queue for the next implementation round. It should contain only open issues. Remove or narrow an item only after code inspection, implementation, targeted example runs, full-suite validation, and manual transcript review show that the issue is solved.

The project remains an **option-grounded multi-user decision simulator**. Do not turn it into a generic chatbot or open-ended society simulator. Fixed options must remain central. Participants discuss options, compare tradeoffs, may shift stance, may reach compromise conditions, and finish with exactly one of:

- `successful`: all sims visibly agree on the same winning option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement is reached.

Do not add a fourth outcome label. If a participant cannot plausibly accept the winning option, the transcript should show refusal, reservation, or continued support for another option, which naturally leads to `majority` or `unresolved`.

Use `gpt` as the dialogue-generation provider unless a task explicitly compares providers.

## Required implementation protocol

For every issue below, follow this process exactly.

1. Work on **one issue at a time** unless the issue explicitly says it must be combined with another one.
2. Before starting, move existing logs into `logs/archive/` so new evaluation artifacts are easy to inspect.
3. Read the issue carefully and understand why it affects dialogue quality.
4. Read the relevant code files carefully before editing. Do not patch blindly.
5. Prefer deterministic controller/state/validation fixes over broad prompt expansion. Keep token usage under control.
6. Implement the smallest robust fix that solves the issue without breaking already-working behavior.
7. Run static checks, at minimum:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
```

8. Make targeted example runs before continuing:
   - one `n=3` run is mandatory;
   - then make 2-3 additional runs with varying group sizes from `n=2` to `n=7`;
   - inspect both `transcript.md` and `run.json`, not only metrics.
9. Confirm that the fix works across varying sim counts. If it does not, fix again before moving on.
10. If a new clear issue appears during this work and is not listed here, fix it directly if it blocks or distorts the current issue. Otherwise add it to this file with priority and evidence.
11. After the issues form the list are done and look good, run the full suite:

```powershell
py run_eval_suite.py --full
```

12. Inspect the full-suite logs manually. Ensure the intended behavior works there as well.
13. Update all relevant documentation after the verified fix:
   - `CLAUDE.md`
   - `README.md`
   - `docs/todo.md`
   - relevant `info/*.md` files
14. Only then remove or narrow the completed issue in this file.

## Open issues

The 2026-07-06 naturalness round (P1-P11) is complete; every item was closed against targeted runs plus the full 12-case suite (final suite 2026-07-06 19:31-19:42, all rc=0). P11 recheck result: free-discussion shares deviate from trait targets by at most ±0.11 (typically ±0.05), top free-discussion shares span 0.28-0.53, and engagement correlations are high except in near-flat casts where correlation is statistically meaningless — anti-dominance damping was left unchanged because it demonstrably does not erase legitimate dominance.

Small items worth monitoring (not blocking, escalate only if they recur):

- **M1 — occasional cross-option attribute mixups.** ~0.08 printed unsupported turns per run remain (e.g. "Cleaning Trial uses shared supplies" merging two cards' facts). The tripwire flags them; repair usually fixes them. If printed cases rise above ~0.2/run, extend the cross-option token check rather than the judge prompt.
- **M2 — split-reservation addressing.** In a two-holdout split round, the reservation exchange sometimes lets the caller voice their own reservation before the explicitly addressed holdout answers (q02 final suite, turns 87-88). Harmless conversationally; fix only if it starts reading as ignored questions.

## Non-goals for this round

Do not prioritize:

- full Generative Agents-style memory or reflection;
- full agenda simulation;
- open-ended roleplay or society simulation;
- new outcome labels;
- large prompt expansion;
- adding more personality traits;
- provider comparisons before the `gpt` baseline is stable;
- broad rewrites unrelated to the listed issues.
