# Development guide

Read `README.md`, `SIMPLIFICATION_ACTION_PLAN.md`, and `info/00_overview.md` before changing runtime behavior.

## Architectural invariants

1. A `UserSimulator` independently constructs and selects one complete `UserAction`, or remains silent.
2. Engagement controls voluntary willingness. Required openings, direct answers, and votes bypass willingness.
3. The floor selects an intact bid. It may apply priority and light turn balancing, but it must not rewrite the act, focus, reason, addressee, movement, or vote.
4. The LLM realizes opening, discussion, narrowing, and required-answer wording only. It does not choose speakers, actions, stance changes, or votes. Formal vote wording is deterministic.
5. Ordinary actions are `REACT`, `SUPPORT`, `OBJECT`, `COMPARE`, `ASK`, and `ACCEPT`; protocol actions are `OPENING`, `ANSWER`, and `VOTE`.
6. At most one lightweight `DiscussionThread` exists. The first direct answer may be required; later related contributions remain voluntary and bounded.
7. A thread closes after no related bid or the configured turn cap. Do not reintroduce issue status, partial-resolution, owner-reaction, or stale-thread state machines.
8. Point keys use structured option attributes. An already public point must not open another question, and later thread turns should add a new point, comparison, or movement.
9. Public preferences and acceptances change only after a structured movement utterance explicitly names its target option. The simulator owns the movement decision; the validator does not infer acceptance from a phrase list. Votes must visibly match the structured vote.
10. Openings, preference movements, and votes require an explicit validated option reference. Missing an exact alias is soft for ordinary discussion turns and must not cause an otherwise usable question, reaction, objection, answer, support, or comparison to be dropped.
11. Runtime validation covers hard correctness failures only. Do not reject understandable wording because it misses a narrow template, omits a redundant option alias, or uses words such as `guarantees`.
12. Repair is allowed only for openings. Required answers are not semantically rescored or replaced by fallbacks. Formal votes are deterministic; invalid voluntary turns are dropped and logged.
13. A deterministic opening is a last-resort protocol fallback, not a substitute for accepting valid generated aliases.
14. Never add an LLM validator or judge to the live dialogue path.
15. Scenario setup is structurally validated and receives three total complete-generation attempts (the initial attempt plus up to two feedback-guided regenerations). Do not reintroduce generic superlative inference, missing-attribute inference, or selective semantic repair.
16. Alias and participant-name generation share one lightweight setup call after the option board is valid. Request one or two aliases per option and one unique first name per participant. Invalid names use local fallbacks and are propagated consistently through the persona card. Automatic aliases must derive from the full name, contain at least two words and no numbers, remain unique after normalization, and not end in an incomplete connector.
17. Option coverage is observational only and must not drive mandatory discussion.
18. Any strict public-preference majority proceeds to voting. Without a majority, derive at most one leader from public preferences and visible acceptances. When several strongest options remain tied, use the run's seeded RNG to select one tied option as the bounded compromise target, including complete preference splits. The moderator names current holdouts and asks whether that target fits their requirements. Up to two holdouts may accept it with a grounded reason, reject it with a remaining concern, or remain silent.
19. Preserve the participant's latest public stance during formal narrowing: a prior visible acceptance of the leader carries forward without another random draw. Otherwise rank-4 holdouts with stubbornness 1–3 accept, rank-4 with stubbornness 4 remains probabilistic, rank-5 accepts, and ranks 1–3 or hard blockers do not move. A positive formal-narrowing acceptance may switch only to the selected public leader in the participant's final vote. Compromise happens before one authoritative final vote; do not allow reciprocal narrowing targets or routine re-voting.
20. The moderator remains deterministic and sparse: opening, one optional liveness intervention, at most one narrowing question, an optional vote request, and closure. It never narrates whether movement did or did not occur.
21. Keep realization prompts compact but retain previous-turn connection, varied openings, recent-own-wording avoidance, and natural pronoun/reference guidance.
22. Keep the runtime explainable for an eight-page project report. Do not add emotions, coalitions, deception, unrestricted memory, or research-scale infrastructure.
23. When removing a concept, remove its models, config fields, prompts, metrics, tests, and documentation in the same change.
24. `SUPPORT`/`OPENING`/`ACCEPT` use positive stance evidence; `OBJECT`/concern questions use negative stance evidence. Neutral attributes may support reactions and comparisons but must not be arbitrarily converted into objections.
25. Every `COMPARE` action carries the same named public attribute from both focused options. Encourage both references for clarity, but do not drop an otherwise usable comparison solely because an exact alias is missing. Keep comparisons infrequent, state both values accurately, and never force one fixed sentence template or infer an additional fact.

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

The discussion loop processes a required answer first; otherwise it collects simulator bids, selects one, realizes it, validates it, commits visible state, and maintains or closes the bounded thread.

## Testing

```powershell
py -m pytest -q
```

Tests should verify ownership, structural setup, alias isolation, bidding/willingness, bounded threads, point reuse, contextual references, visible movement, hard validation, one final voting round, logging, and evaluation scripts. Do not test deleted issue/coverage machinery.

## Evaluation entry points

- `eval/run_eval_suite.py`: small focused LLM-backed development suite;
- `eval/run_scenarios.py`: broader topic batch;
- `eval/summarize_runs.py`: deterministic report-facing summary;
- `eval/judge_transcripts.py`: independent post-hoc LLM judges.

Generated log folders are artifacts, not source. Judges must never be invoked by the runtime. Moderator turns contribute to interaction-level judge dimensions, but the moderator is never evaluated as a persona.
