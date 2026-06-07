# Implementation Notes

This refactor replaces the previous patch-heavy runtime with a service-based architecture.

## Main architectural changes

- `schemas.py` defines typed data objects for scenarios, options, personas, dialogue acts, runtime state, move intents, validation results, and outcomes.
- `scenario_builder.py` owns option/scenario generation and validates structured option cards.
- `persona.py` now builds cooperative-by-default personas. Rare hard blockers are sampled explicitly and no longer emerge accidentally from ordinary low-agreeableness traits.
- `act_parser.py` is the single canonical parser for vote, accept, reject, question, addressee, and compromise-proposal signals.
- `state.py` owns structured dialogue state and updates coverage, open questions, stances, votes, acceptances, rejections, and compromise proposals.
- `controller.py` replaces fixed phase lengths with adaptive readiness scoring.
- `router.py` returns `MoveIntent` objects instead of just speakers. The LLM receives a local act such as answer, compare, push_back, propose_compromise, vote, or accept.
- `utterance_generator.py` generates exactly one participant message and handles one repair attempt plus deterministic fallback.
- `validator.py` performs deterministic checks for structure, invalid option references, ungrounded numeric facts, repetition, decision clarity, and register problems.
- `consensus.py` requires explicit public acceptance or votes. Private initial acceptability is no longer treated as public consensus.
- `logger.py` writes a readable transcript and structured JSON metrics.
- `orchestrator.py` now coordinates services rather than owning all runtime logic.
- `prompts.py` contains all LLM-facing prose and deterministic moderator templates.
- `config.yaml` contains all tunable thresholds and sampling profiles.

## Important behavior changes

Normal sims are compromise-seeking. They may push back or prefer different options, but only the explicit rare hard-blocker can behave as a true veto player. Initial acceptable options remain private beliefs; they are not counted as public acceptance until the participant says so in the chat.

The discussion no longer uses fixed phase lengths. It begins with each participant stating an initial priority, then uses coverage, reason depth, participant participation, open questions, and no-progress signals to decide when to narrow.

The moderator is deterministic and minimal: opening and closure only in the base flow. The router lets simulated participants propose compromises before formal confirmation.

## Compatibility files

`policy.py`, `simulator.py`, `prompt_context.py`, and `verifier.py` are now compatibility shims. They no longer contain the old active logic. The uploaded `state(1).py` was moved to `_legacy_uploads/state_legacy.py`; the active module is now `state.py`.

## Smoke test

The project was syntax-checked with:

```bash
python3 -m py_compile *.py
```

A mocked LLM run was also executed to verify that the orchestration loop, state updates, validation path, consensus detection, and logging complete without requiring access to the real `uni` endpoint.
