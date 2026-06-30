# AGENTS.md

Guidance for coding agents working on this repository.

## Purpose

This is a university project for generating natural group-decision discussions with 2-7 simulated participants. The system creates a grounded option board, samples personas with OCEAN traits and initial preferences, then runs a small multi-party discussion until there is a visible decision outcome.

## Stable anchors

- `main.py` is the CLI entry point.
- `config.yaml` contains tunable parameters. Do not scatter new dials through code.
- `src/prompts.py` contains all prose sent to LLMs. Do not put LLM prompts in other modules.
- `src/llm_client.py` owns provider integration.
- `logs/` stores generated run artifacts.

## Current architecture

- `src/builders.py`: setup. It generates option cards and personas. Initial preferences are sampled before persona prompting and enforced during parsing.
- `src/dialogue.py`: discussion engine. It owns phases, lightweight move routing, state mutation, moderator nudges, vote rounds, and consensus.
- `src/parsing.py`: metadata trailer parsing and option-reference detection.
- `src/logger.py`: transcript, JSON, metrics, and optional prompt logs.

Removed complexity: the old separated router/scoring/validation stack was deleted. Do not recreate many small files unless there is a clear, tested need.

## Behavioral contract

- Options are the source of truth. Generated dialogue must stay grounded in option cards and shared context.
- Participants have initial preferences but may compromise when trait-derived behavior and visible dialogue make it plausible.
- Hard blockers are represented by traits, especially agreeableness=1, plus an optional grounded rejection. They should remain civil but very resistant.
- The moderator is not a participant. It opens the option board, occasionally nudges stalled discussion, and closes the run.
- No generated greetings or farewells.
- A run must not close before a visible decision opportunity.
- Outcomes use visible votes/acceptances only: unanimous = `successful`, unique majority = `majority`, otherwise `unresolved`.

## Validation workflow

1. Compile first: `py -m compileall .`.
2. Run smoke tests with n=2, n=3, n=5, and n=7 using fresh topics.
3. Read transcripts manually. Check local flow, grounding, trait expression, moderator frequency, and final outcome correctness.
4. Update `docs/known_failures.md` when an issue is fixed or discovered.
