---
name: improve-simulator
description: Improve the group-discussion simulator one tracked issue at a time across naturalness, persona expression, convergence, state integrity, moderation, parsing, grounding, and endpoint portability.
---

# Improve the simulator

Follow `AGENTS.md` and `docs/known_failures.md` as the authoritative policy and backlog. Keep every change topic-agnostic, valid for group sizes 2 through 7, and independent of one provider's wording habits.

## Upgrade boundary

One upgrade is one backlog issue and one independently verifiable task unless the user explicitly groups issues. Complete its tests, live evidence when required, transcript review, and information-file synchronization before starting another issue. Stop at that boundary unless the user explicitly requests automatic continuation.

## Diagnose

1. Read `docs/known_failures.md` and the relevant implementation.
2. Inspect the newest `logs/<run_id>/transcript.md` and `run.json` files.
3. Distinguish controller defects from provider-specific generation tendencies. Fix the endpoint-independent contract, not one model's favorite wording.
4. For conversation quality, assess whether the exchange sounds like plain-spoken friends: neither Gen-Z/slang-heavy nor corporate, academic, or presentation-like. Check local responsiveness, sentence complexity, repetition, visible trait behavior, and response length by trait.

## Implement one issue

1. Add a failing unit test first for deterministic parsing, validation, routing, scoring, or state behavior.
2. Make the smallest topic-agnostic change valid for group sizes 2 through 7.
3. Put numeric dials in `config.yaml` and all generated or moderator prose in `src/prompts.py`.
4. Preserve visible commitment gating, trait-driven stubbornness, grounded claims, configured outcome rules, and explicit failure on unusable setup or generation.
5. Do not use provider-specific regexes, phrase lists, quoted prompt examples, forced naturalness turns, or injected dialogue.

Use these ownership boundaries:

- `src/parsing.py`: trailers and option references.
- `src/validation.py`: deterministic guardrails and repair decisions.
- `src/router.py`: speaker and dialogue-act selection.
- `src/dialogue.py`: orchestration, phases, state, and outcomes.
- `src/scoring.py`: shared lean and option-support calculations.
- `src/llm_client.py`: stateless provider adapters only.

## Verify

1. Run the full offline suite:

   ```powershell
   & .\ds_proj\Scripts\python.exe -m pytest .\tests -v
   ```

2. Use the provider explicitly authorized by the current user and `AGENTS.md`. If it is unavailable, report the failure; do not silently substitute another endpoint.
3. Run the required live spread across relevant topics and participant counts.
4. Read every relevant transcript and `run.json`; metrics alone are insufficient.
5. Compare evidence with the issue's acceptance criteria and check state, grounding, outcome, persona, and cost regressions.
6. Close an issue only when visible behavior improves without an obvious regression.

## Maintain the backlog

Keep `docs/known_failures.md` limited to currently open, reproducible issues. Record evidence, endpoint scope, relevant code, and the smallest provider-independent fix direction. Remove stale claims that the supplied code disproves; do not claim resolution based only on unit tests or one provider run.

Consolidate overlapping symptoms under one issue and retain historical IDs when useful. Before completing each upgrade, audit and update every applicable active information source: `AGENTS.md`, `CLAUDE.md`, both repository copies of this skill, active memory/index files, `docs/known_failures.md`, `README.md`, and other affected workflow documentation. Historical per-fix memories remain historical; active guidance must describe the current repository.
