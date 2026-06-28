# AGENTS.md

## Project

This is a Python CLI that simulates a 2-7 person group discussion. The deterministic controller owns speaker selection, dialogue intent, state, pacing, and consensus; the LLM only renders each stateless turn.

## Run and test

Use the repository's prebuilt virtual environment:

```powershell
# Interactive
& .\ds_proj\Scripts\python.exe .\main.py

# Headless
"Plan a weekend team offsite" | & .\ds_proj\Scripts\python.exe .\main.py

# Batch: one topic per line; lines beginning with # are ignored
& .\ds_proj\Scripts\python.exe .\main.py .\evals\topics.txt

# Offline unit tests
& .\ds_proj\Scripts\python.exe -m pytest .\tests -v

```

Live runs require a reachable provider; there is no offline or mock LLM mode.

## Provider rules

- Provider selection, models, endpoints, sampling, and timeouts live under `llm` in `config.yaml`.
- Supported providers are `uni`, `groq`, `gemini`, and `gpt`. Credentials come from `.env` as `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`; `uni` uses the configured Bamberg Ollama endpoint and requires VPN access.
- Behavioral validation must use `uni`. Never substitute Groq or Gemini because `uni` is slow or unavailable. If `uni` cannot be reached, stop and report it. Use another provider for validation only when the user explicitly requests it in the current task.

## Repository layout

- `main.py`: CLI entry point.
- `config.yaml`: all tunable numeric parameters and provider configuration.
- `src/prompts.py`: all prose sent to the LLM or printed as moderator text.
- `src/models.py`: typed scenario, persona, turn, runtime, and outcome state.
- `src/builders.py`: scenario and persona setup; invalid worlds retry, then raise.
- `src/router.py`: speaker and dialogue-act selection.
- `src/dialogue.py`: orchestration, phase control, state tracking, and consensus.
- `src/parsing.py`: machine-trailer and option-reference parsing.
- `src/scoring.py`: shared lean and option-support calculations.
- `src/validation.py`: deterministic guardrails and repair decisions.
- `src/llm_client.py`: stateless provider abstraction.
- `src/logger.py`: writes `logs/<run_id>/transcript.md`, `run.json`, optional prompts, and `logs/metrics.csv`.
- `tests/`: offline deterministic tests. `evals/topics.txt`: optional batch topic corpus.
- `docs/known_failures.md`: current issue backlog and validation protocol.

## Design constraints

- Keep fixes topic-agnostic and valid for every group size from 2 through 7.
- Put tunable numbers in `config.yaml`; do not scatter numeric dials through code.
- Put LLM and moderator prose in `src/prompts.py`.
- Do not fabricate fallback content when setup or generation fails; surface the failure.
- Preserve commitment gating: accept, vote, and reject become binding only during routed decision phases. Hedged acceptance stays neutral.
- Outcomes use visible commitments only: `successful` is unanimous visible support, `majority` meets the configured fallback fraction, and otherwise the result is `unresolved`.
- Stubbornness is trait-driven through `agreeableness == 1`; do not add a separate hard-blocker flag or mechanical vote override.
- Pacing is derived from group size and composition, not a fixed turn count.
- Keep prompts compact. When adding a prompt rule, trim or merge an existing rule. Never seed guidance with quoted example phrases; models copy them into dialogue.
- Keep option claims grounded in option cards. Missing facts are unknown, not an invitation to invent details.
- Do not add forced routing or injected turns solely to make dialogue seem natural; naturalness should emerge from traits and existing dynamics.

## Conversation target

- Discussions should read like friends making a decision together: casual and plain-spoken, but neither slang-heavy/Gen-Z nor corporate, academic, or presentation-like.
- Persona traits must be visible in conversational behavior such as directness, caution, curiosity, compromise, and initiative. Do not express traits through stereotypes, catchphrases, or repeated self-description.
- Configured response length must produce observable differences between personas, while even the longest setting remains appropriate for a chat rather than a speech or mini-essay.
- Turns should respond locally to what another participant just said. Avoid standalone option pitches, card summaries, and unnecessarily complex sentences.

## Change workflow

For simulator-quality fixes, use `docs/known_failures.md` as the source of truth. Each upgrade is one issue and one independently verifiable task unless the user explicitly groups issues:

1. Add a failing unit test first when the behavior is deterministic, especially for parsing or validation.
2. Implement the smallest targeted change.
3. Run the full offline test suite.
4. Validate with one mandatory `n=3` live run on the provider explicitly authorized for the task, then the requested spread across `n=2-7` when behavioral validation is required.
5. Read every relevant `transcript.md` and `run.json` as well as metrics. Do not proceed to another issue until the target behavior improves without an obvious regression.
6. Before completing the upgrade, audit and synchronize every applicable information source: `AGENTS.md`, `CLAUDE.md`, repository skills, active memory/index files, `docs/known_failures.md`, `README.md`, and other workflow documentation. Historical per-fix records may remain unchanged, but active guidance must not contradict the current repository.
7. Stop at the completed upgrade boundary unless the user explicitly asks to continue automatically.
