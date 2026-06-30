# AGENTS.md

## Project

This is a Python CLI that simulates a 2-7 person group discussion. The deterministic controller owns speaker selection, dialogue intent, state, pacing, and consensus; the LLM only renders each stateless turn.

## Run

Use the repository's prebuilt virtual environment:

```powershell
# Interactive
& .\ds_proj\Scripts\python.exe .\main.py

# Headless
"Plan a weekend team offsite" | & .\ds_proj\Scripts\python.exe .\main.py

# Batch: one topic per line; lines beginning with # are ignored
& .\ds_proj\Scripts\python.exe .\main.py .\evals\topics.txt

```

Live runs require a reachable provider; there is no offline or mock LLM mode.

## Provider rules

- Provider selection, models, endpoints, sampling, and timeouts live under `llm` in `config.yaml`.
- Supported providers are `uni`, `groq`, `gemini`, and `gpt`. Credentials come from `.env` as `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`; `uni` uses the configured Bamberg Ollama endpoint and requires VPN access.
- Validation must use the provider explicitly authorized for the current task. Never silently substitute a different endpoint. If the authorized provider cannot be reached, stop and report the failure.

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
- `evals/topics.txt`: optional batch topic corpus.
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
- Attaching an exact recent message to an already-routed contribution is context, not a forced turn. Prefer the most recent relevant option point, then the latest other-speaker turn.

## Conversation target

- Discussions should read like friends making a decision together: casual and plain-spoken, but neither slang-heavy/Gen-Z nor corporate, academic, or presentation-like.
- The five OCEAN traits must be visible through behavior such as directness, caution, curiosity, compromise, and initiative. Response length and compromise willingness are derived from those traits; do not generate separate initiative, directness, or detail state. Do not express traits through stereotypes, catchphrases, or repeated self-description.
- OCEAN also shapes dialogue-act weights and act-specific turn behavior: openness favors exploration, conscientiousness concrete constraints, extraversion participation and initiative, agreeableness building versus challenging, and neuroticism risk sensitivity. These remain derived effects, not new persona fields.
- Configured response length must produce observable differences between personas, while even the longest setting remains appropriate for a chat rather than a speech or mini-essay.
- Non-opening discussion replies should receive one exact local message to answer. Targeted replies omit the full background/private goal and must not restart the speaker's option pitch or biography. Openings retain personal context so the initial positions still feel motivated.

## Change workflow

For simulator-quality fixes, use `docs/known_failures.md` as the source of truth. Each upgrade is one issue and one independently verifiable task unless the user explicitly groups issues:

1. Implement the smallest targeted change.
2. Before live validation, move existing timestamped run directories from `logs/` into `logs/archive/`; preserve the archive and `logs/metrics.csv`.
3. Validate with one mandatory `n=3` live run on the provider explicitly authorized for the task, then the requested spread across `n=2-7` when behavioral validation is required.
4. Read every relevant `transcript.md` and `run.json` as well as metrics. Do not proceed to another issue until the target behavior improves without an obvious regression.
5. Before completing the upgrade, audit and synchronize every applicable information source: `AGENTS.md`, `CLAUDE.md`, repository skills, active memory/index files, `docs/known_failures.md`, `README.md`, and other workflow documentation. Historical per-fix records may remain unchanged, but active guidance must not contradict the current repository.
6. Stop at the completed upgrade boundary unless the user explicitly asks to continue automatically.
