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
- `src/aliases.py`: shared validation, fallback, and collision handling for option aliases.
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

## Run flow

1. `builders.py` generates option cards, then personas. The controller samples a configured primary-preference partition before provider calls and assigns each persona a required primary after option generation. Scenario and persona retries are isolated.
2. `router.py` produces a `MoveIntent`, prioritizing pending answers, unanswered challenges, coverage gaps, then weighted speaker selection across opening, discussion, narrowing, confirmation, and closure.
3. `dialogue.py` renders each routed move, parses its machine trailer, updates typed state, and checks consensus through the orchestrator, controller, tracker, and consensus manager.
4. `validation.py` repairs structural failures; style findings remain diagnostics. Moderator and farewell text receive a lightweight option-reference check before entering the transcript.
5. `logger.py` writes the transcript, structured run data, aggregate metrics, and optional prompt traces.

## Design constraints

- Keep fixes topic-agnostic and valid for every group size from 2 through 7.
- Put tunable numbers in `config.yaml`; do not scatter numeric dials through code.
- Put LLM and moderator prose in `src/prompts.py`.
- Do not fabricate fallback content when setup or generation fails; surface the failure.
- Preserve commitment gating: accept, vote, and reject become binding only during routed decision phases. Visible support requires one named option plus an explicit first-person choice or acceptance; typographic apostrophes are normalized, and uncertain or conditional wording stays neutral.
- Outcomes use visible commitments only: `successful` is unanimous visible support, `majority` meets the configured fallback fraction, and otherwise the result is `unresolved`.
- Stubbornness is trait-driven through `agreeableness == 1`; do not add a separate hard-blocker flag or mechanical vote override.
- Pacing is derived from group size and composition, not a fixed turn count.
- Keep prompts compact. When adding a prompt rule, trim or merge an existing rule. Never seed guidance with quoted example phrases; models copy them into dialogue.
- Keep option claims grounded in option cards. Missing facts are unknown, not an invitation to invent details.
- Do not add forced routing or injected turns solely to make dialogue seem natural; naturalness should emerge from traits and existing dynamics.
- Attaching an exact recent message to an already-routed contribution is context, not a forced turn. Prefer the most recent relevant option point, then the latest other-speaker turn.
- Count option coverage only from option names visible in generated text. Count a reason only when a claim slot is present.
- Ground ASK and ANSWER turns in option-card attributes and shared context. Unknown facts stay unknown; a second hedged answer closes an unanswerable question.
- Keep moderator state language distinct from binding votes. Never say someone voted without a visible vote, and do not propose merging fixed options.
- Classify a pending participant from transcript evidence as a visible supporter, an actual alternative/objection, or a missing explicit commitment. Ask the last group only for confirmation; never call them holdouts.
- Keep confirmation from cycling on the same failed acceptance: a repaired but still unclear confirmation becomes a soft rejection for routing purposes.
- Reject setup before dialogue when a topic's explicit participant count contradicts configuration or an option name is structurally truncated.

## Conversation target

- Discussions should read like friends making a decision together: casual and plain-spoken, but neither slang-heavy/Gen-Z nor corporate, academic, or presentation-like.
- The five OCEAN traits must be visible through behavior such as directness, caution, curiosity, compromise, and initiative. Response length and compromise willingness are derived from those traits; do not generate separate initiative, directness, or detail state. Do not express traits through stereotypes, catchphrases, or repeated self-description.
- OCEAN also shapes dialogue-act weights and act-specific turn behavior: openness favors exploration, conscientiousness concrete constraints, extraversion participation and initiative, agreeableness building versus challenging, and neuroticism risk sensitivity. These remain derived effects, not new persona fields.
- Configured response length must produce observable differences between personas, while even the longest setting remains appropriate for a chat rather than a speech or mini-essay.
- Non-opening discussion replies should receive one exact local message to answer. Targeted replies omit the full background/private goal and must not restart the speaker's option pitch or biography. Openings retain personal context so the initial positions still feel motivated.
- Reply-capable acts receive the focused option's most recent relevant turn, falling back to the latest other-speaker turn. Repair prompts retain that target.
- Turn behavior uses at most two act-specific OCEAN cues. Trait effects remain derived: routing uses `trait_act_slope`, response length uses openness/conscientiousness/extraversion, and objection behavior reflects agreeableness, caution, and directness.
- Social beats are optional, limited to one probability-gated line from the most extraverted persona, and must not duplicate moderator work.
- Deterministic surface cleanup removes malformed punctuation and boilerplate openers while preserving the machine trailer. Structural validation still decides whether a participant turn needs repair.
- A repair may not replace a state-valid turn with a new state-blocking defect; retain the original generated line when a rewrite regresses semantic validity.

## Pacing and outcomes

- Compute discussion floors and hard limits from group size and composition. Full convergence plus stall, or exhausted claim slots plus stall, may end discussion early.
- If every remaining voter exhausts configured vote attempts, advance from narrowing to confirmation instead of cycling.
- Only `explicit_vote` and `accepted_options` count toward outcomes. Hidden preferences, routing leans, and unverified trailers do not.
- Finalization scans visible support across every option rather than trusting the controller's current candidate. Moderator summaries and closures use visible preferences/commitments rather than hidden or routing-derived leans.
- `successful` requires unanimous visible support. `majority` requires the configured visible-support fraction. Everything else is `unresolved`.

## Outputs and configuration

Each run writes `logs/<run_id>/transcript.md`, `run.json`, and optionally `prompts.jsonl`; aggregate rows append to `logs/metrics.csv`. Core diagnostics include outcome status, support fraction, repair/flag counts, question density, turn length, and option coverage.

The primary tuning controls are grouped under `DIALS THAT MATTER` in `config.yaml`. Preference partition weights are defined separately for group sizes 2-7; `forced_shape` is reserved for controlled runs.

## Change workflow

For simulator-quality fixes, use `docs/known_failures.md` as the source of truth. Each upgrade is one issue and one independently verifiable task unless the user explicitly groups issues:

1. Implement the smallest targeted change.
2. Before live validation, move existing timestamped run directories from `logs/` into `logs/archive/`; preserve the archive and `logs/metrics.csv`.
3. Validate with one mandatory `n=3` live run on the provider explicitly authorized for the task, then the requested spread across `n=2-7` when behavioral validation is required.
4. Read every relevant `transcript.md` and `run.json` as well as metrics. Do not proceed to another issue until the target behavior improves without an obvious regression.
5. Before completing the upgrade, audit and synchronize every applicable information source: `AGENTS.md`, `CLAUDE.md`, repository skills, active memory/index files, `docs/known_failures.md`, `README.md`, and other workflow documentation. Historical per-fix records may remain unchanged, but active guidance must not contradict the current repository.
6. Stop at the completed upgrade boundary unless the user explicitly asks to continue automatically.
