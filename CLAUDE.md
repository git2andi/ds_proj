# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university project: an LLM-driven group discussion simulator. Given a topic, it generates a multi-party chat where 2–7 personas debate and try to reach a decision. A deterministic controller (routing, state tracking, consensus) decides **who speaks, when, and with what intent**; the LLM only renders the surface text of each turn via a single stateless call per turn.

## New machine setup

Memory files are stored in `.claude/memory/` in the repo. On first use on a new machine, copy them to Claude's config directory so they load automatically:

```powershell
$dest = "$env:USERPROFILE\.claude\projects\C--Users-Andi-Desktop-ds-proj\memory"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item .\.claude\memory\* $dest -Force
```

(Assumes the repo is cloned to `C:\Users\Andi\Desktop\ds_proj`. If the path differs, adjust the `C--Users-Andi-Desktop-ds-proj` folder name to match: replace `\` with `-` and `:` with nothing, `_` with `-`.)

## Running

Activate the `ds_proj` virtualenv. No requirements.txt install step — the venv is pre-built.

```powershell
# Interactive single run
(ds_proj) PS> py .\main.py
Topic: Plan a weekend team offsite

# Batch (one topic per line, # comments ignored)
(ds_proj) PS> py .\main.py scenarios.txt

# Headless (pipe a topic, using the venv python directly)
"Example Topic" | & .\ds_proj\Scripts\python.exe .\main.py
```

Provider is set in `config.yaml` under `llm.provider`: `uni` (Bamberg Ollama endpoint, requires VPN), `groq`, `gemini`, or `gpt`. API keys come from `.env`. There is no offline/mock mode: a run requires a reachable provider and raises on failure. Validation uses the provider explicitly authorized for the current task; do not silently substitute another endpoint.

## Architecture

The split is deliberate and load-bearing:

- **`config.yaml`** — every tunable number lives here, nowhere else.
- **`src/prompts.py`** — every piece of prose sent to an LLM or printed as moderator text lives here, nowhere else.
- **`src/models.py`** — typed state objects (`dataclass(slots=True)`). Routing, consensus, logging, and validation operate on these, not raw dicts.
- **`src/validation.py`** — deterministic guardrails (detection + logging). Style checks are warn-level diagnostics; only structural errors trigger LLM repair. Also houses discourse-frame classification and claim-slot tracking used by prompts to nudge variety.

### Run flow

1. **`builders.py`** — two sequential LLM calls: first generates option cards (with `shared_context` and `short_name`), then per-persona hidden belief state. Traits, names, preference camps, and a rotating common-compromise option are selected by the controller and passed to setup. Generated participant counts, aliases, score/list consistency, camp structure, and shared compromise are validated; invalid worlds retry up to `setup_generation_attempts` and then raise rather than being silently rewritten.
2. **`router.py`** — emits a `MoveIntent` each turn. Priority: (1) answer pending question, (2) respond to unanswered challenge via `_unanswered_challenge`, (3) fill coverage gaps, (4) weighted speaker selection. Drives phases: opening → discussion → narrowing → confirmation → closure.
3. **`dialogue.py`** — orchestration: calls the router, renders the turn via `prompts.sim_utterance`, parses the trailer, updates state, checks consensus. Contains `Orchestrator`, `DialogueController`, `StateTracker`, `ConsensusManager`.
4. **`parsing.py`** — extracts the machine trailer `[act=…; opt=…; stance=…]` from each generated turn, resolves option references.
5. **`scoring.py`** — shared `current_lean` / `leading_option` used by routing, consensus, and moderator prompts.
6. **`validation.py`** — deterministic checks. Structural errors (missing trailer, invented option, malformed vote) trigger repair. Style issues (robotic phrasing, self-narration, card-reading, repeated starts) are logged as warnings only. Possessive openers (`strip_possessive_opener`) and `Considering X,` openers (`_strip_considering_opener`) are stripped deterministically before validation so they never reach the transcript. Discourse-frame classification (`classify_discourse_frames`) and claim-slot tracking (`classify_claim_slots`) provide variety hints to the prompt layer.
7. **`llm_client.py`** — thin provider abstraction (uni/groq/gemini/gpt). Each turn is a stateless call; the full prompt is re-sent every time (no session memory on the endpoint side).
8. **`logger.py`** — writes `transcript.md`, `run.json`, `metrics.csv`, and optional `prompts.jsonl` per run under `logs/<run_id>/`.

### Prompt design

The per-turn prompt uses a compact `runtime_speaker_card`, not the full persona profile. It includes the current lean, one concern, behavior cues derived from traits, and one prior self-point to avoid repeating. Generated role and speech-style labels are deliberately omitted because they can induce stereotype-like or formal prose. Each turn has one local job; direct-response context names one exact routed turn and is removed from the recent-chat block to avoid duplication. Router-selected focus matching can identify a relevant recent turn without injecting a new speaker or act. Full option cards are rendered only for OPENING/COMPARE/VOTE; other acts get option names. Recognizable aliases are shared by setup, prompting, parsing, and validation.

Face-work modifiers describe behavior for objection/push-back acts: agreeable speakers acknowledge the other side, cautious speakers surface risk, and direct speakers put the problem early. Response length is trait-adjusted, monotonic, capped by `utterances.max_chat_words`, and even the longest setting is limited to two short sentences.

The target register is a chat among friends: casual and plain-spoken without Gen-Z slang, corporate jargon, formal debate language, or mini-essays. Traits and configured response length must be visible in behavior and turn length without becoming stereotypes or repeated verbal tics. Direct-response turns should answer or engage with the local point before introducing another option claim.

### Key design constraints

- **No fabricated fallbacks**: if setup or a turn call returns something unusable, the run raises rather than papering over it with defaults.
- **Commitment gating**: `accept`/`vote`/`reject` only count as binding on routed decision turns (narrowing/confirmation), not during free discussion. Hedged accepts ("still not sure", "not fully sold") are clamped to neutral.
- **Trait-driven stubbornness**: a participant sampled with `agreeableness=1` (4% per-run chance, at most one per run) only accepts their preferred option. The LLM plays this from the trait card; the router uses the trait value for vote-focus and persuasion gating. No mechanical vote override — behaviour emerges from traits. If the LLM's trailer names a different option on a VOTE/ACCEPT turn, `HARD_BLOCKER_WRONG_VOTE` repair fires.
- **Outcome taxonomy**: `successful` = every participant visibly voted/accepted the same option; `majority` = one option has ≥66% visible support; `unresolved` = no option met either threshold. Only explicit `explicit_vote` and `accepted_options` entries count — hidden preferences and routing leans are excluded.
- **Pacing is derived, not fixed**: `min_discussion_turns`, `force_narrow_turns`, `hard_max_turns` are computed per run from group size and composition.
- **Concession bridges**: when a speaker accepts/votes for a non-preferred option, persona-specific bridge guidance fires.
- **Prompts stay short**: llama3.3 ignores excess rules. Every new rule added to `sim_utterance` should come with an old one being cut or merged.
- **Contribution-based routing**: for n>=4, speakers are skipped when their only available move is restating a known preference. Extraversion and initiative drive turn frequency.
- **Stall-to-concrete routing**: when discussion stalls (2+ turns no progress), ASK probability doubles and SUPPORT dampens, steering toward concrete questions instead of preference restating.
- **Surface cleanup**: deterministic removal of space-before-punctuation, repeated punctuation, stray quotes in `clean_generated`; possessive openers (`OptName's`) and `Considering X,` openers stripped after `clean_generated` so they never survive into the transcript.
- **No example seeding in guidance**: guidance strings must describe desired behavior, never demonstrate it with quoted phrases (e.g., never `"Try: 'What if we...'"`) — the model copies examples verbatim across all topics.
- **Epistemic grounding rule**: the per-turn prompt tells the model "you know only what's in the option cards — anything else is unknown: say 'I'm not sure' or 'we'd need to check'". This prevents confident invented facts about real-world named places (seating, pricing, amenities not in the cards).

## Outputs

Each run writes to `logs/<run_id>/`:
- `transcript.md` — human-readable chat + outcome + metrics
- `run.json` — full structured run (per-turn acts, validation issues, `repair_trigger_codes` on repaired turns)
- `logs/metrics.csv` — master file, one row appended per run

Key metrics: `outcome_status`, `final_support_fraction`, `repaired_turns`, `flagged_turns`, `question_density`, `avg_words_per_turn`, `option_coverage`.

## Validation and diagnostics

### Implementation process

Every upgrade follows this cycle (see `docs/known_failures.md` for the full protocol). One upgrade is one issue and one independently verifiable task unless the user explicitly groups issues:

1. Pick one item from the priority list in `docs/known_failures.md`.
2. Implement the smallest provider-independent fix.
3. Validate with the provider explicitly authorized for the task: one n=3 run is mandatory when live validation is requested, followed by the requested size/topic spread.
4. Read every relevant `transcript.md` and `run.json`; metrics alone are insufficient.
5. Update `docs/known_failures.md` with the result and any stable new failure pattern.
6. Audit and synchronize `AGENTS.md`, this file, repository skills, active memory/index files, `README.md`, and any other affected information files before completing the upgrade.
7. Do not start the next issue until this upgrade is fully verified; stop unless the user explicitly requested automatic continuation.

### Evaluation files

- `evals/topics.txt` — ~40 diverse topics for random validation runs (everyday, travel, academic, work, creative, hypothetical).
- `docs/known_failures.md` — single tracking file for open issues, priorities, acceptance criteria, and implementation protocol.

## Configuration quick reference

The ~12 knobs worth touching are listed in `config.yaml` under the "DIALS THAT MATTER" header. Everything below that header is structural constants set once.
