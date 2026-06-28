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

Provider is set in `config.yaml` → `llm.provider`: `uni` (Bamberg Ollama endpoint, requires VPN), `groq`, or `gemini`. API keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`) come from `.env`. There is no offline/mock mode — a run requires a reachable provider and raises on failure.

## Architecture

The split is deliberate and load-bearing:

- **`config.yaml`** — every tunable number lives here, nowhere else.
- **`src/prompts.py`** — every piece of prose sent to an LLM or printed as moderator text lives here, nowhere else.
- **`src/models.py`** — typed state objects (`dataclass(slots=True)`). Routing, consensus, logging, and validation operate on these, not raw dicts.
- **`src/validation.py`** — deterministic guardrails (detection + logging). Style checks are warn-level diagnostics; only structural errors trigger LLM repair. Also houses discourse-frame classification and claim-slot tracking used by prompts to nudge variety.

### Run flow

1. **`builders.py`** — two sequential LLM calls: first generates the option cards (with `shared_context` and `short_name` per option), second generates per-persona hidden belief state (preferred/acceptable/rejected, utility scores, backstory) given those options. Traits and names are sampled in code from a diverse pool and passed to the model. Invalid worlds raise, never silently default. `_clean_short_name` validates LLM-provided short names (rejects if ending on a stopword or >3 words). After persona generation, `_validate_preference_plan()` checks that same-camp participants share a preferred option and different camps chose different options; a violated camp structure retries setup (up to `setup_generation_attempts`) rather than mutating personas. Safe postprocessing ensures `preferred_option` is always in `acceptable_options`, adds a common compromise to non-stubborn participants, and guarantees a minimum of `non_blocker_min_acceptable` acceptable options per non-stubborn persona.
2. **`router.py`** — emits a `MoveIntent` each turn. Priority: (1) answer pending question, (2) respond to unanswered challenge via `_unanswered_challenge`, (3) fill coverage gaps, (4) weighted speaker selection. Drives phases: opening → discussion → narrowing → confirmation → closure.
3. **`dialogue.py`** — orchestration: calls the router, renders the turn via `prompts.sim_utterance`, parses the trailer, updates state, checks consensus. Contains `Orchestrator`, `DialogueController`, `StateTracker`, `ConsensusManager`.
4. **`parsing.py`** — extracts the machine trailer `[act=…; opt=…; stance=…]` from each generated turn, resolves option references.
5. **`scoring.py`** — shared `current_lean` / `leading_option` used by routing, consensus, and moderator prompts.
6. **`validation.py`** — deterministic checks. Structural errors (missing trailer, invented option, malformed vote) trigger repair. Style issues (robotic phrasing, self-narration, possessive openers, card-reading, repeated starts) are logged as warnings only. Discourse-frame classification (`classify_discourse_frames`) and claim-slot tracking (`classify_claim_slots`) provide variety hints to the prompt layer.
7. **`llm_client.py`** — thin provider abstraction (uni/groq/gemini). Each turn is a stateless call; the full prompt is re-sent every time (no session memory on the endpoint side).
8. **`logger.py`** — writes `transcript.md`, `run.json`, `metrics.csv`, and optional `prompts.jsonl` per run under `logs/<run_id>/`.

### Prompt design

The per-turn prompt uses a compact `runtime_speaker_card` (not the full persona profile). It includes: current lean, one concern, one speaking habit derived from traits, last 2 prior claims, concession state, and discourse-frame/claim-slot hints when repetition is detected. A `_responding_to_line` anchors each turn to the most relevant prior turn. Full option cards are only rendered for COMPARE/VOTE acts; all other acts get option names only. The prompt has 3 rules focused on voice, leading with own thought (not recapping), and no stock phrases. After the opening, an alias instruction tells the model to use shorthand for already-discussed options; aliases come from `OptionCard.short_name` (LLM-generated at setup, e.g. "Ticket to Ride" → "Ride", "Settlers of Catan" → "Catan") with a deterministic fallback for missing/invalid short names.

Face-work modifiers are added to guidance for objection/push-back acts based on persona traits (agreeable speakers soften, neurotic speakers show anxiety, direct speakers skip diplomacy).

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
- **Backchannel injection**: 5% chance per discussion turn that the router injects a `short_react=True` REACT intent (skips act sampling entirely). Guard: previous turn must be ≥10 words and not itself a backchannel. Validation enforces ≤8-word limit via `BACKCHANNEL_TOO_LONG` repair code.
- **Surface cleanup**: deterministic removal of space-before-punctuation, repeated punctuation, stray quotes in `clean_generated`.
- **No example seeding in guidance**: guidance strings must describe desired behavior, never demonstrate it with quoted phrases (e.g., never `"Try: 'What if we...'"`) — the model copies examples verbatim across all topics.
- **Epistemic grounding rule**: the per-turn prompt tells the model "you know only what's in the option cards — anything else is unknown: say 'I'm not sure' or 'we'd need to check'". This prevents confident invented facts about real-world named places (seating, pricing, amenities not in the cards).

## Outputs

Each run writes to `logs/<run_id>/`:
- `transcript.md` — human-readable chat + outcome + metrics
- `run.json` — full structured run (per-turn acts, validation issues, `repair_trigger_codes` on repaired turns)
- `logs/metrics.csv` — master file, one row appended per run

Key metrics: `outcome_status`, `final_support_fraction`, `repaired_turns`, `flagged_turns`, `question_density`, `avg_words_per_turn`, `option_coverage`.

## Testing and evaluation

```powershell
# Unit tests (offline, instant — run after every code change)
& .\ds_proj\Scripts\python.exe -m pytest tests/ -v

# Post-run regression checks on existing logs
& .\ds_proj\Scripts\python.exe evals\run_eval.py --check-latest 4

# Drive new eval runs from the scenario spread (requires VPN for uni)
& .\ds_proj\Scripts\python.exe evals\run_eval.py --run
```

### Implementation process

Every fix follows this cycle (see `docs/known_failures.md` for the full protocol):

1. Pick one item from the priority list in `docs/known_failures.md`.
2. Implement the fix. Run `pytest tests/`.
3. Validate: one n=3 run (mandatory), then 5–6 additional runs across n=2–7 with random topics from `evals/topics.txt`. Always use the `uni` provider.
4. Read the transcripts. Check the fix works and nothing regressed.
5. If new issues surface, add them to `docs/known_failures.md` before continuing.
6. Only move to the next item when all runs are reviewed.

### Test and eval files

- `tests/test_validation.py` — covers all deterministic guardrails in `validation.py` (essential cases for each check, plus discourse-frame and claim-slot classification).
- `tests/test_consensus.py` — covers consensus support fraction, outcome state consistency.
- `tests/test_parsing.py` — covers trailer extraction, commitment gating, hedge detection, option resolution.
- `evals/run_eval.py` — reads `run.json` files and checks for regressions (same-speaker back-to-back, question density, opener variety, mid-discussion accepts, duplicate moderator lines, robotic templates, outcome sanity) plus interaction quality metrics (named rate, responsive rate, self-repetition, echoed phrases).
- `evals/topics.txt` — ~40 diverse topics for random validation runs (everyday, travel, academic, work, creative, hypothetical).
- `evals/scenarios.yaml` — the topic/size spread used for batch evaluation.
- `docs/known_failures.md` — single tracking file for open issues, fix priorities, and implementation protocol.
- `docs/evaluation.md` — full evaluation workflow reference.

## Configuration quick reference

The ~12 knobs worth touching are listed in `config.yaml` under the "DIALS THAT MATTER" header. Everything below that header is structural constants set once.
