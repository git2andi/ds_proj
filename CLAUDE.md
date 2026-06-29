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
- **`src/aliases.py`** — one shared alias-validation contract used by setup, prompts, validation, and parsing. `validated_short_alias` checks the LLM-proposed nickname; `deterministic_alias` provides a fallback; `short_alias_map` resolves the complete set and eliminates collisions.
- **`src/validation.py`** — deterministic guardrails (detection + logging). Style checks are warn-level diagnostics; only structural errors trigger LLM repair. Also houses discourse-frame classification and claim-slot tracking used by prompts to nudge variety. Moderator and farewell turns run a lightweight `invalid_option_refs` check in `dialogue.py` with one retry before the text enters the transcript.

### Run flow

1. **`builders.py`** — two sequential LLM calls: first generates option cards (with `shared_context` and `short_name`), then per-persona belief state (minimal: `name`, `background`, `private_goal`, `preferred_options` list of 1–2, optional `rejection`/`rejection_reason`). The controller samples only the five OCEAN traits and assigns preference camps before the LLM call; LLM never generates scores, acceptable lists, or reasons. Response length and compromise willingness are derived deterministically from traits; stubborn profiles come from separate five-trait ranges when `hard_blocker_probability` activates.
2. **`router.py`** — emits a `MoveIntent` each turn. Priority: (1) answer pending question, (2) respond to unanswered challenge via `_unanswered_challenge`, (3) fill coverage gaps, (4) weighted speaker selection. Drives phases: opening → discussion → narrowing → confirmation → closure.
3. **`dialogue.py`** — orchestration: calls the router, renders the turn via `prompts.sim_utterance`, parses the trailer, updates state, checks consensus. Contains `Orchestrator`, `DialogueController`, `StateTracker`, `ConsensusManager`.
4. **`parsing.py`** — extracts the machine trailer `[act=…; opt=…; stance=…]` from each generated turn, resolves option references.
5. **`scoring.py`** — shared `current_lean` / `leading_option` used by routing, consensus, and moderator prompts.
6. **`validation.py`** — deterministic checks. Structural errors (missing trailer, invented option, malformed vote) trigger repair. Style issues (robotic phrasing, self-narration, card-reading, repeated starts) are logged as warnings only. Possessive openers (`strip_possessive_opener`) and `Considering X,` openers (`_strip_considering_opener`) are stripped deterministically before validation so they never reach the transcript. Discourse-frame classification (`classify_discourse_frames`) and claim-slot tracking (`classify_claim_slots`) provide variety hints to the prompt layer.
7. **`llm_client.py`** — thin provider abstraction (uni/groq/gemini/gpt). Each turn is a stateless call; the full prompt is re-sent every time (no session memory on the endpoint side).
8. **`logger.py`** — writes `transcript.md`, `run.json`, `metrics.csv`, and optional `prompts.jsonl` per run under `logs/<run_id>/`.

### Prompt design

The per-turn prompt uses a compact `runtime_speaker_card` that includes the persona's background, private goal, current lean, behavior cues derived from five traits, and a "Voice:" register note for personas with strongly expressed traits (blunt/warm/cautious/firm/very short). Generated role and speech-style labels are deliberately omitted. Each turn has one local job; when responding to a specific message that job block shows the exact message to address first, placed before the option facts. Full option cards are rendered for OPENING/COMPARE/VOTE; ASK acts get a compact attribute brief (`_option_brief`); other acts get option names only. Option name variety is encouraged — the prompt shows full-name → shortening pairs; the resolver handles both forms.

Face-work modifiers describe behavior for objection/push-back acts: agreeable speakers acknowledge the other side, cautious speakers surface risk, and direct speakers put the problem early. Response length uses "Hard limit: N words" framing co-located with "contractions OK, no semicolons, no formal transitions" in every `_verbosity_note` level so both are enforced together.

The target register is a chat among friends: "write like you'd text a friend, not like you're presenting." Background and goal give each persona a specific personal angle. Greeting prompts constrained to 2-6 words; farewell instruction says "Don't open with the option name — lead with your reaction"; tone is differentiated by outcome (got pick / came around / resigned) and persona background. Moderator agreement and closure prompts are explicitly constrained to declarative statements — no questions.

Deterministic surface cleanup (`dialogue.py`): semicolons in the message body replaced with em-dashes via `_strip_body_semicolons()` (trailer-aware, protects `[act=...; opt=...; stance=...]`). Accidental `A=`/`D=` prefixes stripped in `_surface_cleanup`.

### Key design constraints

- **No fabricated fallbacks**: if setup or a turn call returns something unusable, the run raises rather than papering over it with defaults.
- **Commitment gating**: `accept`/`vote`/`reject` only count as binding on routed decision turns (narrowing/confirmation), not during free discussion. Hedged accepts ("still not sure", "not fully sold") are clamped to neutral.
- **Trait-driven stubbornness**: `hard_blocker_probability` can select at most one participant whose five-trait profile uses agreeableness 1, low openness, and a firmer conscientiousness range. Everyone else uses agreeableness 3–5. Compromise willingness is derived from the five traits rather than sampled independently. No mechanical vote override — behaviour emerges from traits. If the LLM's trailer names a different option on a VOTE/ACCEPT turn, `HARD_BLOCKER_WRONG_VOTE` repair fires.
- **Outcome taxonomy**: `successful` = every participant visibly voted/accepted the same option; `majority` = one option has ≥66% visible support; `unresolved` = no option met either threshold. Only explicit `explicit_vote` and `accepted_options` entries count — hidden preferences and routing leans are excluded.
- **Pacing is derived, not fixed**: `min_discussion_turns`, `force_narrow_turns`, `hard_max_turns` are computed per run from group size and composition.
- **Concession bridges**: when a speaker accepts/votes for a non-preferred option, persona-specific bridge guidance fires.
- **Prompts stay short**: llama3.3 ignores excess rules. Every new rule added to `sim_utterance` should come with an old one being cut or merged.
- **Contribution-based routing**: for n>=4, speakers are skipped when their only available move is restating a known preference. Extraversion and recent participation drive turn frequency; no separately generated initiative/directness/detail controls exist.
- **Stall-to-concrete routing**: when discussion stalls (2+ turns no progress), ASK probability doubles and SUPPORT dampens, steering toward concrete questions instead of preference restating.
- **Surface cleanup**: deterministic removal of space-before-punctuation, repeated punctuation, stray quotes in `clean_generated`; possessive openers (`OptName's`) and `Considering X,` openers stripped; semicolons in the body replaced with em-dashes (`_strip_body_semicolons`, trailer-aware); accidental `A=`/`D=` option-letter prefixes stripped.
- **No example seeding in guidance**: guidance strings must describe desired behavior, never demonstrate it with quoted phrases (e.g., never `"Try: 'What if we...'"`) — the model copies examples verbatim across all topics.
- **Epistemic grounding rule**: the per-turn prompt explicitly forbids claiming facilities, features, or services not listed in the card, and bans turning uncertainty into a new invented fact. `_INVENTED_FACILITY_PATTERN` in `validation.py` catches confident facility-noun inventions (lodges, shuttles, indoor spaces, group discounts, etc.) and fires `INVENTED_OPTION_ATTRIBUTE` (repair) when absent from the card.
- **Coverage gating**: `_update_coverage` in `dialogue.py` counts only options visibly named in text (via `ids_in_text`), not those inferred from routing intent. A reason is counted only when a claim slot is also present — prevents inflated reason counts from routine mentions.
- **ASK grounding**: ASK turns receive a compact card brief (`_option_brief`) for the focus options and explicit guidance to ask about a specific attribute, number, or trade-off listed in the card. Unanswerable questions close after a second hedge answer.

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
