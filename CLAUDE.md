# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university project: an LLM-driven group discussion simulator. Given a topic, it generates a multi-party chat where 2–7 personas debate and try to reach a decision. A deterministic controller (routing, state tracking, consensus) decides **who speaks, when, and with what intent**; the LLM only renders the surface text of each turn via a single stateless call per turn.

## Running

Activate the `dspro` virtualenv. No requirements.txt install step — the venv is pre-built.

```powershell
# Interactive single run
(dspro) PS> py .\main.py
Topic: Plan a weekend team offsite

# Batch (one topic per line, # comments ignored)
(dspro) PS> py .\main.py scenarios.txt

# Headless (pipe a topic, using the venv python directly)
"Example Topic" | & .\dspro\Scripts\python.exe .\main.py
```

Provider is set in `config.yaml` → `llm.provider`: `uni` (Bamberg Ollama endpoint, requires VPN), `groq`, or `gemini`. API keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`) come from `.env`. There is no offline/mock mode — a run requires a reachable provider and raises on failure.

## Architecture

The split is deliberate and load-bearing:

- **`config.yaml`** — every tunable number lives here, nowhere else.
- **`src/prompts.py`** — every piece of prose sent to an LLM or printed as moderator text lives here, nowhere else.
- **`src/models.py`** — typed state objects (`dataclass(slots=True)`). Routing, consensus, logging, and validation operate on these, not raw dicts.
- **`src/validation.py`** — deterministic guardrails (detection + logging). Style checks are warn-level diagnostics; only structural errors trigger LLM repair. Also houses discourse-frame classification and claim-slot tracking used by prompts to nudge variety.

### Run flow

1. **`builders.py`** — two sequential LLM calls: first generates the option cards, second generates per-persona hidden belief state (preferred/acceptable/rejected, utility scores, backstory) given those options. Traits are sampled in code and passed to the model. Invalid worlds raise, never silently default.
2. **`router.py`** — emits a `MoveIntent` each turn. Priority: (1) answer pending question, (2) respond to unanswered challenge via `_unanswered_challenge`, (3) fill coverage gaps, (4) weighted speaker selection. Drives phases: opening → discussion → narrowing → confirmation → closure.
3. **`dialogue.py`** — orchestration: calls the router, renders the turn via `prompts.sim_utterance`, parses the trailer, updates state, checks consensus. Contains `Orchestrator`, `DialogueController`, `StateTracker`, `ConsensusManager`.
4. **`parsing.py`** — extracts the machine trailer `[act=…; opt=…; stance=…]` from each generated turn, resolves option references.
5. **`scoring.py`** — shared `current_lean` / `leading_option` used by routing, consensus, and moderator prompts.
6. **`validation.py`** — deterministic checks. Structural errors (missing trailer, invented option, malformed vote) trigger repair. Style issues (robotic phrasing, self-narration, possessive openers, card-reading, repeated starts) are logged as warnings only. Discourse-frame classification (`classify_discourse_frames`) and claim-slot tracking (`classify_claim_slots`) provide variety hints to the prompt layer.
7. **`llm_client.py`** — thin provider abstraction (uni/groq/gemini). Each turn is a stateless call; the full prompt is re-sent every time (no session memory on the endpoint side).
8. **`logger.py`** — writes `transcript.md`, `run.json`, `metrics.csv`, and optional `prompts.jsonl` per run under `logs/<run_id>/`.

### Prompt design

The per-turn prompt uses a compact `runtime_speaker_card` (not the full persona profile). It includes: current lean, one concern, one speaking habit derived from traits, last 2 prior claims, concession state, and discourse-frame/claim-slot hints when repetition is detected. A `_responding_to_line` anchors each turn to the most relevant prior turn. Full option cards are only rendered for COMPARE/VOTE acts; all other acts get option names only. The prompt has 4 rules focused on voice, reactivity, originality, and grounding.

### Key design constraints

- **Topic-agnostic**: nothing is tuned for a specific scenario. Any fix must work for arbitrary topics and group sizes 2–7.
- **No fabricated fallbacks**: if setup or a turn call returns something unusable, the run raises rather than papering over it with defaults.
- **Commitment gating**: `accept`/`vote`/`reject` only count as binding on routed decision turns (narrowing/confirmation), not during free discussion.
- **Hard blockers are immovable**: a hard-blocker persona only ever backs their preferred option; any vote/accept elsewhere is ignored.
- **Pacing is derived, not fixed**: `min_discussion_turns`, `force_narrow_turns`, `hard_max_turns` are computed per run from group size and composition.
- **Concession bridges**: when a speaker accepts/votes for a non-preferred option, persona-specific bridge guidance (residual worry, condition, trade-off, or next step) fires before chorus detection.
- **Prompts stay short**: llama3.3 ignores excess rules. Every new rule added to `sim_utterance` should come with an old one being cut or merged.

## Outputs

Each run writes to `logs/<run_id>/`:
- `transcript.md` — human-readable chat + outcome + metrics
- `run.json` — full structured run (per-turn acts, validation issues)
- `logs/metrics.csv` — master file, one row appended per run

Key metrics: `outcome_status`, `final_support_fraction`, `repaired_turns`, `flagged_turns`, `question_density`, `avg_words_per_turn`, `option_coverage`.

## Testing and evaluation

```powershell
# Unit tests (offline, instant — run after every code change)
& .\dspro\Scripts\python.exe -m pytest tests/ -v

# Post-run regression checks on existing logs
& .\dspro\Scripts\python.exe evals\run_eval.py --check-latest 4

# Drive new eval runs from the scenario spread (requires VPN for uni)
& .\dspro\Scripts\python.exe evals\run_eval.py --run
```

- `tests/test_validation.py` — covers all deterministic guardrails in `validation.py` (147 tests, including discourse-frame and claim-slot classification).
- `tests/test_consensus.py` — covers consensus support fraction, outcome state consistency.
- `tests/test_parsing.py` — covers trailer extraction, commitment gating, hedge detection, option resolution.
- `evals/run_eval.py` — reads `run.json` files and checks for regressions (same-speaker back-to-back, question density, opener variety, hard-blocker integrity, mid-discussion accepts, duplicate moderator lines, robotic templates, outcome sanity) plus interaction quality metrics (named rate, responsive rate, self-repetition, echoed phrases).
- `evals/scenarios.yaml` — the topic/size spread used for batch evaluation.
- `docs/known_failures.md` — tracked failures with fix status and regression signals.
- `docs/dialogue_quality_refactor_plan.md` — completed refactor plan with results.
- `docs/evaluation.md` — full evaluation workflow reference.

## Configuration quick reference

The ~12 knobs worth touching are listed in `config.yaml` under the "DIALS THAT MATTER" header. Everything below that header is structural constants set once.
