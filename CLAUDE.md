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
- **`src/validation.py`** — deterministic guardrails only, not a second policy engine.

### Run flow

1. **`builders.py`** — two sequential LLM calls: first generates the option cards, second generates per-persona hidden belief state (preferred/acceptable/rejected, utility scores, backstory) given those options. Traits are sampled in code and passed to the model. Invalid worlds raise, never silently default.
2. **`router.py`** — emits a `MoveIntent` each turn (speaker, addressee, act type, length hint). Drives phases: opening → discussion → narrowing → confirmation → closure.
3. **`dialogue.py`** — orchestration: calls the router, renders the turn via `prompts.sim_utterance`, parses the trailer, updates state, checks consensus. Contains `Orchestrator`, `DialogueController`, `StateTracker`, `ConsensusManager`.
4. **`parsing.py`** — extracts the machine trailer `[act=…; opt=…; stance=…]` from each generated turn, resolves option references.
5. **`scoring.py`** — shared `current_lean` / `leading_option` used by routing, consensus, and moderator prompts.
6. **`validation.py`** — deterministic checks + repairs (empty turns, speaker prefix, echo guard, robotic phrasing, question chains, invented numbers, opener variety, card-reading, self-narration, collective voice). Warn-level flags are logged but not repaired by default.
7. **`llm_client.py`** — thin provider abstraction (uni/groq/gemini). Each turn is a stateless call; the full prompt is re-sent every time (no session memory on the endpoint side).
8. **`logger.py`** — writes `transcript.md`, `run.json`, `metrics.csv`, and optional `prompts.jsonl` per run under `logs/<run_id>/`.

### Key design constraints

- **Topic-agnostic**: nothing is tuned for a specific scenario. Any fix must work for arbitrary topics and group sizes 2–7.
- **No fabricated fallbacks**: if setup or a turn call returns something unusable, the run raises rather than papering over it with defaults.
- **Commitment gating**: `accept`/`vote`/`reject` only count as binding on routed decision turns (narrowing/confirmation), not during free discussion. This prevents premature fake-unanimous outcomes.
- **Hard blockers are immovable**: a hard-blocker persona only ever backs their preferred option; any vote/accept elsewhere is ignored.
- **Pacing is derived, not fixed**: `min_discussion_turns`, `force_narrow_turns`, `hard_max_turns` are computed per run from group size and composition.
- **Natural dialogue, cooperative by default**: all personas strive to find a compromise. No persona actively disrupts the discussion. However, trait-driven stubbornness (controlled by `personas.hard_blocker_probability` in `config.yaml`) can make a persona genuinely immovable, leading to an unresolved outcome — this is realistic and intended, but should occur in roughly 5–10% of runs, not more.

## Active refactor: dialogue quality

The simulator produces valid decision transcripts, but the dialogue is too repetitive and mechanical. The full plan lives in `docs/dialogue_quality_refactor_plan.md`; open/fixed issues are tracked in `docs/known_failures.md`. The refactor targets these root causes:

1. **Repair layer burns LLM calls without fixing the root cause** — 30–50% of turns trigger repair, roughly doubling generation time, yet flagged patterns still appear. Style checks (robotic phrasing, self-narration, repeated opener, possessive subject, card-reading, awkward closing) should become logging-only diagnostics. Keep LLM repair only for structural errors: missing/invalid trailer, invented option, malformed vote, multi-speaker output, hard-blocker violation, invented numeric facts.

2. **Per-turn prompt is too large** — every turn dumps the full persona profile (traits, scores, backstory, all reasons, all options). Replace with a compact `runtime_speaker_card()` that includes only: current lean, one active concern, one speaking habit, one prior claim, addressee/interaction cue, and concession state if relevant. Keep full persona in state for consistency; stop rendering it all every turn.

3. **Router is phase-first, not adjacency-first** — turns don't respond to each other because the router picks speakers by phase progress and turn balance. New priority: (1) answer pending direct question, (2) respond to challenge, (3) relevant self-selection, (4) invite quiet participant only if relevant, (5) phase/coverage as fallback. Add `InteractionObligation` to state and `source_turn_id` + `routing_reason` to `MoveIntent`.

4. **Turns are board-responses, not turn-responses** — each turn answers the option board instead of reacting to a specific previous point. The prompt should say "respond to X's point: '...'" and provide only the relevant option facts, not the full board every time. Full option cards only for compare/vote turns.

5. **Stance changes are unmotivated** — participants accept compromises without visible concession bridges. Before allowing `current_preference` to change, require a visible triggering argument from another participant. Prompt must show: old preference + reason → specific prior argument → new position.

6. **Closure ignores outcome type** — consensus/fallback/unresolved all use the same closing language. Consensus: brief agreement. Fallback: name the majority pick but preserve dissent. Unresolved: state the deadlock directly.

7. **Moderator is generic** — should diagnose the actual split, ask targeted holdouts, propose test criteria, not just "can we all get behind X."

### Implementation order

Follow the sequence in `docs/dialogue_quality_refactor_plan.md` § "Suggested order of work." Each step should be tested with `evals/run_eval.py` and compared against existing logs before moving to the next.

### What NOT to do

- Do not remove the option board — it provides controlled candidates and enables evaluation.
- Do not remove persona depth — traits, goals, backstory are useful, just don't dump them all into every prompt.
- Do not add more research-paper abstractions or new modules for their own sake.
- Do not add more repair checks — fix routing and prompting instead.
- Do not make the moderator responsible for solving every deadlock.
- Do not make prompts longer — the underlying model (llama3.3) ignores excess rules. Every new rule added to `sim_utterance` should come with an old one being cut or merged.

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

# Drive new eval runs from the scenario spread (requires VPN)
& .\dspro\Scripts\python.exe evals\run_eval.py --run
```

- `tests/test_validation.py` — covers all deterministic guardrails in `validation.py`.
- `tests/test_parsing.py` — covers trailer extraction, commitment gating, hedge detection, option resolution.
- `evals/run_eval.py` — reads `run.json` files and checks for regressions (same-speaker back-to-back, question density, opener variety, hard-blocker integrity, mid-discussion accepts, duplicate moderator lines, robotic templates, outcome sanity).
- `evals/scenarios.yaml` — the topic/size spread used for batch evaluation.
- `docs/known_failures.md` — tracked failures with root cause, fix status, and regression signals.
- `docs/evaluation.md` — full evaluation workflow reference.

## Configuration quick reference

The ~12 knobs worth touching are listed in `config.yaml` under the "DIALS THAT MATTER" header. Everything below that header is structural constants set once.