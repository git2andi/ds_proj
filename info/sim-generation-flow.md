# Simulation Generation Flow

This document traces one complete run from `py main.py` to the final log files.

---

## 1. Entry point — `main.py`

The user types a topic (or feeds a batch file). `main.py` calls:

```python
Orchestrator(topic).run()
```

Everything that follows happens inside that call. The CLI only collects the topic and prints the closing summary.

---

## 2. Setup — `builders.py :: SetupBuilder.build(n)`

Before any dialogue, the controller builds the world. This is the only part where two LLM calls happen.

### 2a. Controller samples traits (no LLM)

`_trait_rows(n)` picks random names from the name pool and samples five OCEAN trait integers (1–5) for each persona from the ranges in `config.yaml`. One persona may be a "hard blocker" (agreeableness=1) based on `hard_blocker_probability`.

`_preference_shape(n)` samples an exact primary-preference partition from `personas.preference_distribution.shape_weights[n]`. A configured `forced_shape` bypasses sampling for controlled runs. The shape is validated before any provider call; after options exist, `_preference_assignments()` maps each shape part to a distinct concrete option and each participant to one required primary.

### 2b. LLM call #1 — scenario and option cards

`_generate_scenario()` sends `prompts.setup_scenario(topic, n)` to the LLM and expects JSON:

```
scenario:
  decision_kind        – category label (e.g. travel_destination)
  opening_question     – one casual question for the opening round
  shared_context       – 2–3 stable background facts everyone would know
  options:
    id, name, short_name  – option label (A/B/C/D), specific name, casual nickname
    attrs                 – 2–4 stable topic-specific attributes with concrete values
    upside, tradeoff      – best case and main cost
    concern, best_for     – standing objection and which priority it serves
```

The parser validates: correct number of options, correct ids, required attribute count, all required fields present. Any failure triggers a retry; if all attempts fail the run raises.

### 2c. LLM call #2 — persona belief states

`_generate_personas()` sends `prompts.setup_personas(...)` with the trait rows and a concrete required primary option for every participant. The LLM must return one participant per slot, each matching the given id and trait values, with:

```
name                     – short first name
background               – one sentence: a relevant prior experience, habit, or context
private_goal             – what they personally want from this decision
preferred_options        – list of 1–2 favourites (ordered); first item must equal the assigned primary
rejection                – optional; at most one option they refuse (must not be a preferred option)
rejection_reason         – why (only if rejection is set)
```

The builder validates each primary against its row-local assignment, plus valid rejections, one preference for stubborn personas (agreeableness=1), and unique names. Persona failure retries only the persona stage and preserves the validated scenario. Contradictions raise immediately — no silent mutation.

### 2d. Validation (deterministic, no LLM)

`_parse_scenario()` runs `_validate_participant_references(shared_context, n)` inline: if the LLM wrote an explicit group-size claim (e.g. "a group of four friends") in `shared_context` that contradicts the configured `n`, parsing raises immediately and the scenario retry loop re-tries before raising.

`_validate_world()` does a final sanity check across the complete world (option IDs, unique names, valid preferences, hard-blocker preference count) and raises if anything is inconsistent. There is no post-processing step: the minimal schema makes structural guarantees unnecessary.

---

## 3. State initialization — `dialogue.py :: initialise_state()`

Once the world is valid, `initialise_state(scenario, personas)` creates:

- **`DialogueState`** — the single mutable object passed everywhere. Contains the scenario, personas, current phase, turn list, per-persona runtimes, per-option coverage, open questions, candidate option, and outcome.
- **`ParticipantRuntime`** per persona — turn count, last-spoke turn, current preference (starts at `preferred_options[0]`), explicit votes, accepted/rejected options, hard rejections (seeded from `persona.rejection`), and recent utterances.
- **`OptionCoverage`** per option — mention count, reason count, objection count, acceptance count, and covered claim slots.
- **Pacing targets** from `derive_pacing(personas)` — `min_discussion_turns`, `force_narrow_turns`, and `hard_max_turns` derived from group size, preference diversity, average compromise willingness, and average deliberativeness. Jitter ensures no two similar groups run exactly the same length.

---

## 4. Dialogue loop — `dialogue.py :: Orchestrator.run()`

### 4a. Opening frame

The moderator prints a fixed option board (names + attributes, from `prompts.moderator_opening()`).

An optional **social greeting beat** (`_social_round`) selects at most the most extraverted persona, probability-gated by extraversion, for one brief line from `prompts.greeting_line()`. Cosmetic only — it does not affect stance or coverage state.

### 4b. Main turn loop

Each iteration:

1. **Phase update** — `DialogueController.update_phase()` advances through phases:
   - OPENING → DISCUSSION once everyone has spoken once.
   - DISCUSSION → NARROWING when `_can_start_narrowing()` is true: min turns reached, coverage threshold met, readiness (concentration) score high enough, or cap hit.
   - On natural convergence, the router first emits one `PROPOSE_COMPROMISE` turn (`narrowing_called = True`) so a participant — not the system — calls for the vote.
   - NARROWING → CONFIRMATION after all participants cast an explicit vote.
   - CONFIRMATION → CLOSURE after `max_confirmation_turns` of accept/reject without unanimity, or if a question is outstanding and unanswered.
   - Any phase → CLOSURE when `hard_max_turns` is reached.

2. **Consensus check** — `ConsensusManager.detect()` returns a `RunOutcome` if every participant has explicitly voted or accepted the same option. This can short-circuit CONFIRMATION early.

3. **Moderator intervention** (rate-limited, optional) — `_moderator_intervention()` may fire a stall nudge or address 1–2 pending participants in CONFIRMATION. Transcript evidence classifies each as an actual alternative/objection holder or as merely missing an explicit commitment, so aligned participants receive confirmation language rather than resistance language.

4. **Intent routing** — `TurnRouter.next_intent()` decides who speaks and with what act:
   - Priority 1: answer the oldest pending open question.
   - Priority 2: respond to an unanswered OBJECT/PUSH_BACK (`_unanswered_challenge`).
   - Priority 3: fill a coverage gap for options nobody has mentioned yet.
   - Priority 4: weighted speaker selection + act sampling (trait-modulated probabilities from `config.yaml::routing.act_probabilities`).
   - In NARROWING: route unvoted participants to VOTE.
   - In CONFIRMATION: route pending participants to ACCEPT or REJECT based on trait-driven willingness (agreeableness ≠ 1 can accept; hard blockers cannot accept a non-preferred option).
   - Returns a `MoveIntent`: speaker_id, act, option_focus, length_hint, addressee_id, respond_to_turn, moves_lean.

5. **Turn generation** — `_generate_turn()`:
   - Builds prompt via `prompts.sim_utterance()`. OPENING includes background and goal. Targeted discussion replies instead include the exact recent message to answer, omit biography, and receive one act-specific OCEAN behavior cue. Untargeted turns get only a compact personal stake. Focused option facts and recent chat remain bounded.
   - LLM generates raw text.
   - `clean_generated()` normalises whitespace, strips the speaker prefix, removes the `Considering X,` opener, and trims to the word cap.
   - `strip_possessive_opener()` removes any `<OptName>'s` leading phrase.
   - `MessageValidator.validate()` checks for structural errors (missing trailer, invented option, wrong-target vote) and style warnings (robotic phrasing, self-narration, repeated opener).
   - If structural errors: up to `max_repairs_per_turn` repair calls using `prompts.repair_utterance()`, each re-running validation. Remaining issues are logged; no fabricated fallback.
   - Returns: final text, parsed `TurnMove`, full prompt, token counts, issue codes, repaired flag, trigger codes.

6. **State update** — `StateTracker.apply_participant()`:
   - Appends `TurnRecord` to `state.turns`.
   - `_update_runtime()`: increments turn count, records preference changes (vote/accept/propose changes lean), tracks accepted/rejected options.
   - `_update_coverage()`: records visibly named options, claim-slot-backed reasons, objections, acceptances, and covered claim slots.
   - `_update_questions()`: registers new open questions; clears answered ones (allows one hedge before clearing).
   - `_update_progress()`: compares a snapshot of stances + coverage; resets or increments `no_progress_count`.

### 4c. Closure frame

Once the loop exits, `ConsensusManager.finalize()` determines the outcome:
- `successful` — unanimous explicit votes/accepts on one option.
- `majority` — whichever option has the most visible support reached `majority_fallback_fraction`; the controller candidate cannot override the visible tally.
- `unresolved` — neither threshold met.

The moderator says a closure line from visible supporter/non-supporter state. An optional **farewell social beat** may add one brief line from the most extraverted persona; a majority non-supporter must not imply they accepted the result.

---

## 5. Logging — `logger.py`

`DialogueLogger.finish()` writes all output to `logs/<run_id>/`:

| File | Contents |
|------|----------|
| `transcript.md` | Human-readable chat with speaker labels, outcome, and per-turn metrics |
| `run.json` | Full structured data: all turns (text, act, tokens, validation issues, repair trigger codes), personas, scenario, outcome |
| `logs/metrics.csv` | One row appended: outcome_status, support fraction, repaired turns, flagged turns, token totals, question density, avg words/turn, option coverage |

---

## Data flow summary

```
topic string
  └─ SetupBuilder.build(n)
       ├─ [no LLM] trait sampling + preference shape
       ├─ [LLM #1] scenario + option cards → Scenario, [OptionCard]
       ├─ [LLM #2] persona belief states → [Persona]
       └─ postprocess + validate → raises on failure
  └─ initialise_state() → DialogueState
  └─ Orchestrator main loop (per turn):
       ├─ phase update
       ├─ consensus check
       ├─ moderator intervention (optional LLM)
       ├─ TurnRouter → MoveIntent
       ├─ [LLM] generate turn (+ optional repair LLM calls)
       └─ StateTracker → update runtimes, coverage, questions, progress
  └─ ConsensusManager.finalize() → RunOutcome
  └─ moderator closure (LLM) + farewell social round (LLM per speaker)
  └─ DialogueLogger.finish() → transcript.md, run.json, metrics.csv
```

Each participant turn uses one stateless LLM call. The full prompt is re-sent every time — there is no session memory on the endpoint side.
