# CLAUDE.md

## Project Overview

This project is a university dialogue-simulation system. It creates multi-party
group chats in which several simulated participants ("Sims") deliberate over a
random decision topic and try to reach a workable group outcome.

The design goal is **a real deliberation, not a roll-call to a decision.**
The decision should emerge from the discussion; if the discussion is hollow,
the decision is meaningless. Concretely the system aims for:

- participants have stable preferences but argue from reasons, not by repeating
  a position;
- disagreement is visible, grounded in each person's priorities and experience,
  and resolved by exchange rather than declared by the moderator;
- questions are answered before the group moves on; pushback gets engaged with;
- message length and structure vary naturally — mostly short chat-like turns,
  occasionally longer when a point genuinely needs explanation;
- each participant sounds like a distinct person, not a copy of the same polite
  evaluator.

## Research grounding (three layers)

The architecture is organised by *layer of behaviour*, not "one paper per
module". Each cited source maps to a specific, load-bearing piece of code.

### Layer A — Conversation mechanics

| Source | Module | Role |
|---|---|---|
| **Sacks, Schegloff & Jefferson (1974)** — turn-taking | `policy.select_next_speakers()` | Strict SSJ cascade: addressed > name-mentioned > personality-biased self-selection. One speaker per round. |
| **Ouchi & Tsuboi (2016)** — addressee selection | `policy.extract_discourse()`, `state.DiscourseGraph` (+ `add_challenge` / `answer_challenge`) | Tracks last-addressed, pending questions, and **challenge ⇄ response edges** (Stage 5). Questions and pushbacks can't be silently dropped. |
| **MUCA (2024)** — multi-user agent cooldowns | `state.ParticipantState.strategy_cooldowns` + filter in `policy.plan_turn()` | Recently-used acts are filtered out of the sampling pool; prevents strategy lock-in. |

### Layer B — Cognitive / persona scaffolding

| Source | Module | Role |
|---|---|---|
| **Toulmin (1958)** — argumentation structure | `persona.AgentBeliefs` (`reasons`, `reservation`, `would_reconsider_if`), `prompts.agent_beliefs_group`, `prompt_context.build_speaker_card` | Each persona carries an **argument kit**: 1–2 concrete reasons (warrants), one honest concern about a rival (reservation), and the condition that would change their mind. This is what lets sims argue rather than restate. |
| **McCrae & John (1992)** — Big Five | `persona.Persona` traits, `policy.derive_bias()`, `persona.derive_speech_signature()` | Big Five is the source for trait sampling. It now routes **two ways**: into act-sampling weights (planner bias) and into a deterministic **SpeechSignature** (hedge / directness / think-aloud / detail) that scaffolds distinct voices in the speaker card. |
| **Shanahan (2023, *Role-Play with LLMs*)** | `persona.SpeechSignature`, `prompt_context.build_speaker_card` | Justifies external scaffolding of persona over "be the character" instructions. The speech signature lives in the speaker card as a register descriptor, never as named phrases. |
| **Park et al. (2023, Generative Agents)** — memory + reflection (scaled down) | `state.ParticipantState.points_made`, `prompt_context.build_memory_block` | Per-sim **relevance-filtered memory** replaces a raw transcript dump in the speaker card: what you've already said (anti-repeat), pushback aimed at you, others' live arguments, and what others said they care about. No reflection step — the structural pieces are kept; the heavy machinery is not. |

### Layer C — Deliberation quality & convergence

| Source | Module | Role |
|---|---|---|
| **Liang et al. (2023)**, **Du et al. (2023)** — multi-agent debate / divergent thinking | `persona._enforce_divergence`, `moderation.detect_facilitatable_disagreement` | Cooperative sims that start in **different places for good reasons** are the fuel for real exchange. A deterministic post-belief-generation pass spreads preferred options. The moderator surfaces the live disagreement by name when the group isn't already engaging. |
| **Deliberative-quality framework** (justification, reciprocity, reflexivity) | `reasoning.deliberation_metrics()` (in `_eval.json`) | Evaluation-only metrics: share of stance turns with a reason marker, share of turns addressing/answering another sim, share of turns showing update (concession / conditional accept). |
| **Fisher (1970)** — decision emergence (favourable / unfavourable ratios) | `reasoning.fisher_ratios()` (in `_eval.json`) | **Evaluation-only.** Logged per dialogue; does not drive control flow. (Was load-bearing on paper before; honest framing now.) |

### What changed from the original five-paper roster

- **Fisher** has been demoted from "load-bearing module" to **eval-only metric**. It was already logging-only in the code; the framing now matches reality.
- **Toulmin**, **Generative Agents (scaled down)**, **Shanahan**, and **multi-agent debate (Liang/Du)** are now first-class. They are what makes deliberation possible in the first place.
- **SSJ, Ouchi/Tsuboi, MUCA** remain load-bearing as before. Ouchi/Tsuboi is **stronger** now — the addressee graph also carries challenge edges and answered-challenge tracking, which gates phase transitions.
- **McCrae & John** is now **load-bearing twice over**: act-bias weights AND speech signature. Same paper, two routes from trait to behaviour.

## Entry Points

`main.py` is the single executable entry.

```
python main.py                   interactive — prompted for one topic
python main.py scenarios.txt     batch — one topic per non-comment line
```

`config.yaml` controls all behaviour. **No magic numbers in code.**

## Module Layout

All source files are flat under `src/` — no subpackages.

```
src/
├── config_loader.py    typed cfg object; all modules import cfg from here
├── llm_client.py       LLM provider abstraction (uni/groq/gemini)
├── prompts.py          ALL LLM-facing text lives here (hard rule)
├── persona.py          Persona, AgentBeliefs (argument kit), SpeechSignature,
│                       PersonaBuilder (single grouped LLM path), divergence
│                       enforcement
│
├── state.py            DialogueAct enum (incl. CHALLENGE), TurnRecord,
│                       StanceUpdate, OptionState, StanceTable,
│                       DiscourseGraph (+ ChallengeRecord),
│                       ParticipantState (+ memory: points_made,
│                       stated_priority, position_with_reason_stated),
│                       StructuredState, StateTracker
│
├── policy.py           PersonalityBias, derive_bias, sample_hard_blocker,
│                       select_next_speakers, repetition_pressure,
│                       extract_discourse, TurnPlan, plan_turn (now
│                       routes open challenges and uses argument kit)
│
├── reasoning.py        ConsensusEngine, fisher_ratios (eval-only),
│                       deliberation_metrics (Stage 5 signals),
│                       fact_check (scoped to claims-about-options),
│                       repair_directive
│
├── prompt_context.py   compact speaker-card builders + memory_block
│                       + perceived-priorities + others-arguments
│
├── orchestrator.py     loop coordinator; wires all layers together;
│                       deliberation-gated narrowing
├── simulator.py        generates one participant turn via LLM, with the
│                       memory block and open-challenger hook
├── moderation.py       facilitator-style intervention timing + LLM lines
│                       (replaces the blunt info-gap shutdown)
├── logger.py           2-file output per dialogue (.txt, .eval.json),
│                       now includes deliberation metrics + challenges
└── utils.py            OptionResolver (prose↔state bridge) + history helpers

eval/
└── eval_scenarios.txt  fixed topics for batch runs
```

## High-Level Pipeline

For each topic:

1. `Orchestrator` asks the LLM to **classify the decision kind** (concrete vs abstract pick) and generate four fitting options + an opening question.
2. `PersonaBuilder.generate_names_and_roles()` — one LLM call: N names + roles tuned to the topic.
3. `PersonaBuilder.build_all()` samples Big Five traits, enforces trait diversity, then one LLM call for all backstories + goals (each backstory carries a **concrete experience** that becomes argumentative evidence).
4. `PersonaBuilder.assign_beliefs()` — one LLM call for all private belief states (the **Toulmin argument kit**), followed by deterministic **divergence enforcement** (spread preferred options) and **acceptable-overlap enforcement** (guarantee one common fallback).
5. Each persona is wrapped by a `Simulator`.
6. `Orchestrator.run_simulation()` initialises `StateTracker`, samples the per-dialogue hard-blocker flag (5%), and drives the dialogue through phases.
7. Each round: `select_next_speakers()` picks ONE speaker; `plan_turn()` plans their act; `Simulator.generate_turn()` calls the LLM with the speaker card, **memory block**, recent turns, and move instruction.
8. `StateTracker.update()` parses each turn into a `TurnRecord` and updates `StructuredState` (stance updates, challenge edges, captured priority, per-sim memory).
9. `ConsensusEngine` recomputes consensus state from the public `StanceTable`.
10. `ModerationEngine` decides if/when to intervene — facilitator moves first (surface disagreement, ask what would change a mind, reframe missing detail), force-close only when truly spent.
11. `DialogueLogger` writes the chat `.txt` (with personas + reasons + reservations + voice sig at the top) and `.eval.json` (with `fisher_ratios`, `deliberation` metrics, `challenges`, `stated_priorities`).

Total LLM setup cost per dialogue: **3 calls** (options/opening, names+roles+concepts, beliefs) — independent of participant count.

## Dialogue Phases

The arc is shaped so people establish *what they want* before pitching options,
then converge by exchange — not by roll-call.

- `opening` — each participant, once: a natural hello **and the one thing they care about** (a priority/constraint), **not an option**. This is captured into `ParticipantState.stated_priority` and shown to every other sim as a perceived priority (theory-of-mind).
- `negotiation` — open discussion: react to each other, weigh priorities, ask, push back. Options come up only when relevant; turns are *not* forced into "I vote X". Sims may use their backstory experience as a warrant; the grounding check forbids only invented option attributes.
- `narrowing` — moderator asks everyone to land on a pick; sims commit to an option here (14–30 words). **Stage 5: triggered by deliberation signals** (positions stated with reasons + at least one answered challenge + repetition pressure) within a turn-count floor/ceiling, not turn-count alone.
- `emergence` — every sim has voted; the group closes in on the leading candidate (cap: 32 words).
- `confirmation` — moderator asks once if the candidate works; everyone answers in one natural pass (cap: 14 words).
- `closure` — each participant signs off in their own words (cap: 16 words). Gracious if it wasn't their pick.

**Turns are not over-directed.** `plan_turn` marks each `TurnPlan` as `directive` only for real obligations (answer a pending question, answer an unanswered challenge aimed at this sim, commit a vote at narrowing, confirm/reject). Sampled negotiation acts are `directive=False` and are **not** surfaced to the model as "Planned act: …" — they only steer stance tracking.

### Scaling with participant count

The system is built for 2–5 sims (set `simulation.num_participants`). Turn budget and pacing scale with `n`:

- `hard_ceiling = max(turns.min_ceiling, n * turns.ceiling_per_participant)`
- narrowing has a **floor** (`turns.min_before_narrowing_per_participant`) and a **ceiling** (`turns.narrow_after_per_participant`); between them the deliberation gate fires it
- moderator escalation gets `n` rounds of grace so larger groups finish collecting votes
- dissenter tolerance is `min(consensus.max_dissenters, (n-1)//2)`

### Register

Relaxed but articulate. The `Persona.style_rule()` block sets length register from `response_length`; the `SpeechSignature` block adds trait-driven voice features (hedge / directness / think-aloud / detail). Together they keep sims from sounding identical.

## Personas

### Big Five model

`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`.
`response_length` (1–5) is communication-style control, not a Big Five trait.

Default `trait_ranges` lean cooperative; post-sample enforcement prevents everyone-too-agreeable groups.

### Argument kit (Toulmin)

Each persona's private `AgentBeliefs` carries:

- `preferred` — top option before discussion
- `acceptable` — options they could live with (size controlled by `divergence.target_acceptable_*`)
- `rejected` — options they actively resist (typically empty)
- `key_concern` — main reason behind their preference
- **`reasons`** — 1–2 concrete reasons drawn from the persona's goal/backstory, phrased as their knowledge/experience (Toulmin warrants)
- **`reservation`** — one honest concern about a rival option, framed as a concern (not a veto)
- **`would_reconsider_if`** — the concrete condition that would move them off `preferred` (makes "what would change your mind" answerable)

**Belief consistency rules:** `key_concern` must be consistent with `preferred`. `reasons` must come from the backstory. `reservation` must NOT be a refusal of a rival; only the rare hard-blocker (`stubbornness.hard_blocker_dialogue_probability`) is a refusal.

### Divergence enforcement

After belief generation, `persona._enforce_divergence` spreads `preferred` so the group doesn't all start on the same option (the fuel for real discussion). `persona._enforce_acceptable_overlap` guarantees a shared fallback option so consensus is reachable. The hard-blocker flag is the only sanctioned route to `force_close`.

### Speech signature (Shanahan)

`Persona.speech_signature()` deterministically maps the five traits to four floats:

- `hedge_propensity` — neuroticism + agreeableness lift hedging
- `directness` — (1 - agreeableness) drives plain "no"
- `thinkaloud_propensity` — extraversion + (1 - conscientiousness)
- `detail_orientation` — conscientiousness drives citing concrete option text

These appear in the speaker card as a register descriptor (`"hedges naturally; thinks aloud; cites concrete details"`). Phrases are never prescribed; only register hints. All weights live in `config.yaml :: voice`.

## Turn-Taking (SSJ)

`policy.select_next_speakers()` returns **one speaker per round**:

1. **Obligated addressees** of pending questions (`DiscourseGraph.pending_questions`)
2. **Open invitation** — questions without an explicit addressee force any non-asker to respond
3. **Recently name-mentioned participant** (no pending question)
4. **Personality-biased self-selection** (extraversion, participation debt, novelty)

Phase-specific rules: opening covers everyone once before repeats; confirmation gives the primary the first slot.

## Act Planner

`policy.plan_turn()` selects a `TurnPlan` before each LLM call.

Priority cascade:

1. Discourse obligation (pending question → `ANSWER`)
2. **Open challenge aimed at this sim** → `ANSWER` (Stage 5)
3. Phase obligation (opening → `OPEN_PRIORITY`; narrowing+no-vote → `COMMIT_VOTE`; confirmation → `CONFIRM` / `REJECT_WITH_REASON`)
4. Emergence soft-path (candidate in `acceptable` → `CONDITIONAL_ACCEPT` with `would_reconsider_if` as condition)
5. Hard-blocker path (`is_true_hard_blocker` + candidate in `rejected` → `REJECT_WITH_REASON`)
6. Personality-biased sampling with MUCA cooldowns (base weights in `config.yaml :: act_planner.base_weights`; bias multipliers in `personality_bias`)

The `DialogueAct` enum now includes `CHALLENGE` — explicit addressed pushback at another sim. The state tracker fires `add_challenge` when it sees this act, and `answer_challenge` when the target rebuts within the configured window.

## Memory block (Stage 6, Park 2023 scaled down)

`prompt_context.build_memory_block` produces a compact, relevance-filtered view that **replaces** sending the raw last-N transcript to the model. Four sections, each optional and capped per `cfg.memory`:

- **What you've already said** — the speaker's own substantive points, for anti-repetition (kills the "Liam-says-sandwiches-3×" failure).
- **Pushback aimed at you** — unanswered challenges this sim must engage.
- **Others' live arguments** — for build-on; not a transcript.
- **What others care about** — captured stated priorities from the opening (theory-of-mind).

The raw last-N transcript is still passed under `RECENT TURNS` so the model has actual prose to anchor on, but the memory block is what makes turns specific, on-topic, and non-repetitive.

## Prompting

All LLM-facing templates live in `src/prompts.py`. No other module constructs LLM-facing prose.

### Compact speaker-card prompt

```
SPEAKER CARD     name, role, register, voice signature, personality,
                 background, stance + Toulmin argument kit
OPTIONS          only the relevant options (preferred + acceptable + candidate)
GROUP STATE      candidate, current votes, unresolved rejections
YOUR MEMORY      points you've made, pushback aimed at you,
                 others' live args, what others care about
YOUR MOVE        phase instruction + optional TurnPlan act + interaction + position
RECENT TURNS     last N turns (raw)
OUTPUT           word budget + formatting rules
```

`prompt_context.py` builds each section from structured orchestrator state.

### Position discipline (3 templates)

- Decision phase + candidate is in sim's `acceptable` (or anchor) → "say yes briefly"
- Decision phase + candidate is in sim's `rejected` → "say no with one specific reason"
- Decision phase + neither → "hedge briefly or name what you'd need"
- Negotiation without candidate → "engage with what others said; don't restate"

### Repetition (single signal)

`policy.repetition_pressure()` is the only repetition metric: rolling Jaccard overlap across the last `pressure_window` participant turns. When it crosses `cfg.repetition.stall_increment_threshold`, the prompt receives a single "loop detected, change move" line and the orchestrator increments `stall_rounds`. There are no forbidden-opener lists, no semantic-keyword detection.

### Moderator (facilitator style)

`ModerationEngine` is a facilitator. It intervenes when one of the following fires:

- **Info-chase** — sims explicitly chasing a missing option attribute (cost, price, exact number). The moderator **reframes** toward judgment (`prompts.moderator_reframe_missing_detail`); it never says "decide based on what's listed".
- **Facilitate disagreement** — two sims hold opposing stances on the same option and no challenge ⇄ response has happened. The moderator surfaces the disagreement by name (`prompts.moderator_facilitate_disagreement`).
- **Outlier** — a sim repeats themselves verbatim after narrowing (one nudge).
- **Stall** — repetition pressure crosses threshold for ≥2 rounds.

A `facilitate_cooldown` prevents two facilitation moves in a row. Escalation levels (0–3) scale moderator directness. Level 3 force-closes via `ConsensusEngine.best_available_decision()`.

## Prose ↔ State bridge (`utils.OptionResolver`)

Unchanged in shape; now exposes `option_mention_spans()` so the grounding check can scope claims-about-options correctly (Stage 4).

## Consensus

`reasoning.ConsensusEngine` derives consensus state from `StanceTable` only:

- `none` → `candidate_emerging` → `majority_candidate` → `conditional_consensus` → `full_consensus`
- Special states: `blocked` (active hard blocker) and `failed`

`leading_weights` (the "what's winning" weights) and `stance_weights` (the force-close weights) are both in `config.yaml :: consensus`. Dissenter tolerance scales with group size.

## Grounding (Stage 4)

`reasoning.fact_check()` now scopes its check to **claims about options**, not all world-knowledge:

- **Currency / percentages** — flagged if absent from source (always option-attribute-like).
- **Bare numbers** — flagged ONLY when they sit within `cfg.grounding.option_proximity_chars` of an option letter or alias. A persona saying "CRISPR has been around since 2012" passes; a persona saying "Option A is 40 dollars" gets flagged.
- **Quoted strings** — flagged if absent (invented named feature).
- **Parenthesised digit-asides** — the classic fabrication shape.

When flagged, the turn is regenerated once with `repair_directive()`, which now permits world knowledge / experience and forbids only invented option attributes.

## Logging — two files per dialogue

| File | Contents |
|---|---|
| `<id>.txt` | Header + **persona block** (name, traits, **voice signature**, goal, backstory, beliefs + **reasons + reservation + would_reconsider_if**) + chat transcript + Outcome/Tokens footer. |
| `<id>.eval.json` | Metadata, outcome, tokens, Gini + per-speaker/per-phase counts, vote flips, confirmation rejections, full personas, **Fisher ratios (eval-only)**, **`deliberation` metrics** (justification / reciprocity / reflexivity + gating signals), **`challenges`** (challenger, target, answered_turn_id), **`stated_priorities`**, consensus state, public preferences, full per-turn structured trace. |

## Key config.yaml values

| Section | Key | Default | Notes |
|---|---|---|---|
| `simulation` | `num_participants` | `3` | 2–5 supported |
| `turns` | `min_before_narrowing_per_participant` | `3` | Floor — narrowing can't fire below this |
| `turns` | `narrow_after_per_participant` | `5` | Ceiling — narrowing forced at or above this |
| `argument_kit` | `reasons_per_persona` / `reasons_min` / `reasons_max` | `2 / 1 / 3` | Toulmin warrants on each persona |
| `argument_kit` | `reservation_required` / `reconsider_required` | `true` | The shape of the argument kit |
| `divergence` | `target_acceptable_min` / `max` | `2 / 3` | Smaller acceptable sets so disagreement is real |
| `divergence` | `enforce_distinct_preferred` | `true` | Spread preferred options across the group |
| `divergence` | `required_common_acceptable` | `1` | Shared fallback that keeps consensus reachable |
| `deliberation` | `challenges_to_unlock_narrowing` | `1` | Stage 5 — must have at least one answered exchange |
| `deliberation` | `exhaustion_pressure_threshold` | `0.60` | Below the global stall threshold |
| `deliberation` | `challenge_window_turns` | `6` | A challenge counts as "answered" if rebutted within N |
| `memory` | `points_made_max` | `4` | Compact anti-repeat memory size |
| `memory` | `others_arguments_max` | `4` | Build-on memory size |
| `memory` | `open_challenges_max` | `2` | Pushbacks shown to a sim |
| `voice` | `hedge_*` / `directness_*` / `thinkaloud_*` / `detail_*` | various | Trait → speech-feature weights |
| `act_planner` | `base_weights.negotiation` | varies | Per-phase MUCA base weights (incl. `CHALLENGE`) |
| `personality_bias` | `concession_*` / `objection_*` / `clarification_*` | varies | Big Five → propensity weights |
| `grounding` | `option_proximity_chars` | `60` | Window around option mentions for number-attribute flag |
| `moderation` | `facilitate_cooldown_rounds` | `2` | Prevents back-to-back facilitation moves |
| `repetition` | `stall_increment_threshold` | `0.70` | Global stall trigger |
| `stubbornness` | `hard_blocker_dialogue_probability` | `0.05` | Per-dialogue rare hard blocker |
| `fisher` | `window_size` | `8` | Eval-only ratio window |

## Development Notes

After edits, sanity-check with:
```
python -m compileall -q main.py src
```

### Hard rules

- **All magic numbers live in `config.yaml`.** No literal constants in code that could be tuned.
- **All LLM-facing prose lives in `prompts.py`.** Other modules assemble structured context; they do not write prose.
- **Do not list specific filler phrases in prompts.** Any named phrase gets overused.
- **Belief generation quality dominates runtime fixes.** If `acceptable` is too narrow or `reasons` are generic, no heuristic will fix the deadlock or the hollow exchange.
- **Private beliefs never enter the structured force-close path.** `ConsensusEngine.best_available_decision()` uses public stances only.
- **One speaker per round.** Don't reintroduce multi-speaker round logic.
- **Repetition is one signal.** Don't layer phrase-blacklists or opener-tracking on top of `repetition_pressure`.
- **Fisher is logging-only.** Don't add control flow that branches on Fisher ratios — narrowing is gated on deliberation signals, not Fisher.
- **No fabricated defaults.** Setup LLM calls (options, names+roles, concepts, beliefs) and turn generation raise on failure or malformed output. The only kept non-LLM substitution is the force-close system line when the model impersonates a participant (a malformed line, not a failed call).
- **Everything reads through `OptionResolver`.** Don't reintroduce raw `\boption [a-d]\b` scanning anywhere.
- **Closure must be honest.** Forced close acknowledges no full agreement; a real agreement uses the affirming closure.
- **Keep it scalable.** New pacing/threshold logic must derive from `n = len(sims)`, not hardcode for 3. Test 2 and 5 before assuming it generalises.
- **Grounding lets world knowledge through.** Sims arguing from their own experience (Toulmin warrants) is intended; only INVENTED OPTION ATTRIBUTES are blocked. Don't tighten the check back into a sterilising filter.
- **The moderator facilitates, it doesn't terminate.** Don't reintroduce a blunt "isn't specified, decide on what's listed" shutdown. The reframer is the answer; the discussion is the product.
