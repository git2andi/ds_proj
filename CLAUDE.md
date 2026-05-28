# CLAUDE.md

## Project Overview

This project is a university dialogue-simulation system. It creates multi-party group chats in which several simulated participants ("Sims") discuss a random decision topic and try to reach a group outcome.

The research/design goal is not just to produce a correct final option. The generated transcript should feel like a plausible human group chat:

- participants have stable preferences but do not repeat the same sentence forever
- disagreement is visible and grounded in each person's priorities
- questions should be answered before the group moves on
- message length and structure should vary naturally — mostly short chat-like turns, occasionally longer when a point needs explanation
- each participant should sound like a distinct person, not a copy of the same polite evaluator

Behavior comes from the interaction of the architecture below — each layer is grounded in one of the five reference papers.

## Research Grounding

Each paper is represented by a specific, load-bearing piece of the architecture:

| Paper | Module | Use |
|---|---|---|
| **Fisher (1970)** — decision emergence | `reasoning.fisher_ratios()` | Favorable / unfavorable / ambiguous / conditional ratios computed per dialogue and logged in `_eval.json`. Not used for control flow. |
| **Sacks/Schegloff/Jefferson (1974)** — turn-taking | `policy.select_next_speakers()` | Strict cascade: 1a obligated addressees > 1a' name-mentioned > 1b personality-biased self-selection. One speaker per round. |
| **Ouchi & Tsuboi (2016)** — addressee selection | `policy.extract_discourse()` + `state.DiscourseGraph` | Tracks last-addressed + pending-question targets so questions can't be silently dropped. |
| **McCrae & John (1992)** — Big Five | `persona.Persona` traits + `policy.derive_bias()` | Big Five operationalised as probabilistic behavioral biases over dialogue acts and speaker selection — never as named phrases. |
| **MUCA (2024)** — multi-user agent cooldowns | `state.ParticipantState.strategy_cooldowns` + filter in `policy.plan_turn()` | Recently-used acts are filtered out of the sampling pool, preventing strategy lock-in. |

## Entry Points

`main.py` is the single executable entry.

```
python main.py                   interactive — prompted for one topic
python main.py scenarios.txt     batch — one topic per non-comment line
```

`config.yaml` controls all behavior. No magic numbers in code.

## Module Layout

All source files are flat under `src/` — no subpackages.

```
src/
├── config_loader.py    typed cfg object; all modules import cfg from here
├── llm_client.py       LLM provider abstraction (uni/groq/gemini)
├── prompts.py          ALL LLM-facing text lives here (hard requirement)
├── persona.py          Persona, AgentBeliefs, PersonaBuilder (single grouped LLM path)
│
├── state.py            DialogueAct enum, TurnRecord, StanceUpdate, OptionState,
│                       StanceTable, DiscourseGraph, ParticipantState,
│                       StructuredState, StateTracker
│
├── policy.py           PersonalityBias, derive_bias, sample_hard_blocker,
│                       select_next_speakers, repetition_pressure,
│                       extract_discourse, TurnPlan, plan_turn
│
├── reasoning.py        ConsensusEngine, fisher_ratios, fact_check, repair_directive
│
├── prompt_context.py   compact speaker-card prompt builders
│
├── orchestrator.py     loop coordinator; wires all layers together
├── simulator.py        generates one participant turn via LLM
├── moderation.py       intervention timing + LLM-generated moderator lines
├── logger.py           2-file output per dialogue (.txt, .eval.json)
└── utils.py            OptionResolver (prose↔state bridge) + history helpers

eval/
└── eval_scenarios.txt  fixed topics for batch runs
```

## High-Level Pipeline

For each topic:

1. `Orchestrator` asks the LLM to generate four concrete options + an opening question.
2. `PersonaBuilder.generate_names_and_roles()` — one LLM call: N names + roles tuned to the topic.
3. `PersonaBuilder.build_all()` samples Big Five traits, enforces diversity, then one LLM call for all backstories + goals.
4. `PersonaBuilder.assign_beliefs()` — one LLM call for all private belief states.
5. Each persona is wrapped by a `Simulator`.
6. `Orchestrator.run_simulation()` initialises `StateTracker`, samples the per-dialogue hard-blocker flag (5%), and drives the dialogue through phases.
7. Each round: `select_next_speakers()` picks ONE speaker; `plan_turn()` plans their act; `Simulator.generate_turn()` calls the LLM.
8. `StateTracker.update()` parses each turn into a `TurnRecord` and updates `StructuredState`.
9. `ConsensusEngine` recomputes consensus state from the public `StanceTable`.
10. `ModerationEngine` decides if/when the moderator should intervene.
11. `DialogueLogger` writes the chat `.txt` (with personas at the top) and `.eval.json`.

Total LLM setup cost per dialogue: **4 calls** (options, names+roles, concepts, beliefs) — independent of participant count.

## Dialogue Phases

The arc is shaped so people establish *what they want* before pitching options, then converge — not a roll-call.

- `opening` — each participant, once: a natural hello **and the one thing they care about** (a priority/constraint), **not an option**. This surfaces genuine difference before options enter. (cap: 28 words)
- `negotiation` — open discussion: react to each other, weigh priorities, ask, push back. Options come up only when relevant; turns are *not* forced into "I vote X".
- `narrowing` — moderator asks everyone to land on a pick; sims commit to an option here (14-30 words).
- `emergence` — every sim has voted; the group closes in on the leading candidate (cap: 32 words).
- `confirmation` — moderator asks once if the candidate works; everyone answers in one natural pass (no per-sim roll-call; cap: 14 words).
- `closure` — each participant signs off in their own words, generated (gracious if it wasn't their pick; cap: 16 words).

Phase transitions are turn-count based. **Fisher (1970) ratios are logged per dialogue but do not drive transitions** — they exist as an evaluation signal, not a control input.

**Turns are not over-directed.** `plan_turn` marks each `TurnPlan` as `directive` only for real obligations (answer a pending question, commit a vote at narrowing, confirm/reject). Sampled negotiation acts are `directive=False` and are **not** surfaced to the model as "Planned act: …" — they only steer stance tracking. This is what stops every turn reading "I support Option A" and keeps discussion conversational.

### Scaling with participant count

The system is built for 2–5 sims (set `simulation.num_participants`). Turn budget and pacing scale with `n`:

- `hard_ceiling = max(turns.min_ceiling, n * turns.ceiling_per_participant)`
- narrowing fires after `n * turns.narrow_after_per_participant` participant turns (or on a stall)
- moderator escalation gets `n` rounds of grace so larger groups finish collecting votes before the patience clock starts
- dissenter tolerance is `min(consensus.max_dissenters, (n-1)//2)`

### Register

Relaxed but articulate — like a thoughtful adult in a group chat, not exaggerated slang and not a formal panel. Normal punctuation and capitalization are expected; full sentences and the occasional fragment both fine. Length varies per turn (`response_length` 2–4 by default). Voice rules live in `prompts.sim_turn_compact`.

## Personas

### Big Five personality model

`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`. `response_length` (1-5) is communication-style control, not a Big Five trait.

Default `trait_ranges` bias toward cooperative sims (`agreeableness [3,5]`, `openness [3,5]`); enforced post-sample diversity prevents an entire group of high-agreeableness sims.

### Stubbornness — one knob

`stubbornness.hard_blocker_dialogue_probability` (default 0.05): per-dialogue Bernoulli that picks ONE participant whose act planner is biased toward `REJECT_WITH_REASON` when the candidate is in their `rejected` list.

### Beliefs

Each persona has private `AgentBeliefs`:

- `preferred` — top option before discussion
- `acceptable` — options they could live with (aim: 3 of 4)
- `rejected` — options they actively resist (typically empty)
- `key_concern` — main reason behind their preference
- `concession` — condition under which they could accept a non-preferred option

**Belief consistency:** the `key_concern` must be consistent with `preferred`. The `acceptable` list must include any option that meaningfully addresses the concern. Overly narrow acceptable lists cause artificial deadlock that no runtime heuristic can fix — fix the belief prompt.

Private beliefs guide coherence but **decisions must be visible in the transcript**. `ConsensusEngine` never reads private beliefs.

### Personality and register

`personality_summary()` produces register descriptors only — never named phrases. `_STYLE_RULE` maps `response_length` (1-5) to per-turn word budgets `[14, 22, 30, 42, 55]`. Phase caps override budgets when stricter.

`PersonalityBias` (in `policy.py`) maps the same trait values to float biases consumed by `plan_turn()` (act sampling weights) and `select_next_speakers()` (self-selection score). These floats never appear in prompts.

## Turn-Taking (SSJ)

`policy.select_next_speakers()` implements the Sacks/Schegloff/Jefferson (1974) priority cascade and returns **one speaker per round**:

1. **Obligated addressees** of pending questions (`DiscourseGraph.pending_questions`)
2. **Open invitation** — questions without an explicit addressee force any non-asker to respond
3. **Recently name-mentioned participant** (no pending question)
4. **Personality-biased self-selection** (extraversion, participation debt, novelty)

Phase-specific rules: opening covers everyone once before repeats; confirmation gives the primary the first slot.

## Act Planner

`policy.plan_turn()` selects a `TurnPlan` (planned `DialogueAct` + target option) before each LLM call.

Priority cascade:

1. Discourse obligation (pending question -> `ANSWER`)
2. Phase obligation (opening -> `OPEN_PRIORITY`; narrowing+no-vote -> `COMMIT_VOTE`; confirmation -> `CONFIRM` / `REJECT_WITH_REASON`)
3. Emergence soft-path (candidate in `acceptable` -> `CONDITIONAL_ACCEPT`)
4. Hard-blocker path (`is_true_hard_blocker` + candidate in `rejected` -> `REJECT_WITH_REASON`)
5. Personality-biased sampling with MUCA-style act cooldowns

The `DialogueAct` enum is the minimum set actually consumed by act planning, stance extraction, or moderator branching.

## Prompting

All LLM-facing templates live in `src/prompts.py`. No other module constructs LLM-facing prose.

### Compact speaker-card prompt

```
SPEAKER CARD     name, role, register, private stance
SHARED FACTS     only the relevant options (preferred + acceptable + candidate)
GROUP STATE      candidate, current votes, unresolved rejections
YOUR MOVE        phase instruction + optional TurnPlan act + interaction + position
RECENT TURNS     last N turns
OUTPUT           word budget + formatting rules
```

`prompt_context.py` builds each section from structured orchestrator state.

### Position discipline (3 templates)

Collapsed into three branches:
- Decision phase + candidate is in sim's `acceptable` (or anchor) -> "say yes briefly"
- Decision phase + candidate is in sim's `rejected` -> "say no with one specific reason"
- Decision phase + neither -> "hedge briefly or name what you'd need"
- Negotiation without candidate -> "engage with what others said; don't restate"

### Repetition (single signal)

`policy.repetition_pressure()` is the only repetition metric: rolling Jaccard overlap across the last `pressure_window` participant turns. When it crosses `cfg.repetition.stall_increment_threshold`, the prompt receives a single "loop detected, change move" line and the orchestrator increments `stall_rounds`.

There are no forbidden-opener lists, no semantic-keyword detection, no "what if" hypothetical detection. One signal, one nudge.

### Moderator (one style)

`ModerationEngine` is a single direct style. It intervenes when:
- the group is **fishing for info the options don't hold** (clarify) — `detect_info_gap()` fires on an explicit lament ("not specified", "still waiting", "what's the cost") or a pile-up of ≥2 questions in the last 4 turns,
- a sim repeats themselves verbatim after narrowing (outlier),
- repetition pressure crosses threshold for ≥2 rounds (stall).

The **clarify** intervention is the answer to hallucination: instead of letting sims invent missing facts, the moderator answers from the option texts or states plainly that a detail isn't specified and the group should decide on what's listed (`prompts.moderator_clarify_info`). It runs at most once per `n+1` rounds (`state.info_gap_cooldown`). This is why options must be **self-contained** — `option_generation` is instructed that the option lines are the only information the group will ever have, to describe "best for" by priority (never an invented person's name), and to use qualitative trade-offs ("higher cost") rather than numbers the sims would then chase.

Escalation levels (0-3) scale the moderator's directness. Level 3 triggers force-close via `ConsensusEngine.best_available_decision()` — no separate "test the compromise" path.

## Prose ↔ State bridge (`utils.OptionResolver`)

Sims talk like people — they name venues ("let's do La Brisa", "Can Culleretes feels pricey"), they rarely say "Option A". The entire state layer keys off option letters, so a resolver translates between the two. It is built once per dialogue from the option texts and threaded into `StateTracker`, every `Simulator`, the `ModerationEngine`, and the `Orchestrator` so all layers read the same references.

- `option_aliases()` parses each option's label into proper-name aliases: parentheticals ("(The Grand Kaiser)"), the leading name phrase, and the name minus a trailing category word ("La Brisa Cafe" → "la brisa"). Generic single tokens that collide across options ("hotel", "budget") are dropped.
- `options_in(text)` → every option referenced (letter or alias), possessive-aware ("la brisa's").
- `vote_in(text)` → the single option a turn *commits* to. Requires an explicit signal: a first-person frame ("I'd go with X"), a committal cue after the name ("X works for me"), or an agreeing lead ("yeah, X"). A bare mention or a soft suggestion ("what about X?") is discussion, not a vote. Negation-guarded.

Without this bridge the `StanceTable` stays empty, every dialogue force-closes on an arbitrary option, and Fisher ratios read zero. It is the precondition for everything below.

## Consensus

`reasoning.ConsensusEngine` derives consensus state from `StanceTable` only:

- `none` → `candidate_emerging` → `majority_candidate` → `conditional_consensus` → `full_consensus`
- Special states: `blocked` (active hard blocker) and `failed`

The orchestrator concludes (success) on `full_consensus`/`conditional_consensus`, or on `majority_candidate` once everyone has voted and dissenters are within tolerance. `best_available_decision()` is the single force-close path — `cfg.consensus.stance_weights` over public stances.

Dissenter tolerance scales with group size: `min(cfg.consensus.max_dissenters, (n-1)//2)` — 0 for a pair, 1 for 3–4, etc. Vote detection routes through `OptionResolver.vote_in()`.

## Grounding

`reasoning.fact_check()` deterministically flags numbers, quoted strings, and parenthesised asides in generated turns that have no basis in the option texts or topic. Fully deterministic — no LLM call for the check itself.

When `cfg.grounding.enable_fact_check: true`, turns with suspicious claims are regenerated once with a repair directive. Quote/aside flags only trigger on turns ≥ `min_words_to_check` (12), but an **invented number is always caught** regardless of length — a short "yeah, if it's under $100" is exactly where fabrication slips in. The structural defence is the clarify intervention above: if sims aren't chasing absent numbers, they don't invent them.

## Logging — two files per dialogue

| File | Contents |
|---|---|
| `<id>.txt` | Header + **persona block** (name, traits, goal, backstory, beliefs) + chat transcript + Outcome/Tokens footer. Self-contained — readable without opening the JSON. |
| `<id>.eval.json` | Everything else for analysis: metadata, outcome, tokens (setup vs dialogue), Gini + per-speaker/per-phase counts, vote flips, confirmation rejections, full personas, Fisher ratios, consensus_state_final, public preferences, structured per-turn trace (acts, addressees, stance updates). |

## Key config.yaml values

| Section | Key | Default | Notes |
|---|---|---|---|
| `simulation` | `num_participants` | `3` | Group size; 2–5 supported |
| `turns` | `ceiling_per_participant` | `9` | Hard ceiling = max(min_ceiling, n × this) |
| `turns` | `min_ceiling` | `18` | Floor for the hard ceiling |
| `turns` | `narrow_after_per_participant` | `4` | Force narrowing after n × this turns |
| `turns` | `escalation_level_3` | `7` | (+n grace) triggers force-close |
| `consensus` | `stall_rounds_to_force` | `2` | Rounds of stall before forcing |
| `consensus` | `max_dissenters` | `1` | Upper bound; effective = min(this, (n-1)//2) |
| `personas` | `trait_ranges.response_length` | `[2, 4]` | Slightly fuller, varied turn lengths |
| `repetition` | `jaccard_threshold_self` | `0.45` | Self-repetition trigger |
| `repetition` | `stall_increment_threshold` | `0.70` | Global stall trigger |
| `personas` | `trait_ranges.agreeableness` | `[3, 5]` | Cooperative baseline |
| `personas` | `trait_ranges.openness` | `[3, 5]` | Open to alternatives |
| `stubbornness` | `hard_blocker_dialogue_probability` | `0.05` | Per-dialogue rare hard blocker |
| `fisher` | `window_size` | `8` | Window for Fisher ratio computation |

## Development Notes

After edits, sanity-check with:
```
python -m compileall -q main.py src
```

### Hard rules

- **All LLM-facing prose lives in `prompts.py`.** Other modules assemble structured context; they do not write prose.
- **Do not list specific filler phrases in prompts.** Any named phrase gets overused.
- **Belief generation quality dominates runtime fixes.** If `acceptable` is too narrow or `key_concern` contradicts `preferred`, no heuristic will fix the deadlock.
- **Private beliefs never enter the structured force-close path.** `ConsensusEngine.best_available_decision()` uses public stances only.
- **One speaker per round.** Don't reintroduce multi-speaker round logic — it muddies SSJ.
- **Repetition is one signal.** Don't layer phrase-blacklists or opener-tracking on top of `repetition_pressure`.
- **Fisher is logging-only.** Don't add control flow that branches on Fisher ratios — phases are turn-count driven.
- **No fabricated defaults.** Setup LLM calls (options, names+roles, concepts, beliefs) and turn generation raise on failure or malformed output — they do not substitute generic personas, options, or beliefs. A broken LLM must surface as an error, not a plausible-looking transcript. Batch mode catches per-dialogue so one failure doesn't stop the run. The only non-LLM substitution kept is the force-close system line when the model impersonates a participant (a malformed line, not a failed call).
- **Everything reads through `OptionResolver`.** Don't reintroduce raw `\boption [a-d]\b` scanning in any module — sims speak in venue names, so letter-only matching goes blind. Vote/mention/stance detection all route through the one resolver built from the option texts.
- **Closure must be honest.** A forced call uses `prompts.closure_templates_forced` (acknowledges it wasn't unanimous); only a real agreement (`state.agreement_reached`) uses the affirming `closure_templates`.
- **Keep it scalable.** New pacing/threshold logic must derive from `n = len(sims)`, not hardcode for 3. Test 2 and 5 before assuming it generalizes.
