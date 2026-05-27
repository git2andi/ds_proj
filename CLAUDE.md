# CLAUDE.md

## Project Overview

This project is a university dialogue-simulation system. It creates multi-party group chats in which several simulated participants ("Sims") discuss a random decision topic and try to reach a group outcome.

The main research/design goal is not just to produce a correct final option. The generated transcript should feel like a plausible human group chat:

- participants have stable preferences but do not repeat the same sentence forever
- disagreement is visible and grounded in each person's priorities
- questions should be answered before the group moves on
- compromise should be tested explicitly before a moderator forces closure
- message length and structure should vary naturally — mostly short chat-like turns, occasionally longer when a point needs explanation
- each participant should sound like a distinct person, not a copy of the same polite evaluator

The system is a controlled decision dialogue, not free-form chat. Behavior comes from the interaction of the architecture described below.

## Entry Points

`main.py` is the executable entry point.

Runtime modes:

- `python main.py` — interactive mode; asks for one topic
- `python main.py scenarios.txt` — batch mode; one topic per non-comment line
- `python main.py --personas my_personas.json` — uses persona overrides
- `python main.py scenarios.txt --personas my_personas.json` — batch mode with fixed personas

`config.yaml` controls all behavior. Every number that changes dialogue behavior lives there. No magic numbers in code.

## Module Layout

All source files are flat under `src/` — no subpackages.

```
src/
├── config_loader.py    — typed cfg object; all modules import cfg from here
├── llm_client.py       — LLM provider abstraction (uni/groq/gemini)
├── prompts.py          — ALL LLM-facing text lives here (hard requirement)
├── persona.py          — Persona dataclass, AgentBeliefs, PersonaBuilder
│
├── state.py            — all structured dialogue state in one file:
│                           DialogueAct enum, StanceUpdate, TurnRecord,
│                           OptionState, StanceTable, DiscourseGraph,
│                           ParticipantState, PhaseEvidence, StructuredState,
│                           StateTracker
│
├── policy.py           — all decision logic in one file:
│                           PersonalityBias, derive_bias, sample_hard_blockers,
│                           select_next_speakers, repetition_pressure,
│                           extract_discourse, TurnPlan, plan_turn
│
├── reasoning.py        — ConsensusEngine, PhaseDetector, fact_check,
│                           repair_directive
│
├── prompt_context.py   — compact speaker-card prompt builders:
│                           build_speaker_card, build_relevant_options,
│                           build_group_state, build_local_context,
│                           build_move_instruction, build_output_contract
│
├── orchestrator.py     — loop coordinator; wires all layers together
├── simulator.py        — generates one participant turn via LLM
├── moderation.py       — intervention timing + LLM-generated moderator lines
├── logger.py           — 3-file output per dialogue (.txt, .csv, _meta.json)
├── consensus.py        — ConsensusDetector (legacy regex + soft-language scan)
└── utils.py            — deterministic regex helpers (vote extraction etc.)

eval/
└── eval_scenarios.txt  — fixed topics across 6 decision domains (for batch runs)
```

## High-Level Dialogue Pipeline

For each topic:

1. `main.py` creates an `Orchestrator`.
2. `Orchestrator` asks the LLM to generate four concrete options (with example names where applicable) and an opening question.
3. If `simulation.use_llm_names: true`, a single LLM call generates participant names and roles tuned to the topic vibe (workplace topics get professional names, friend topics get casual names, etc.).
4. `PersonaBuilder` assigns Big Five traits (with optional stubbornness overrides), generates topic-specific backstories and goals, and produces private belief states over options A-D.
5. Each persona is wrapped by a `Simulator`.
6. `Orchestrator.run_simulation()` initialises `StateTracker`, samples the hard-blocker flag, and drives the dialogue through phases.
7. Each round: `select_next_speakers()` (SSJ cascade) picks who speaks; `plan_turn()` (ActPlanner) plans each turn; `Simulator.generate_turn()` calls the LLM.
8. `StateTracker.update()` parses each turn into a `TurnRecord` and updates `StructuredState`.
9. `PhaseDetector` recomputes `PhaseEvidence` from Fisher ratios (informational — orchestrator does not hard-branch on it).
10. `ConsensusEngine` recomputes `ConsensusState` from the `StanceTable` (public stances only).
11. `ConsensusDetector` (legacy) also checks agreement via regex + soft-language scan.
12. `ModerationEngine` decides if/when the moderator should intervene.
13. `DialogueLogger` writes all output.

## Dialogue Phases

- `greeting` — each participant says hello once (phase cap: 8 words)
- `opening` — each participant gives a first reaction or priority
- `negotiation` — participants compare options, answer each other, disagree, raise trade-offs
- `narrowing` — moderator asks for explicit preferred options (16-32 words; randomized prompts from `prompts.narrowing_lines()`)
- `emergence` — the system tests whether a compromise can become acceptable (cap: 34 words)
- `confirmation` — moderator asks for explicit yes/no confirmation (cap: 10 words)
- `closure` — participants say goodbye, sometimes acknowledging the final option

Fisher (1970) alignment: `PhaseDetector` tracks favorable/unfavorable/ambiguous ratio shifts over a rolling window (`phase_policy.window_size: 8`). `PhaseEvidence` stores phase confidence — phases are gradual, not hard switches. This signal is stored in `_meta.json` for analysis but the orchestrator uses turn-count-based phase transitions.

## Personas

Personas live in `src/persona.py`.

### Big Five personality model

`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`.

`response_length` is intentionally separate — it is not a Big Five trait. It sets word budget and communication register (default range: [1, 3], biasing toward short messages).

### Stubbornness — two independent knobs

| Config key | Default | Effect |
|---|---|---|
| `stubbornness.sim_stubborn_probability` | 0.05 | Per-sim Bernoulli: override traits with rigid combo (low agree + low openness + high consc + high neuro) |
| `stubbornness.hard_blocker_dialogue_probability` | 0.05 | Per-dialogue Bernoulli: one participant's act planner is biased toward REJECT_WITH_REASON when the candidate is in their `rejected` list |

95% of sims sample from cooperative defaults (`agreeableness [3,5]`, `openness [3,5]`) and tend to find compromise. The 5% stubborn exception overrides traits at persona-creation time via `_apply_stubbornness_distribution()` in `persona.py`.

### Beliefs

Each persona has private `AgentBeliefs`:

- `preferred` — top option before discussion
- `acceptable` — options they could live with (aim for 3 out of 4 options; `rejected` is typically empty)
- `rejected` — options they actively resist (normally empty)
- `key_concern` — the main reason behind their preference
- `concession` — condition under which they could accept a non-preferred option

**Belief consistency rule (enforced in the prompt):** The `key_concern` must be consistent with the `preferred` option. The `acceptable` list must include any option that meaningfully addresses the `key_concern` — aim for 2-3 options to represent realistic flexibility. Overly narrow acceptable lists cause artificial deadlock that no runtime heuristic can fix.

Private beliefs guide coherence but decisions must be visible in the transcript. The moderator cannot select an option purely because hidden beliefs say it is acceptable.

## Personality and Register

`personality_summary()` in `persona.py` produces **register descriptors only** — never named phrases or example sentences. Big Five traits are operationalised as probabilistic behavioral biases:

| Trait level | Register cue |
|---|---|
| High openness | considers angles others haven't raised; comfortable reframing the question |
| Low openness | prefers concrete options on the table; impatient with speculation |
| High extraversion | energetic, quick to react, thinks aloud |
| Low extraversion | reserved; speaks up only when there is something specific to add |
| High agreeableness | acknowledges before pushing back; seeks common ground |
| Low agreeableness | direct, skeptical, blunt; states disagreement plainly |
| High neuroticism | sensitive to uncertainty; concern and caution show through tone |
| Low neuroticism | calm and steady even when discussion stalls |

**Do not list specific filler phrases in prompts.** Any phrase named explicitly (e.g. "use 'honestly'") will be overused. Describe register and behavior instead.

`_STYLE_RULE` in `persona.py` maps `response_length` (1-5) to register descriptions with word budgets:

| Level | Budget | Register |
|-------|--------|----------|
| 1 | 14 | One punchy fragment or short sentence, cut everything non-essential |
| 2 | 22 | One clear point, sometimes with a short reason |
| 3 | 30 | One or two compact sentences, explain when needed not by default |
| 4 | 42 | Can explain a bit and riff, keep it breezy like a group chat |
| 5 | 55 | Give useful detail and context, never write an essay |

Phase caps override these budgets when they are stricter (greeting: 8, confirmation: 10, narrowing: 16-32, emergence: 34).

The `PersonalityBias` dataclass in `policy.py` maps the same trait values to float biases consumed by `plan_turn()` (act sampling weights) and `select_next_speakers()` (self-selection score). These floats never appear in prompts.

## Turn-Taking (SSJ)

`policy.py::select_next_speakers()` implements the Sacks/Schegloff/Jefferson (1974) priority cascade:

1. **Obligated addressees** of pending questions (`DiscourseGraph.pending_questions`)
2. **Open invitation** — questions without an explicit addressee force any non-asker to respond (`open_invitations: dict[int, str]` mapping turn_id to asker name)
3. **Recently name-mentioned participant** (no pending question)
4. **Personality-biased self-selection** (extraversion, participation debt, novelty)
5. Current speaker continues or moderator intervenes

`DiscourseGraph` persists pending questions until they are answered, not just for one turn. Phase-specific rules: greeting/opening cover everyone once before repeats; confirmation gives each participant one shot; closure: primary first.

## Act Planner

`policy.py::plan_turn()` selects a `TurnPlan` (planned `DialogueAct` + target option + addressee) before each LLM call.

Priority cascade:

1. Discourse obligation (pending question addressed to speaker -> ANSWER)
2. Phase obligation (greeting -> GREET; narrowing+no-vote -> COMMIT_VOTE; confirmation -> CONFIRM/REJECT_WITH_REASON)
3. Emergence soft-path (candidate in acceptable -> CONDITIONAL_ACCEPT)
4. Hard-blocker path (is_true_hard_blocker + candidate in rejected -> REJECT_WITH_REASON)
5. Personality-biased sampling with MUCA-style cooldowns

The `TurnPlan.to_prompt_str()` is prepended to the move instruction block — it tells the LLM what move to make without dictating exact phrasing.

**Position discipline is computed for all decision phases** (narrowing, emergence, confirmation), not just emergence. This prevents sims from incorrectly rejecting options that are in their `acceptable` list.

## Prompting

All LLM-facing templates live in `src/prompts.py`. No other module constructs LLM-facing prose.

### Compact speaker-card prompt (~400-600 tokens)

```
SPEAKER CARD     — name, role, register, private stance
SHARED FACTS     — only the 2-3 relevant options
GROUP STATE      — candidate, current votes, unresolved rejections
YOUR MOVE        — phase instruction + optional TurnPlan act description
RECENT TURNS     — last 4 turns
OUTPUT           — word budget + formatting rules
```

`prompt_context.py` builds each section from structured orchestrator state. No history re-scan: group state comes directly from `DialogueState` fields.

### Voice rules

"Friends chatting, not a panel. MIX LENGTHS naturally. Punctuation: use when it helps, not mandatory. Skipping final period is fine. Casing is your call. Filler words welcome ('tbh', 'ngl', 'lol'). No corporate-speak."

Output contract: "Hard cap: N words. Most turns should be shorter — use the full budget only when explaining something real."

### Moderator anti-impersonation

All 5 moderator prompts include: "Do NOT write any participant attribution like 'Name: ...' — you are the moderator, write only your own line."

`_clean_moderator_line(text, participant_names)` in `moderation.py` detects "Name:" prefix matching a participant name and returns "" to drop that line if the LLM produces it anyway.

### Option generation

The option-generation prompt requests concrete example names where applicable ("The Bellwether Theater", "Bella Vita Restaurant"). The LLM decides whether the topic warrants named examples.

### Narrowing prompts

`prompts.narrowing_lines()` returns 7 natural variants sampled randomly, e.g.: "ok so where's everyone landing?", "alright, what are we thinking?", "quick check — where's everyone at?".

### Name generation

`prompts.names_and_roles(topic, n)` generates N names + roles tuned to the topic vibe in a single LLM call. Result is passed to `PersonaBuilder.build_all(names, pre_role_map=...)` which skips the separate role LLM call.

### Forbidden openers

`_recent_openers()` in `simulator.py` tracks the last 6 participant turns and extracts the first 1-2 words of each as forbidden openers. Two-word phrases are tracked so repetitive openers are caught even when surface wording shifts slightly.

### Repetition detection (layered)

1. **Surface overlap** — Jaccard word overlap >= `cfg.repetition.jaccard_threshold_self` in last 2 own turns
2. **Semantic looping** — keywords of 4+ chars appearing in 2+ of last 5 own turns. Fires: "That point is on the table — don't repeat it."
3. **Speculative crutch** — "what if" appearing in 2+ of last 4 own turns fires: "Stop — make a direct claim instead."

### Goalpost detection

When a sim is the active rejecting speaker and has spoken `cfg.repetition.turns_since_rejection_escalation` turns since their rejection, the instruction escalates: "Name your single concrete dealbreaker for Option X, or say whether you can accept it with one specific condition."

## Grounding

`reasoning.py::fact_check()` deterministically flags numbers, quoted strings, and parenthesised asides in generated turns that have no basis in the option texts or topic.

When `cfg.grounding.enable_fact_check: true`:
- Turns shorter than `cfg.grounding.min_words_to_check` (12 words) are skipped.
- If suspicious claims are found and `cfg.grounding.repair_attempts >= 1`: regenerate once with `repair_directive()` appended.
- If repair fails or is disabled: log a warning and keep the original turn.

Fully deterministic — no LLM call for the check itself.

## Consensus and Phase Detection

### Legacy path (ConsensusDetector, active)

`consensus.py::ConsensusDetector` scans latest turns per speaker using:

1. **Soft** — agreement language scan across latest turns
2. **Regex** — explicit option-letter vote extraction with primary participant weighting
3. **Reduced opposition** — emergence phase only; checks dissenters' last 2 turns

### Structured path (ConsensusEngine, parallel)

`reasoning.py::ConsensusEngine` derives consensus state from `StanceTable` only — no private beliefs:
- States: `none` -> `candidate_emerging` -> `majority_candidate` -> `conditional_consensus` -> `full_consensus`
- Special states: `blocked` (hard blocker actively opposes) and `failed` (ceiling reached)

Stance weights configurable under `consensus.stance_weights`.

`reasoning.py::PhaseDetector` computes `PhaseEvidence` from Fisher ratios over the last `cfg.phase_policy.window_size` stance updates. Output stored in `_meta.json["structured"]`.

### Vote extraction hardening

`utils.py::extract_preference_vote()` guards against false positives:

- **Pre-negation**: "not / never / rather not / anything but / except / avoid / skip / hate / dislike / reject / refuse / nope to" BEFORE the option mention -> returns None
- **Post-negation**: "isn't / aren't / wasn't / might be / could be / won't / can't / seems / sounds bad|terrible|awful / sucks / nope" AFTER the option -> returns None
- **Non-committal nearby**: "either / maybe / perhaps / possibly / might work" near the option mention -> returns None

### Confirmation rejection

A "no" during confirmation stores the rejected option, sets `consensus_cooldown`, and routes priority to the rejecting speaker. The same option is not re-tested while cooldown is active.

## Compromise Testing and Force-Close

### Compromise flow

`_best_compromise_option()` scores options using private acceptability + vote counts minus rejections, restricted to options that received actual participant votes.

### Force-close

`_force_conclusion()` selects the final option by vote plurality:
- Each actual participant vote dominates private-belief scores and rejection penalties.
- Never picks an option with zero votes (falls back to mentioned options, then all options).
- When `ConsensusEngine` has a best-available decision, that is used and outcome label becomes `best_available_decision`.

## Moderator Behavior

`moderation.py::ModerationEngine` handles intervention timing and LLM-generated lines. `orchestrator.py` handles state transitions, confirmation, compromise, and force-close.

Good behavior:
- does not interrupt immediately after a fresh unanswered question
- asks the dissenter why they rejected an option
- tests compromise before force-closing
- asks for yes-with-condition or no-with-objection
- avoids repeatedly confirming the same rejected option

## Logging and Output

All outputs go to `logs/`. Three files per dialogue:

| File | Contents |
|---|---|
| `.txt` | Clean human-readable transcript + Outcome/Tokens footer |
| `.csv` | One row per turn: phase, speaker, text, selected_reason, tokens, persona traits |
| `_meta.json` | Everything else: dialogue metadata, outcome, token totals, Gini + speaker/phase turn counts, vote flips, confirmation rejections, compromise info, full personas with beliefs, and structured per-turn state (acts, addressees, stance updates, Fisher phase evidence) |

`.txt` footer format:
```
--- Outcome: <outcome> ---
--- Tokens : setup=<in>/<out>  dialogue=<in>/<out>  total=<in>/<out> (in/out) ---
```

`_meta.json["structured"]` is optional — only written when `StateTracker` is active.

## LLM Providers

`src/llm_client.py`. Supported: `uni` (Ollama-style), `groq` (OpenAI-compatible), `gemini` (Google GenAI). Provider/model configured in `config.yaml` under `llm.provider`.

## Key config.yaml Values

| Section | Key | Default | Notes |
|---|---|---|---|
| `simulation` | `use_llm_names` | `true` | LLM picks names + roles tuned to topic |
| `simulation` | `num_participants` | `3` | Default; randomizable |
| `turns` | `hard_ceiling` | `28` | Max total participant turns |
| `turns` | `escalation_level_2` | `4` | Stall turns before pressure escalates |
| `turns` | `escalation_level_3` | `7` | Further escalation |
| `response_length` | `word_budgets` | `[14, 22, 30, 42, 55]` | Per response_length level 1-5 |
| `response_length` | `phase_caps.greeting` | `8` | Hard cap for greeting turns |
| `response_length` | `phase_caps.confirmation` | `10` | Hard cap for confirmation turns |
| `personas` | `trait_ranges.agreeableness` | `[3, 5]` | Cooperative baseline |
| `personas` | `trait_ranges.openness` | `[3, 5]` | Open to alternatives |
| `personas` | `trait_ranges.response_length` | `[1, 3]` | Short-to-medium messages by default |
| `personas` | `diversity_agree_threshold` | `5` | Only nudge agreeableness down if everyone is at max |
| `stubbornness` | `sim_stubborn_probability` | `0.05` | 5% per-sim rigid trait override |
| `stubbornness` | `hard_blocker_dialogue_probability` | `0.05` | 5% per-dialogue hard-blocker flag |
| `grounding` | `min_words_to_check` | `12` | Skip very short turns in fact-check |

## Development Notes

Use `python -m compileall -q main.py src` after edits.

Prefer keeping changes scoped. The system has many interacting heuristics; small prompt changes can cause large behavior shifts.

**Do not list specific filler phrases in prompts.** Any phrase named explicitly will be overused. Describe register and behavior instead.

**Do not solve repetition with phrase blacklists alone.** Repetition is caused by missing dialogue state: the sim has no new role to play, no unanswered question, and no explicit compromise condition to evaluate. The right fix is to create a conversational obligation:
- answer this person
- explain this rejection
- test this compromise
- respond to this condition
- add a new objection
- name one drawback of your own preferred option

**Belief generation quality matters more than runtime fixes.** If a persona's `acceptable` list is too narrow or their `key_concern` contradicts their `preferred` option, no runtime heuristic will fix the downstream deadlock. Fix the belief prompt before adding more concession logic.

**Vote plurality must dominate force-close.** When participants have cast explicit votes, those votes are the ground truth. Private belief scores and rejection penalties must not override a clear vote majority.

**Private beliefs never enter the structured force-close path.** `ConsensusEngine.best_available_decision()` uses public stances only.

**No forced concession cap.** A hard cap ("after N holdout turns, force concession") produces fake consensus. The correct mechanism is the rare hard-blocker sampler (p=0.05) plus condition-based softening.

**Position discipline must cover all decision phases.** Computing the position-discipline candidate only during `emergence` causes sims to incorrectly reject acceptable options during `narrowing` and `confirmation`. Always compute it for narrowing, emergence, and confirmation.
