# CLAUDE.md

## Project overview

This project is a university dialogue-simulation system for generating small
multi-party group discussions. Given a short decision topic, the system creates
several simulated participants ("Sims"), generates a small set of options, lets
the Sims discuss those options, and tries to reach a valid group outcome.

The goal is **not** to build a fully general social simulator. The goal is a
bounded and useful dialogue generator:

- Sims have stable preferences and reasons.
- Sims sound different from each other.
- Sims can answer briefly instead of explaining every turn.
- Sims avoid repeating the same point.
- Sims usually try to compromise.
- Rare failed or force-closed dialogues are allowed when no viable compromise
  exists.
- Final outcomes must be honest: a force-close must not be presented as full
  consensus.

The current implementation is a simplified control-cleanup version of the
earlier architecture. The project still keeps useful literature-grounded ideas
such as turn-taking, addressee handling, persona scaffolding, compact memory,
argument kits, and evaluation metrics. However, the live decision path is now
much more explicit: **votes, acceptances, rejections, and private acceptability
drive the outcome**, not broad stance inference or challenge-gated deliberation.

---

## Main design principle

The current architecture follows this separation:

| Layer | Responsibility |
|---|---|
| Persona | Stable preferences, traits, voice, response-length tendency, flexibility |
| Prompt | Ask the LLM for one local next message |
| Policy | Decide who speaks next |
| Verifier | Check whether the generated message is valid and non-repetitive |
| State tracker | Extract structured facts for routing, memory, logging, and evaluation |
| Moderator | Move the group through the decision process |
| Consensus/control | Use explicit votes/acceptances/rejections to decide outcomes |
| Evaluation | Measure quality after the dialogue |

Important boundary:

> Evaluation structures may observe rich discourse behaviour, but they should
> not be the main live-control path.

This prevents the project from becoming a large theory-driven control machine
where every utterance has to satisfy many abstract dialogue criteria.

---

## Research grounding

The project is still literature-grounded, but the paper-to-code mapping has
been trimmed. The system no longer tries to implement every cited theory as a
runtime controller.

### Load-bearing literature

#### Sacks, Schegloff & Jefferson (1974): turn-taking

Used for the basic idea that turns are locally organized and that the next
speaker depends on address, prior speaker, and self-selection opportunities.

Current implementation role:

- `policy.select_next_speakers()`
- one speaker is selected per participant turn;
- directed questions have priority;
- repeated immediate self-selection is discouraged;
- participation balance and initiative are used only as light routing signals.

The code does **not** attempt to fully model conversation analysis. It keeps only
the practical routing idea.

#### Ouchi & Tsuboi (2016): addressee / response selection

Used for the multi-party problem of deciding **who should respond**.

Current implementation role:

- `policy.extract_discourse()`
- `state.DiscourseGraph`
- `StateTracker._extract_addressees()`
- `StateTracker._implicit_previous_speaker_target()`

The implementation now distinguishes:

1. explicit name/address target;
2. direct question target;
3. implicit target from the previous speaker's priority/keyword;
4. open invitation.

Open participant questions no longer create hard obligations for arbitrary
non-askers. This was changed because it produced question-answer-question-answer
loops. If a participant asks about a keyword introduced by the previous speaker,
the previous speaker should answer.

Example:

```text
Julian: Hey, hoping for an eco-friendly spot.
Liam: How important is being eco-friendly to our choice?
```

The next likely speaker should be Julian.

#### Toulmin (1958): compact argument kit

Used for the private belief model of each Sim.

Current implementation role:

- `persona.AgentBeliefs`
- `prompts.agent_beliefs_group()`
- `prompt_context.build_speaker_card()`

Each Sim receives:

```text
preferred
acceptable
rejected
key_concern
reasons
reservation
would_reconsider_if
```

This is what gives a Sim material to discuss beyond simply restating "I prefer
A". The implementation is deliberately compact: one or two reasons, one honest
reservation, and one condition under which the Sim might compromise.

#### McCrae & John (1992): Big Five as persona scaffold

Used to create varied but bounded personalities.

Current implementation role:

- `persona.Persona`
- `persona.derive_speech_signature()`
- `Persona.derived_controls()`
- `Persona.derived_controls_descriptor()`
- `Persona.response_length_score()`

The system keeps Big Five traits plus `response_length`:

```text
openness
conscientiousness
extraversion
agreeableness
neuroticism
response_length
```

These traits are used to derive simpler conversational controls such as:

```text
initiative
flexibility
directness
detail_level
warmth
```

Runtime code should prefer these derived controls over raw Big Five values when
possible. Traits shape voice, length, initiative, directness, and flexibility.
They should **not** become a large dialogue-act planner.

#### Shanahan (2023): role-play / persona scaffolding for LLMs

Used for the idea that persona instructions should be structural and external,
not theatrical "be this character" prompts.

Current implementation role:

- `persona.SpeechSignature`
- `prompt_context.build_speaker_card()`

The speaker card describes role, style, stance, reasons, and tone tendencies.
It does not ask the LLM to perform a dramatic character. The goal is distinct
but ordinary chat behaviour.

#### Park et al. (2023): Generative Agents, scaled down

Used only for compact memory and anti-repetition.

Current implementation role:

- `state.ParticipantState.points_made`
- `prompt_context.build_memory_block()`
- `verifier.detect_self_repetition()`
- `verifier.detect_semantic_point_repeat()`

The project does **not** implement full generative-agent memory, reflection, or
planning. It keeps only the useful minimum:

- last own turn;
- recent point signatures;
- recent local dialogue context;
- stated priority;
- enough memory to avoid repeating the same point.

### Demoted or evaluation-only literature

#### Fisher (1970): decision emergence

Fisher-style ratios are evaluation-only.

Current implementation role:

- `reasoning.fisher_ratios()`
- `.eval.json` output

Fisher does not drive live phase transitions.

#### Deliberative quality frameworks

Justification, reciprocity, and reflexivity are useful for later analysis.

Current implementation role:

- `reasoning.deliberation_metrics()`
- `reasoning.evaluation_summary()`
- `.eval.json` output

They are not live-control gates.

#### MUCA / multi-user strategy cooldowns

Earlier versions used strategy cooldowns and richer act planning to prevent
repetition. This has been demoted.

Current implementation:

- no heavy live dialogue-act planner;
- lightweight surface moves may be used to vary message form;
- repetition is handled mainly by memory + verifier + repair.

#### Liang / Du style divergent debate

The project keeps only the practical idea that participants should not all start
from exactly the same preference. It does **not** force formal debate or require
challenge-response pairs before progress.

Current implementation role:

- `persona._enforce_divergence()`
- `persona._enforce_acceptable_overlap()`

Divergence should create useful starting tension, not artificial conflict.

---

## Entry points

`main.py` is the executable entry point.

```bash
python main.py
python main.py scenarios.txt
```

Modes:

- interactive mode: prompts for one topic;
- batch mode: reads one topic per non-comment line from a scenario file.

`main.py` performs the high-level setup:

1. create `Orchestrator`;
2. create `PersonaBuilder`;
3. generate names and roles;
4. build personas;
5. assign private belief states;
6. wrap each persona in `Simulator`;
7. run the dialogue;
8. write logs.

---

## Source layout

All project files are flat under `src/`.

```text
src/
├── config_loader.py
├── llm_client.py
├── logger.py
├── moderation.py
├── orchestrator.py
├── persona.py
├── policy.py
├── prompt_context.py
├── prompts.py
├── reasoning.py
├── simulator.py
├── state.py
├── utils.py
└── verifier.py
```

The flat layout is intentional. This is still a university project, so the code
should remain easy to inspect without a deep package hierarchy.

---

## Module responsibilities

### `config_loader.py`

Loads `config.yaml` once and exposes a typed `cfg` object.

All modules import config from here.

Important:

- no module should parse YAML directly;
- missing config values should be handled carefully only where needed;
- sections declared in `Config` should match the actual YAML.

### `llm_client.py`

Thin LLM provider abstraction.

Supported providers:

```text
gemini
groq
uni
```

Exposes:

```text
generate(prompt: str) -> str
generate_json(prompt: str) -> dict
reset_session()
```

It also tracks token usage:

```text
last_tokens_in
last_tokens_out
session_tokens_in
session_tokens_out
```

Setup tokens and dialogue tokens are separated by `reset_session()` in `main.py`.

### `persona.py`

Owns persona and belief generation.

Important classes:

```text
AgentBeliefs
SpeechSignature
Persona
PersonaBuilder
```

Important functions:

```text
derive_speech_signature()
_enforce_diversity()
_enforce_divergence()
_enforce_acceptable_overlap()
_random_traits()
```

The persona system creates:

- names and roles;
- Big Five traits;
- response-length tendency;
- derived conversational controls;
- compact goals/backstories;
- private option beliefs.

The current system still uses LLM calls for names/roles, persona concepts, and
beliefs, but enforces structural constraints in Python.

### `orchestrator.py`

Coordinates a single dialogue.

Responsibilities:

1. generate topic options and opening question;
2. create and update `DialogueState`;
3. run bounded phases;
4. store participant and moderator lines;
5. update explicit votes/acceptances/rejections;
6. select compromise candidates;
7. finalize success, compromise, force-close, or failure;
8. pass data to the logger.

Important methods:

```text
_run_opening()
_run_discussion()
_run_vote_round()
_run_compromise()
_ask_holdout()
_finalize_success()
_finalize_force_or_failure()
_run_closure()
```

The live outcome path is controlled by explicit data, not by broad inferred
stance labels.

### `policy.py`

Handles speaker selection and lightweight discourse extraction.

Important functions:

```text
select_next_speakers()
extract_discourse()
repetition_pressure()
sample_hard_blocker()
```

`policy.py` should **not** become a dialogue-act planner. Its main job is to
decide who speaks, not what they say.

Routing priorities:

1. opening participants who have not spoken;
2. directed pending-question target;
3. implicit previous-speaker question target;
4. recent explicit addressee;
5. participation balance and initiative;
6. avoid repeated same speaker unless required.

### `simulator.py`

Wraps one `Persona` and generates one participant turn.

Generation flow:

```text
build prompt
call LLM
strip accidental name prefix
verify output
repair once if needed
enforce word budget
return text + token counts
```

Important methods:

```text
generate_turn()
_generate_raw()
_verify_and_repair()
_build_repair_prompt()
_deterministic_fallback()
_enforce_word_budget()
```

`Simulator` should generate exactly one message from exactly one speaker.

### `verifier.py`

Deterministic post-generation validation.

It does not call the LLM.

Participant checks include:

```text
EMPTY_OR_SILENCE
NAME_PREFIX
INVALID_OPTION_REFERENCE
VALID_OPTION_DENIED
INVENTED_OPTION_FACT
SELF_REPETITION
ACK_LOOP
SEMANTIC_POINT_REPEAT
FACT_CHASING_QUESTION
MISSING_EXPLICIT_VOTE
UNCLEAR_CONFIRMATION
```

Moderator checks include:

```text
MODERATOR_NEW_OPTION
MODERATOR_MIXED_SOLUTION
FAKE_CONSENSUS
```

If repair is needed, `simulator.py` performs one repair attempt with a targeted
repair prompt. The verifier prevents common failures, but it should not become
a full evaluator or another reasoning engine.

### `prompt_context.py`

Assembles prompt sections from structured state.

Important functions:

```text
build_speaker_card()
build_relevant_options()
build_group_state()
build_memory_block()
pick_surface_move_kind()
build_local_context()
build_move_instruction()
build_output_contract()
```

Despite the historical name `build_relevant_options()`, the function now returns
**all options**. Option filtering was removed because it caused participants to
deny valid options.

### `prompts.py`

Single registry of LLM-facing prose.

Contains:

- phase instructions;
- interaction instructions;
- position-discipline instructions;
- surface-move hints;
- setup prompts;
- simulation prompts;
- moderator prompts;
- repair prompts.

Important functions:

```text
option_generation()
names_and_roles()
persona_group_generation()
agent_beliefs_group()
sim_turn_compact()
surface_move_hint()
repair_repetition()
repair_ack_loop()
repair_semantic_repeat()
repair_invalid_option()
repair_vote()
repair_confirmation()
repair_invented_fact()
moderator_ask_holdout()
moderator_force_close()
```

Prompting should avoid forcing every turn into:

```text
acknowledge previous point -> name option -> state pro/con -> restate preference
```

Surface moves are only lightweight form hints. They are not a return to a heavy
dialogue-act planner.

`option_generation()` now creates self-contained fictional scenario facts. For
logistics topics it may generate concrete values such as prices, durations,
walking times, difficulty scores, wait estimates, or comfort ratings. For
abstract topics it uses scored dimensions such as policy relevance, clarity,
hands-on potential, and scope difficulty. These values are scenario facts only;
they are not real-time claims. The purpose is to give Sims enough grounded
material to discuss without asking fake lookup questions.

### `state.py`

Tracks structured dialogue information.

Still contains rich structures from earlier versions:

```text
DialogueAct
StanceUpdate
StanceTable
OptionState
DiscourseGraph
ChallengeRecord
ParticipantState
StructuredState
StateTracker
```

Important current distinction:

- these structures are useful for logging, context, and evaluation;
- they are **not** the main live consensus authority.

The live decision path in `orchestrator.py` uses explicit votes/acceptances and
private acceptability.

`StateTracker` still extracts useful facts:

- mentioned options;
- explicit votes;
- addressees;
- question targets;
- stance-like signals for evaluation;
- point signatures;
- stated priorities.

### `reasoning.py`

Contains consensus helpers, fact-checking, and evaluation metrics.

Important functions/classes:

```text
ConsensusEngine
fisher_ratios()
deliberation_metrics()
fact_check()
repair_directive()
evaluation_summary()
```

Current rule:

- `fact_check()` remains runtime-relevant for option-grounding and repairs;
- Fisher and deliberation metrics are evaluation-only;
- `ConsensusEngine` may still compute context/evaluation states, but final live
  decisions should rely on explicit control state in `orchestrator.py`.

### `moderation.py`

Controls moderator interventions and moderator text.

Important methods:

```text
should_narrow()
should_intervene()
detect_info_chase()
detect_facilitatable_disagreement()
run_intervention()
_detect_outlier()
```

The moderator is constrained by `verifier.verify_moderator_turn()`.

Allowed moderator behaviour:

- introduce the task;
- ask for priorities;
- ask for explicit votes;
- ask a named holdout whether they can live with a candidate;
- reframe missing information into a judgment call;
- announce success, compromise, force-close, or failure honestly.

Forbidden moderator behaviour:

- invent new options;
- combine options unless config explicitly allows it;
- claim consensus before enough acceptance exists;
- force-close while a targeted compromise path remains.

### `utils.py`

Deterministic helper module.

Main class:

```text
OptionResolver
```

Important functions:

```text
options_referenced()
participant_turn_count()
last_n_turns_for()
current_votes()
```

`OptionResolver` maps natural option mentions to option labels. It supports:

- `Option A`;
- bare letters such as `A` or `B` in vote/acceptance contexts;
- aliases from option titles;
- vote extraction.

It is intentionally not a full semantic stance parser.

### `logger.py`

Writes two outputs per dialogue:

```text
.txt
.eval.json
```

The transcript file contains the readable chat and persona block.

The eval JSON contains structured metadata, including:

- topic;
- outcome;
- final option;
- tokens;
- participants;
- traits;
- beliefs;
- turns;
- verification results;
- explicit votes;
- explicit accepts;
- explicit rejects;
- outcome reason;
- evaluation metrics.

The logger may still include older rich traces for analysis. Those traces are
not live-control instructions. `outcome_valid` is based on the explicit live
control state, not on the old `StanceTable` consensus label.

---

## Runtime flow

The intended current loop is:

```text
setup
opening
negotiation/discussion
narrowing/vote
targeted compromise
finalize
optional closure
logging
```

More concretely:

1. `Orchestrator` generates four options and an opening question.
2. `PersonaBuilder` creates participants and private beliefs.
3. The moderator introduces the topic and options.
4. Each participant states a priority in opening.
5. The group discusses for a bounded number of turns.
6. The moderator asks where everyone is landing.
7. Each participant gives an explicit vote.
8. If votes match, finalize success.
9. If votes split, choose a candidate from votes and/or private acceptability.
10. Ask named holdouts whether they can live with the candidate.
11. If every participant voted for or explicitly accepted the candidate, finalize
    `compromise_success`.
12. If no candidate can be accepted and the budget is exhausted, force-close or
    fail honestly.
13. Write transcript and evaluation output.

The most important runtime improvement is that compromise can close immediately
once explicit acceptance exists. The system should not continue talking after a
valid compromise has already been reached.

---

## Live decision model

The live decision model is explicit.

Important fields in `DialogueState`:

```text
explicit_votes
explicit_accepts
explicit_rejects
candidate_option
preferred_option
outcome_reason
pending_confirmation_target
pending_confirmation_candidate
required_reason_target
```

Examples:

```text
"I'd go with Option B."       -> explicit vote for B
"I can live with B."          -> explicit acceptance of B
"Not sold on B."              -> explicit rejection/soft rejection of B
"No, B doesn't work for me."  -> explicit rejection of B
```

A candidate is accepted when every participant has either:

- voted for it; or
- explicitly accepted it.

This is the key rule behind `compromise_success`.

Short confirmation replies are bound to moderator context. If the moderator asks
"Léa, could you live with Option C?", then "that's fine", "yeah", or
"I'd still prefer A, but yeah" is recorded as Léa accepting C. "Not really"
or "no" is recorded as rejecting C. This avoids the earlier bug where
context-free parsing ignored bare confirmation answers.

Before voting, the orchestrator can request one concrete option-specific reason
from participants who have not yet contributed one. This prevents dialogues from
narrowing after only logistics questions or shallow rule-out fragments.

Private acceptability is used to choose a plausible candidate to test, not to
pretend that the public transcript already contains agreement.

---

## Question routing

Questions are allowed, but they must not dominate the chat.

Earlier versions allowed too many random participant questions, which caused
chains such as:

```text
question -> answer -> question -> answer -> question -> answer
```

Current policy:

- random question surface moves are disabled or strongly reduced;
- open participant questions do not create hard obligations for arbitrary
  non-askers;
- direct questions still route to the addressed speaker;
- implicit previous-speaker questions route to the previous speaker if the
  question repeats their priority/keyword.

This preserves useful question-answer behaviour without turning the discussion
into an interrogation sequence.

---

## Surface moves

Surface moves are lightweight hints for local style variation.

They are meant to create outputs such as:

```text
Yeah.
Not sold.
That might work.
Can we rule that one out?
I still prefer C, but I can live with B.
Then we're basically between A and D.
```

They are **not** a heavy dialogue-act planner.

Typical surface moves:

```text
ACK_ONLY
SHORT_NO
ANSWER
NEW_REASON
PUSHBACK
COMPROMISE
DECISION_MOVE
QUESTION
```

Questions should be rare in normal discussion. Compromise and decision-move
turns are especially important because they let the chat progress.

---

## Verifier and repair philosophy

The verifier exists because prompting alone is not enough.

It catches common failures after generation:

- invalid options;
- option denial;
- invented option facts;
- fact-chasing questions about live availability, waitlists, exact schedules, or things someone would need to call/check;
- repeated points;
- acknowledgement loops;
- missing votes;
- unclear confirmation;
- fake moderator consensus.

Repair is limited to one attempt. The system should not enter repair loops.
If a phase-critical message still fails, the simulator can use a deterministic
fallback.

The verifier is a correctness and quality guardrail, not a replacement for
good state design.

---

## Moderator philosophy

The moderator helps the group finish.

It should not be an overcreative participant.

Good moderator behaviour:

```text
Okay, where is everyone landing?
Luna, you picked C. Could you live with B, or is that a no?
So B works for everyone as a compromise. Going with B.
```

Bad moderator behaviour:

```text
Let's combine B and C.
Maybe choose B but add features from D.
No consensus, but I'm calling this agreement.
```

Force-close is allowed, but it must be honest.

---

## Logging and evaluation

The project is intended to generate many dialogues, so logging matters.

The `.txt` file is for human inspection.

The `.eval.json` file is for systematic analysis.

Important evaluation dimensions:

- valid final option;
- outcome type;
- token cost;
- turn counts;
- participation balance;
- explicit votes;
- explicit accepts/rejects;
- repair attempts;
- failed repairs;
- self-repetition;
- acknowledgement loops;
- response-length fit;
- persona consistency;
- compromise plausibility.

Later, an external LLM can evaluate transcript quality using the saved persona
and belief metadata. That should remain offline evaluation, not part of the live
generation loop.

---

## Expected outcomes

Possible outcomes:

```text
success
compromise_success
force_close
failed_no_viable_compromise
```

Definitions:

- `success`: everyone explicitly voted for the same option.
- `compromise_success`: everyone voted for or explicitly accepted the same
  option.
- `force_close`: no full acceptance was reached, but the moderator selected the
  best available option after the discussion budget was exhausted.
- `failed_no_viable_compromise`: no honest compromise exists, usually because of
  explicit rejection or a rare true hard-blocker case.

`force_close` must never be presented as consensus.

---

## Current known priorities

The code is now closer to the intended architecture, but the following areas
should remain under observation:

1. **Compromise parsing**
   - Ensure lines like "I can live with B" and "B works for me" are detected.
   - Ensure soft rejection like "not sold on B" is not counted as acceptance.

2. **Question frequency**
   - Keep participant questions rare.
   - Directed and implicit questions should route to the right speaker.

3. **Option generation**
   - Options should be rich enough to discuss and may include concrete
     fictional scenario values when the topic supports them.
   - Flights, hotels, restaurants, hiking trips, and similar logistics topics
     should usually include a few stable attributes such as price, duration,
     travel time, wait estimate, difficulty, comfort, or flexibility.
   - Abstract topics should use scored dimensions instead of fake logistics.
   - Values are scenario facts, not real-time claims. Do not generate fields
     that invite live checking such as "availability unknown", "waitlist
     unknown", "exact refund policy unclear", or "schedule uncertain".

4. **Closure length**
   - Do not force every participant to produce a closing line if the decision is
     already clear.
   - After `force_close` or `failed_no_viable_compromise`, the moderator's
     terminal line should usually be the end. Participant closings can imply
     agreement that does not exist.

5. **Old state structures**
   - `StanceTable`, `DialogueAct`, and `ChallengeRecord` are still present for
     logging/evaluation compatibility.
   - They should not regain control over the live outcome path.

6. **Documentation alignment**
   - If runtime control changes, update this file.
   - This document should describe what the code actually does, not the older
     planned architecture.

---

## What not to reintroduce

Do not reintroduce:

- a large dialogue-act planner;
- challenge-gated narrowing;
- stance-table-only consensus;
- heavy reflection memory;
- unlimited moderator creativity;
- random question-heavy surface moves;
- new literature layers before the explicit control loop is stable.

The intended system is not "maximally theoretical". It is a bounded, evaluable,
literature-informed simulator that produces valid and reasonably natural group
decision chats.


## Round 4 control refinements

The current version adds several dialogue-control refinements:

- The vote round now collects **fresh narrowing-phase votes from every participant**. Discussion-phase support no longer counts as a final vote.
- Narrowing is delayed by a lightweight discussion-readiness gate. This is not a checklist over all options; one issue may take several turns. The gate requires enough participant turns, at least one concrete option-linked reason from each participant, at least two substantively discussed options, and no active local thread that still needs an answer.
- The moderator now gives a short rationale before asking a holdout about a compromise candidate instead of always using the same mechanical line.
- Non-preferred compromise confirmations should include one short reason. Bare lines such as "That's fine" are repaired when the candidate is not the speaker's preferred option.
- Restaurant option cards now may include `allergen_safety_1_5` and `local_business_1_5`, so safety/local-business concerns can be discussed from scenario facts instead of fake call-ahead questions.
- Fact-chasing detection now blocks more lookup-like moves, including call-ahead suggestions, rough external cost estimates, allergen-protocol lookup questions, and missing-budget questions.
- Opening option cards are displayed in a more readable form in the transcript while the structured attribute form remains available internally to the LLM.

---

## Round 5 control fixes

The latest runtime patch focuses on the remaining consensus and question-chain
failures observed after the Round 4 discussion-control update.

### Consensus / compromise selection

The live compromise path now ranks candidates with an explicit candidate order:

```text
fresh vote count
explicit acceptance count
private acceptability count
primary-speaker acceptability
private rejection penalty
```

If a candidate is explicitly rejected during confirmation, it is excluded from
later fallback selection. This prevents the earlier bug where the moderator
asked holdouts about Option B, received acceptance, but then recomputed Option A
as the closest option and failed the dialogue.

Confirmation parsing now recognizes compromise phrases such as:

```text
I still prefer Option A, but Option B works well enough.
Option B works as a compromise.
Option B is acceptable.
```

These are bound to the current pending confirmation candidate and recorded as
explicit acceptance.

### Moderator holdout wording

The moderator no longer repeats the misleading phrase "has the most current
support" for every tested fallback. Holdout prompts now use more neutral wording
such as:

```text
Option B is worth testing as the current candidate.
Option B looks like the best shared fallback from what everyone said.
```

This avoids promising that a candidate already has consensus before holdouts
have answered.

### Question-chain repair

A new verifier issue `QUESTION_CHAIN` repairs messages that ask another question
when recent participant turns are already question-heavy. The repair prompt asks
for a short answer, reaction, comparison, or decision move instead of another
question.

Questions are still allowed, but the system should avoid runs like:

```text
question -> question -> question -> abstract discussion -> another question
```

The intended pattern is closer to:

```text
question -> short answer/reaction -> concrete comparison or decision move
```

### Good-question acknowledgement loop

The anti-acknowledgement rules now also cover "good question" openings. These
were starting to replace the earlier "valid point" loop.


---

## Round 6 local coherence and shared-compromise fixes

The Round 6 patch addresses the remaining local coherence problems found after
Round 5.

### Local self-consistency

A participant who has explicitly rejected or ruled out an option should not vote
for that same option later unless they explicitly say they changed their mind.
The simulator now repairs narrowing votes such as:

```text
Earlier: Given the layover, I'd rule out Option B.
Later:   I'd go with Option B.
```

Valid alternatives are:

```text
I'd go with Option A.
```

or, if the speaker genuinely changes position:

```text
I know I ruled out B earlier, but I've changed my mind because ...
```

The live state now treats clear non-question rule-out statements as explicit
rejections. Question-shaped exploratory lines such as "Can we rule out A?" are
not automatically treated as final rejection.

### Peer compromise before moderator holdout sequence

When votes split, the moderator no longer immediately performs a repeated
holdout interrogation. The system first gives participants a short opportunity
to surface a shared fallback themselves:

```text
Votes are split. Before I check people one by one, does anyone see a compromise everyone could live with?
```

One or two Sims may then propose or accept a compromise. The moderator still
verifies explicit acceptance afterward, but compromise is no longer entirely
moderator-imposed.

### Less repetitive holdout prompts

Holdout prompts now vary by candidate. The first prompt gives one short
rationale. Later prompts for the same candidate use a shorter "same fallback"
form instead of repeating the full explanation.

Example first prompt:

```text
Option A is a possible fallback from the votes. Liam, you picked Option B; could Option A work for you as a compromise, or is it a no?
```

Example follow-up:

```text
Noah, same fallback: you picked Option C. Could Option A work for you too, or is it a no?
```

### Candidate-specific final rationale

Final compromise explanations now describe actual support for the final
candidate only. The moderator should no longer finalize Option A by mentioning
an unrelated rejection of Option D.

Expected final form:

```text
Option A is the shared fallback: Ava picked it, and Liam, Noah can live with it. Compromise works -- Option A.
```

### Stronger question handling

Open participant questions are now treated as answer-worthy by the next selected
speaker, even when the question is not explicitly addressed by name. This nudges
turns toward short answers or reactions instead of another broad question.

`QUESTION_CHAIN` detection is also stricter: another participant question is
repaired if a recent participant question is still active. The repair prompt
asks for a short answer, concrete comparison, or compromise move.

### Option attribute mismatch detection

The verifier now catches direct changes to listed scenario facts. Example:

```text
Option B says: departs 14:00
Generated: the 3 pm departure for Option B
```

This now raises `OPTION_ATTRIBUTE_MISMATCH` and is repaired. The same check also
covers obvious price mismatches when a listed price exists.

### Rule-out repetition

Repeated "rule out X" moves are repaired when an option was already rejected or
ruled out. This reduces pruning-tree conversations and pushes later turns toward
comparison, compromise, or a concrete decision move.

## Round 7 conditional compromise update

The system now distinguishes between three compromise types:

1. A plain single-option decision, e.g. `Option C`.
2. A single option with execution terms, e.g. `Option C, if we split it over two nights`.
3. A true multi-option/hybrid plan, which remains disallowed unless the task explicitly supports multi-step planning.

Runtime compromise state now includes:

```text
compromise_terms: dict[option, list[str]]
confirmation_rejected_options: set[option]
```

Important control changes:

- Discussion-phase objections such as `not sold on C` no longer hard-exclude a candidate. Only a targeted confirmation rejection does that.
- The moderator tests the best public candidate first, especially if it has majority support, and may include emerged terms like `split it over two nights`, `add a brief intro`, or `go early`.
- Peer compromise turns now try to make the current candidate work before drifting to unrelated fallbacks.
- Final compromise rationale mentions the actual final candidate, who voted for it, who accepted it, and any accepted terms.
- A participant may not vote for an option outside their preferred/acceptable set unless the line explicitly marks a change of mind.

The final decision is still one of A-D. Conditional terms are attached to that option and logged, but they do not create Option E.

