# PLAN.md — Refactor Plan for the Group Discussion Simulator

## 0. Purpose

This project should generate small, casual, human-like group discussions for a given decision topic. The group contains 2–7 simulated participants. The moderator presents exactly four grounded options. The participants discuss trade-offs, reveal preferences, react to each other, and try to reach a compromise. Failure to reach consensus is allowed, but it must be rare and explainable, not the default outcome.

The simulator should not try to imitate all findings from the literature as separate modules. The literature should guide the design constraints:

- MUCA: multi-user chat requires deciding **what** to say, **when** to speak, and **who** to answer.
- Sacks/Schegloff/Jefferson: turn-taking should be locally managed, not round-robin or arbitrary.
- Ouchi/Tsuboi: addressee and response selection matter in multi-party dialogue.
- Clark/Brennan: final agreement requires grounding, not just a last message that says “sounds good”.
- Generative Agents / SOTOPIA: agents need persistent private state and social-goal evaluation, but this project should implement only the minimal useful subset.
- DAUS and recent user-simulation work: hallucination, goal inconsistency, repetition, over-cooperation, and homogeneity are core failure modes.

The target architecture is therefore a controlled simulation loop: the LLM generates options, personas, belief states, and one next utterance at a time. Deterministic code owns state, routing, validation, phase control, and consensus.

## 1. Strict assessment of the current code

The current code is not worthless. It already contains several correct ideas: one-turn generation, explicit votes/acceptance, compact memory, option grounding, deterministic verification, logging, and rare hard-blocker sampling. The problem is that these ideas are spread across too many places and implemented as accumulated patches rather than a clean model.

The largest current issue is architectural drift. `orchestrator.py` owns option generation, option display, transcript storage, public state updates, regex-based vote parsing, candidate ranking, discussion readiness, voting, compromise testing, moderator text, and finalization. This makes the behavior hard to reason about and hard to improve. When a chat fails, it is unclear whether the failure comes from routing, persona setup, phase logic, verifier logic, prompt wording, or candidate selection.

The second major issue is duplicated interpretation logic. Option references, votes, acceptances, rejections, changed-mind markers, and confirmation signals are parsed in multiple files: `orchestrator.py`, `simulator.py`, `utils.py`, and `verifier.py`. This creates inconsistent state updates. A message can be treated as a vote in one place, a weak confirmation in another place, and a non-commitment somewhere else.

The third major issue is that the code still contains a fixed phase pipeline: opening, negotiation, vote round, compromise checks, closure. The discussion phase has adaptive conditions, but it still relies on per-participant turn budgets, a repetition threshold, minimum option coverage, and a forced vote round. This is not enough for varying topic complexity. Simple topics may drag; complex topics may narrow too early; highly active threads may be cut off once budgets are reached.

The fourth major issue is the stubbornness model. A hard blocker is sampled rarely, which is good. But ordinary participants can still become functionally stubborn because belief states contain `rejected` options, `acceptable` sets are small, and `_must_reject_candidate()` rejects any candidate not in the acceptable list. That means “not pre-approved as acceptable” becomes “must reject”, which is too strict for cooperative group decision-making. Normal sims should usually be compromise-seeking. Only a rare hard-blocker should truly refuse most alternatives.

The fifth major issue is persona inconsistency. `_enforce_divergence()` may change a persona’s preferred option after the LLM generated reasons, reservations, and reconsideration conditions. That can make the final private belief state internally inconsistent: the preferred option changes, but the reasons may still support the old option. `_enforce_acceptable_overlap()` may also add shared acceptable options without generating a reason why the participant could live with them.

The sixth major issue is that “all prompts sent to the LLM must lie in `prompts.py`” is currently violated in spirit. `prompt_context.py` contains LLM-facing prose such as the speaker card labels and the output contract. Even if these are helper functions, their strings are still part of prompts sent to the model. Either all such prose must move to `prompts.py`, or `prompt_context.py` must become a pure formatter with no instructional text.

The seventh major issue is overfitted deterministic regex logic. The verifier catches many past failures, but it has become a large collection of special cases. This can help in the short term, but it is fragile. The project needs fewer, clearer validation categories: structural validity, grounding validity, turn-taking validity, persona consistency, naturalness/repetition, and decision validity.

The eighth issue is that options are represented as formatted strings. The system repeatedly parses strings like `Option A - Name: attrs: ...; upside: ...`. This is fragile. Option cards should be structured objects internally and rendered to text only when constructing prompts or transcripts.

The ninth issue is that moderator behavior is still too procedural. The moderator currently triggers a formal vote round and then directly checks holdouts. That can produce unnatural “committee meeting” dialogue. Friends do not always explicitly vote and then confirm one by one. A vote round should exist, but only as one possible closure strategy.

The tenth issue is insufficient evaluation. The logger reports useful summaries, but it does not yet measure the failure modes that matter most: addressed-question correctness, option coverage before narrowing, sim-driven compromise proposals, final consensus validity, force-close causes, repeated moderator interventions, and whether a rejection came from a true hard blocker or from ordinary strict belief logic.

## 2. Parts that are wrong and should be removed

Remove the current “patch accumulation” approach. Do not add another layer of if-statements to `orchestrator.py`. The file should be split into smaller modules.

Remove duplicated vote/accept/reject parsing. There should be one canonical `act_parser.py` or `state_tracker.py` that converts a raw message into a `DialogueAct`. All other modules should consume that result.

Remove the rule that a normal participant must reject every candidate outside their initial acceptable set. Replace it with a graded willingness model. Initial acceptable options are starting beliefs, not immutable hard constraints.

Remove deterministic preference rewrites that do not update reasons. `_enforce_divergence()` should not silently change `preferred` after the belief state was generated. Divergence must be created during belief generation or repaired by regenerating a full consistent belief object.

Remove LLM-facing prose from `prompt_context.py`. It may still assemble sections, but the actual wording of headings, instructions, and output contracts should live in `prompts.py`.

Remove hardcoded option letters from business logic where possible. Four options are currently required, but the schema should still expose valid labels from config/schema rather than scattered `A-D` checks.

Remove hardcoded natural-language moderator lines from `orchestrator.py`. Moderator templates should either live in `prompts.py` or in a dedicated non-LLM `moderator_templates.py`. Since the project rule says prompts sent to the LLM must be in `prompts.py`, deterministic transcript templates may be outside `prompts.py`, but for consistency I recommend moving all dialogue wording templates to `prompts.py` as well.

Remove the idea that repetition pressure alone is a readiness signal. Repetition is a stall signal, not proof that the discussion is sufficient.

Remove LLM repair as the default response to every verifier failure. For phase-critical decisions, use deterministic fallbacks. For normal discussion, allow one repair attempt, then either emit a safe fallback or skip the turn. Multiple LLM repairs create unpredictable behavior.

## 3. Parts that need refactoring

### 3.1 `orchestrator.py`

Current role: too broad.

Target role: coordinate the run only.

The new orchestrator should call these services:

1. `ScenarioBuilder` for options and topic profile.
2. `PersonaBuilder` for participants and initial belief states.
3. `DialogueController` for phase/readiness decisions.
4. `TurnRouter` for speaker and move intent.
5. `UtteranceGenerator` for one participant message.
6. `StateTracker` for state updates.
7. `Validator` for generated-message checks.
8. `ConsensusManager` for final decision logic.
9. `DialogueLogger` for logging and metrics.

After refactoring, `orchestrator.py` should mostly read like this:

```python
class Orchestrator:
    def run(self) -> DialogueRunResult:
        scenario = scenario_builder.build(topic)
        group = persona_builder.build(topic, scenario)
        state = dialogue_state.initialize(topic, scenario, group)
        logger.start(state)

        while not controller.should_stop(state):
            intent = router.next_move(state)
            message = utterance_generator.generate(intent, state)
            result = validator.validate(message, intent, state)
            message = repair_or_fallback(result, intent, state)
            state = state_tracker.apply(message, intent, state)
            logger.record(state, message, result)

        outcome = consensus_manager.finalize(state)
        logger.finish(outcome)
        return outcome
```

`orchestrator.py` should not contain regexes, candidate ranking, option parsing, or prompt strings.

### 3.2 `persona.py`

Current role: mostly useful, but too much trait interpretation is hardcoded.

Keep:

- Big Five as a hidden control vector.
- `AgentBeliefs` / argument kit idea.
- cooperative defaults.
- one grouped LLM call for belief generation.

Refactor:

- Move trait-control weights from hardcoded formulas into config.
- Replace low-agreeableness forcing in `_enforce_diversity()` with softer diversity controls.
- Separate “disagreement style” from “willingness to compromise”. A blunt participant can still compromise. A quiet participant can still be firm. Do not equate low agreeableness with obstruction.
- Add `compromise_willingness`, `patience`, `initiative`, `detail_level`, and `conflict_directness` as derived controls.
- Add `hard_blocker` only through the explicit rare sampler, not through normal trait combinations.

Important change: belief generation must output a complete and internally consistent object. If the system modifies preferred/acceptable/rejected options, it must also repair reasons and reconsideration conditions.

### 3.3 `policy.py`

Current role: speaker selection.

Keep:

- one speaker at a time.
- priority for directed questions.
- underrepresented speakers get a boost.
- no simple round-robin.

Refactor:

- Return a `MoveIntent`, not just a speaker.
- Include why the speaker was selected.
- Include what the speaker should do locally: answer, ask, clarify, compare, push_back, concede, propose_compromise, vote, confirm, close.
- Move hard-blocker sampling out of speaker policy into persona/group setup.

The turn router should produce:

```python
@dataclass
class MoveIntent:
    speaker_id: str
    addressee_id: str | None
    act: Literal[
        "answer", "react", "compare", "push_back", "clarify",
        "concede", "propose_compromise", "vote", "confirm", "close"
    ]
    option_focus: list[str]
    reason: str
    length_hint: Literal["short", "medium", "long"]
```

This prevents the LLM from deciding too much implicitly.

### 3.4 `state.py`

Current role: compact memory only.

Refactor into a real state model. Keep compact memory, but add explicit state tables.

Needed state fields:

- `turns`: full structured turn records.
- `open_questions`: who asked, who should answer, whether answered.
- `option_coverage`: per-option mentions, reasons, objections, acceptances.
- `stance_by_participant`: preferred, acceptable, rejected, undecided, can_live_with.
- `pending_intent`: current local move goal.
- `compromise_proposals`: proposals made by sims or moderator.
- `moderator_interventions`: count and reason.
- `last_decision_move`: last vote/proposal/confirmation.
- `topic_profile`: simple/medium/complex decision estimate.

Do not let `orchestrator.py` own these as loose fields.

### 3.5 `simulator.py`

Current role: participant wrapper and utterance generator.

Refactor into `utterance_generator.py` and a lightweight `SimAgent` object.

The sim object should contain private state and identity. The generator should receive a `MoveIntent` and produce one message. This is cleaner than letting `Simulator.generate_turn()` inspect phase, state, memory, candidate, verification state, and repair logic.

Target structure:

```python
class SimAgent:
    id: str
    persona: Persona
    private_state: SimPrivateState

class UtteranceGenerator:
    def generate(self, intent: MoveIntent, state: DialogueState) -> RawUtterance:
        ...
```

`simulator.py` currently contains decision-phase structured-control logic. That should move to `decision_utterances.py` or remain in `utterance_generator.py` but be separated from normal chat generation.

### 3.6 `verifier.py`

Current role: deterministic verification.

Keep:

- invalid option reference check.
- option fact mutation check.
- repetition check.
- missing vote check.
- unclear confirmation check.

Refactor:

- Group checks by category.
- Return `ValidationResult` with severity and recommended action.
- Stop spreading semantic interpretation across verifier, simulator, and orchestrator.
- Move all thresholds into config.
- Replace giant regex growth with clearer checks based on structured state where possible.

Suggested categories:

```python
class ValidationCategory(Enum):
    STRUCTURE = "structure"
    GROUNDING = "grounding"
    TURN_TAKING = "turn_taking"
    PERSONA = "persona"
    REPETITION = "repetition"
    DECISION = "decision"
    STYLE = "style"
```

### 3.7 `prompts.py` and `prompt_context.py`

Current role: mostly correct, but prompt prose leaks into `prompt_context.py`.

Target:

- `prompts.py`: all LLM-facing wording.
- `prompt_context.py`: pure rendering of state data into neutral blocks, or remove it and put renderers under `prompts.py`.

Prompt design should be shorter and stricter. The LLM should receive:

1. Role/persona card.
2. Current private stance.
3. Option cards.
4. Recent turns.
5. Current move intent.
6. Output contract.

The prompt should not describe the entire architecture. It should constrain one local move.

### 3.8 `config.yaml`

Current config is useful but incomplete.

Add sections for:

- topic complexity.
- adaptive readiness.
- move intent probabilities.
- cooperation/stubbornness controls.
- moderator intervention policy.
- validation thresholds.
- evaluation thresholds.
- prompt length budgets.
- hardcoded trait-control weights.

Also rename some sections for clarity:

- `turns` -> `dialogue_control`.
- `divergence` -> `belief_distribution`.
- `structured_control` -> `decision_control`.

## 4. Missing components

### 4.1 `schemas.py`

Add a single schema module. This is the foundation of the refactor.

Required classes:

```python
@dataclass
class OptionCard:
    id: str
    name: str
    attrs: dict[str, str | int | float | bool]
    upside: str
    tradeoff: str
    concern: str
    fit: str
    risk: str
    best_for: str

@dataclass
class PersonaProfile:
    id: str
    name: str
    role: str
    is_primary: bool
    traits: dict[str, int]
    controls: dict[str, float]
    speech_style: str

@dataclass
class BeliefState:
    preferred: str
    acceptable: dict[str, float]       # option -> willingness score
    rejected: dict[str, str]           # true blockers only
    reasons: dict[str, list[str]]      # option -> reasons
    reservations: dict[str, str]
    would_reconsider_if: dict[str, str]

@dataclass
class DialogueAct:
    speaker_id: str
    addressee_id: str | None
    act_type: str
    option_refs: list[str]
    vote: str | None
    accepts: list[str]
    rejects: list[str]
    asks_question: bool
    answered_question_id: str | None
    proposes_compromise: bool
    text: str

@dataclass
class DialogueState:
    phase: str
    topic_profile: TopicProfile
    options: list[OptionCard]
    agents: dict[str, SimAgentState]
    turns: list[DialogueAct]
    open_questions: list[QuestionState]
    option_coverage: dict[str, OptionCoverage]
    proposals: list[CompromiseProposal]
    decision: DecisionState
```

Use dataclasses initially. Pydantic is useful if you want strict runtime validation, but dataclasses are enough for a university project if validation helpers are clear.

### 4.2 `scenario_builder.py`

This module should generate and validate the four options.

Current option generation returns strings. Replace it with JSON objects:

```json
{
  "decision_kind": "restaurant_choice",
  "complexity": "medium",
  "options": [
    {
      "id": "A",
      "name": "...",
      "attrs": {"price_per_person_eur": 18, "noise_level_1_5": 2},
      "upside": "...",
      "tradeoff": "...",
      "concern": "...",
      "fit": "...",
      "risk": "...",
      "best_for": "..."
    }
  ],
  "opening_question": "..."
}
```

Rendering to string should happen only for prompts/transcripts.

### 4.3 `topic_profiler.py`

Add a deterministic or LLM-assisted topic profile.

The profile should estimate:

- decision kind.
- complexity: simple, medium, complex.
- expected discussion intensity.
- minimum option coverage needed.
- whether numeric trade-offs matter.
- whether participants should vote explicitly.

This solves the varying-length problem better than fixed phase lengths.

Example:

```python
@dataclass
class TopicProfile:
    decision_kind: str
    complexity: Literal["simple", "medium", "complex"]
    intensity: Literal["low", "normal", "high"]
    required_options_discussed: int
    required_reasoned_participants_ratio: float
    allow_early_consensus: bool
    require_explicit_vote: bool
```

A simple game-night choice may need only 2–4 real exchanges after opening. A group trip or hotel choice may need all options touched and several trade-offs.

### 4.4 `state_tracker.py`

This must be the only module that updates public dialogue state from messages.

It should:

- parse each message into a `DialogueAct`.
- update open questions.
- update option coverage.
- update public stance.
- update compromise proposals.
- update decision state.

Use deterministic parsing first. Only use an LLM extractor if deterministic parsing is insufficient, and only with a strict JSON schema.

### 4.5 `dialogue_controller.py`

This replaces fixed phase logic.

It should decide whether the dialogue should continue exploring, narrow, test a compromise, ask a moderator question, or close.

Core method:

```python
class DialogueController:
    def next_stage(self, state: DialogueState) -> StageDecision:
        ...
```

Readiness should be score-based, not only threshold-based.

Suggested readiness factors:

- enough participants have spoken.
- no open direct question is unresolved.
- enough options have at least one grounded reason or objection.
- at least one feasible shared fallback exists.
- repetition/stall is rising.
- a sim has proposed a compromise.
- topic complexity requirements are satisfied.

Example formula:

```text
readiness =
  0.25 * participant_reason_coverage +
  0.20 * option_tradeoff_coverage +
  0.20 * shared_fallback_evidence +
  0.15 * no_open_question +
  0.10 * diminishing_novelty +
  0.10 * compromise_signal
```

All weights must be in config.

### 4.6 `turn_router.py`

This should replace `policy.select_next_speakers()`.

It should choose both speaker and move intent.

Priority cascade:

1. Answer direct addressed question.
2. Let challenged participant respond.
3. Let participant with unresolved objection explain or soften it.
4. Let underrepresented participant contribute.
5. Let a participant propose compromise if enough common ground exists.
6. Let moderator intervene only if stuck.
7. Otherwise sample a relevant participant.

The router should output explicit `MoveIntent`, not only a speaker.

### 4.7 `consensus_manager.py`

This module should own final agreement logic.

A consensus is valid only if:

- every participant has explicitly voted for or accepted the final option/proposal, or
- all unresolved holdouts are marked as rare true hard blockers and the configured fallback policy permits no-consensus closure.

It should distinguish:

- `consensus_success`: everyone prefers or accepts final option.
- `compromise_success`: not everyone prefers it, but everyone can live with it.
- `majority_fallback`: allowed only if configured.
- `no_consensus`: honest failure.
- `invalid_forced_close`: internal failure; should count against the simulator.

Do not mark `force_close` as a valid outcome simply because it names an option.

### 4.8 `evaluation.py`

Create a real evaluation module rather than embedding all metrics in the logger.

Minimum metrics:

- outcome distribution.
- valid consensus rate.
- force-close rate.
- no-consensus rate.
- true-hard-blocker rate.
- false-stubbornness rate.
- average turns by topic complexity and participant count.
- option coverage before narrowing.
- addressed-question correctness.
- speaker balance / Gini.
- moderator intervention ratio.
- sim-driven compromise count.
- repeated phrase rate.
- repair rate.
- hallucinated fact count.
- vote/acceptance consistency.

Run at least 50–100 generated dialogues for tuning.

### 4.9 `tests/`

Add tests before further tuning. Without tests, every fix can reintroduce old failures.

Required tests:

- option parser accepts valid option cards and rejects malformed ones.
- vote parser handles “I’d go with B”, “B works for me”, and rejects “not B”.
- direct question routes to addressed participant.
- a participant cannot vote for an explicitly rejected option unless changed-mind marker exists.
- normal non-hard-blocker can accept a non-preferred but feasible compromise.
- true hard-blocker is rare and logged.
- readiness does not allow vote before minimum coverage.
- readiness can close early for simple low-intensity topics.
- consensus manager does not call force-close a valid consensus.

## 5. Target file structure

Recommended structure:

```text
src/
  main.py
  config_loader.py
  llm_client.py

  schemas.py
  scenario_builder.py
  persona_builder.py
  state_tracker.py
  dialogue_controller.py
  turn_router.py
  utterance_generator.py
  validator.py
  repair.py
  consensus_manager.py
  evaluator.py
  logger.py
  prompts.py
  renderers.py
  utils.py

tests/
  test_option_cards.py
  test_act_parser.py
  test_turn_router.py
  test_readiness.py
  test_consensus.py
  test_validator.py

config.yaml
PLAN.md
README.md
```

`renderers.py` may contain pure formatting logic, but it must not contain LLM instructions. If a string is instructional and goes into an LLM prompt, it belongs in `prompts.py`.

## 6. New end-to-end flow

The new flow should be:

```text
1. Build TopicProfile.
2. Generate four structured OptionCards.
3. Generate participants.
4. Generate private belief states with cooperative defaults.
5. Sample at most one rare hard blocker.
6. Initialize DialogueState.
7. Moderator introduces topic and options.
8. Loop:
   a. DialogueController decides current stage.
   b. TurnRouter selects speaker + MoveIntent.
   c. UtteranceGenerator builds prompt from prompts.py and generates one message.
   d. Validator checks structure, grounding, turn-taking, persona consistency, repetition, and decision validity.
   e. Repair/fallback emits final message.
   f. StateTracker parses and applies the message.
   g. Evaluator records metrics incrementally.
9. ConsensusManager finalizes outcome.
10. Logger writes transcript and JSON evaluation.
```

The LLM should never decide the full discussion, the next phase, the final consensus, or the next speaker. It should only generate local language under a move intent.

## 7. Adaptive discussion length

Fixed phase lengths should be replaced by an adaptive budget.

Use three levels:

```yaml
topic_complexity:
  simple:
    min_reasoned_participant_ratio: 0.50
    min_options_discussed: 2
    target_turns_per_participant: [1, 2]
    readiness_threshold: 0.62
  medium:
    min_reasoned_participant_ratio: 0.70
    min_options_discussed: 3
    target_turns_per_participant: [2, 3]
    readiness_threshold: 0.70
  complex:
    min_reasoned_participant_ratio: 0.85
    min_options_discussed: 4
    target_turns_per_participant: [2, 5]
    readiness_threshold: 0.78
```

The upper bound should still exist to avoid endless loops, but it should scale by participant count and complexity.

Use this formula:

```text
max_dialogue_turns = base_by_complexity + participants * turns_per_participant_by_complexity
```

All values belong in config.

## 8. Stubbornness and compromise model

The simulator needs disagreement, not disruption.

Use three different concepts:

1. `preference_strength`: how much the sim likes their top option.
2. `compromise_willingness`: how easily they accept a non-top option.
3. `hard_blocker`: rare flag that creates genuine refusal.

Normal participants should have:

- one preferred option.
- two or more acceptable options.
- no true rejected option unless the scenario gives a serious blocker.
- willingness to accept a shared fallback after enough discussion.

Hard blockers should be sampled rarely, for example 3–7% of dialogues, not 60–70% of sims. If sampled, choose at most one hard blocker. This flag must be logged.

Do not derive hard blocking directly from low agreeableness. Low agreeableness should change expression style: more direct, less validating, more likely to push back. It should not automatically mean “will not compromise”.

Suggested belief model:

```python
acceptable: dict[str, float]
# A: 0.95 preferred
# B: 0.75 acceptable fallback
# C: 0.55 weak but possible
# D: 0.20 real blocker only if hard reason exists
```

Consensus testing then uses a threshold:

```text
can_accept(option) = willingness(option) >= acceptance_threshold
```

The threshold can decrease slightly after a participant’s concerns are addressed, but not below a config floor.

## 9. Moderator behavior

The moderator should be a facilitator, not a script engine.

Allowed moderator actions:

- introduce topic/options.
- invite a quiet participant.
- summarize when the thread stalls.
- ask for a pick only when readiness is high.
- clarify a candidate compromise.
- honestly close success or failure.

Avoid:

- repeated “could you live with Option X?” loops.
- forcing every participant through the same confirmation template.
- announcing a candidate after every small signal.
- closing when a direct question is unresolved.

Add moderator policy config:

```yaml
moderator_policy:
  max_moderator_ratio: 0.25
  min_participant_turns_between_moderator: 2
  allow_sim_compromise_first: true
  ask_vote_only_if_readiness_reached: true
  max_candidate_checks_per_dialogue: 2
  avoid_repeating_candidate_prompt: true
```

## 10. Prompt design

The current prompts are directionally good but still too general. The new prompt should be based on `MoveIntent`.

Example participant prompt shape:

```text
You are writing only the next message for {speaker_name}.

Speaker card:
{speaker_card}

Private stance:
{private_stance}

Options:
{option_cards}

Recent turns:
{recent_turns}

Your local move:
- act: answer
- addressee: Liam
- focus: Option B and Option C
- goal: answer Liam's question about whether Option C is acceptable as fallback
- length: short

Rules:
- casual adult group chat, not formal, not Gen-Z-heavy.
- use only listed option facts.
- do not repeat your previous point.
- output one raw message, no name prefix.
```

The model should never be asked to “continue the discussion naturally” without a concrete local move. That gives it too much freedom and causes loops.

## 11. Validation and repair policy

Validation should happen after every generated message.

Severity levels:

- `fatal`: cannot emit; fallback or skip.
- `repair`: one repair allowed.
- `warn`: emit but log.

Recommended policy:

```text
normal discussion:
  one LLM repair attempt; if still invalid, deterministic short fallback
vote/confirmation:
  no free-form repair by default; deterministic structured fallback
closure:
  deterministic template unless a natural sign-off is needed
```

Keep deterministic validators fast. Add an optional LLM judge only for offline evaluation, not runtime control.

## 12. Config requirements

All tunable numeric values must live in `config.yaml`.

Move these current hardcoded values into config:

- derived-control weights in `Persona.derived_controls()`.
- descriptor thresholds such as `0.65` and `0.30`.
- diversity fallback values such as random agreeableness `1–2`.
- option-letter assumptions where feasible.
- verifier proximity windows not already in config.
- question-chain windows and counts.
- word-count minimums.
- all readiness-score weights.
- all topic-complexity budgets.
- max candidate checks.
- hard-blocker probability and max blockers.

Non-tunable constants may remain in code only if they are true structural constants, e.g. default phase names or dataclass field names.

## 13. Concrete migration order

### Step 1 — Freeze current behavior

Do not start by changing prompts. First add a small test harness and save example outputs from the current version. Keep 10–20 topics covering restaurants, hotels, flights, study plans, activities, hiking, and abstract decisions.

### Step 2 — Add schemas

Create `schemas.py`. Add structured dataclasses. Do not refactor logic yet. Make current modules import the schemas gradually.

### Step 3 — Replace string option cards

Refactor option generation to return `OptionCard` objects. Add render functions for transcript and prompt. Update `OptionResolver` to consume structured options.

### Step 4 — Centralize act parsing

Create `state_tracker.py` or `act_parser.py`. Move vote/accept/reject/option-reference parsing there. Make orchestrator, simulator, verifier, and logger consume the same parsed `DialogueAct`.

### Step 5 — Fix belief/stubbornness model

Change belief state from accepted/rejected lists to willingness scores. Only true hard blockers get hard rejections. Remove silent preference rewrites. Add consistency checks after belief generation.

### Step 6 — Introduce MoveIntent

Refactor `policy.py` into `turn_router.py`. Return speaker plus act, addressee, option focus, and length hint.

### Step 7 — Refactor utterance generation

Make `UtteranceGenerator.generate(intent, state)` the only normal chat generator. Keep decision generation separate but use the same prompt registry and validation result model.

### Step 8 — Replace phase logic with controller

Create `dialogue_controller.py`. Use topic profile and readiness score. Keep upper bounds as safety stops, but do not use fixed phase lengths as the primary behavior.

### Step 9 — Move consensus logic

Create `consensus_manager.py`. It should own candidate ranking, acceptance checks, outcome labels, and final validation.

### Step 10 — Move evaluation out of logger

Create `evaluator.py`. Logger writes files. Evaluator computes metrics.

### Step 11 — Run batch evaluation

Generate 50–100 dialogues. Tune config only after looking at metrics and transcripts.

### Step 12 — Clean dead code

Delete old parsing helpers, old candidate logic, old forced divergence logic, unused persona summaries, and duplicated repair paths.

## 14. Acceptance criteria

The project is “good enough” when these are true across a batch of test topics:

- 80–90% of normal dialogues reach valid consensus or valid compromise.
- No-consensus outcomes happen mostly when a true hard-blocker was sampled.
- Force-close is rare and treated as a failure signal, not success.
- Direct questions are answered by the addressed participant in most cases.
- Every final option was discussed or explicitly accepted.
- At least two options are meaningfully discussed for simple topics; three or four for complex topics.
- Moderator turns stay below the configured ratio.
- Sim-driven compromise proposals occur in a meaningful fraction of dialogues.
- Repeated acknowledgement loops are rare.
- Generated messages vary in length.
- Chats sound casual but not slang-heavy.
- No generated hard facts contradict the option cards.

## 15. Final target claim for the project

The final system should be described as:

> A controlled LLM-based simulator for casual multi-party decision discussions. It uses structured option generation, cooperative persona-conditioned private states, explicit turn-routing, adaptive readiness control, deterministic validation, and consensus tracking to generate grounded synthetic group chats that usually reach a plausible compromise.

Do not claim that the chats are fully human-realistic. Claim that they are grounded, coherent, diverse, natural-sounding, and evaluable.

## 16. Main implementation warning

Do not keep adding special cases. The project already shows signs of patch fatigue: many regexes, many repair prompts, several separate parsers, and a monolithic orchestrator. The next improvement should be structural consolidation. Once state, acts, routing, and consensus are clean, prompts become much easier to tune.
