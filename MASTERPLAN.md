# MASTERPLAN_REVISED.md — Focused Refactor Plan for the Group-Discussion User Simulator

## 0. Purpose

This document is the revised implementation plan for the dialogue-simulation project.

The earlier system already contains many of the right building blocks:

- compact hidden user states;
- diverse personas;
- preferences and acceptable options;
- moderator-guided phase progression;
- explicit options;
- LLM-generated utterances;
- logging and evaluation data.

The project does **not** need another large theoretical layer. It needs a narrower control structure and a strong verification/repair layer around each generated utterance.

The target system is:

> A scalable simulator that generates casual, human-like group chats between distinct simulated users who discuss options, avoid repeating themselves, try to compromise, and either reach a valid decision or honestly fail when compromise is not plausible.

The main correction is:

> Keep the useful architecture. Remove overlapping control logic. Add deterministic verification after generation.

---

## 1. Core conclusion

Your code already roughly follows the research pattern:

1. Define hidden user state.
2. Generate bounded personas.
3. Keep task options.
4. Route turns.
5. Let the LLM generate utterances.
6. Track votes and acceptability.
7. Evaluate output.

The missing or weak part is:

```text
verify the utterance before accepting it
```

This should become the central improvement.

The simulator should not rely only on better prompts. Prompts help, but they will not reliably prevent:

- repeated statements;
- invalid option claims;
- invented option facts;
- unclear votes;
- moderator-created extra options;
- fake consensus;
- conditionals being misread as support.

A lightweight verifier should catch these problems and trigger one repair attempt.

---

## 2. Revised design principle

Use this strict separation:

| Component | Responsibility |
|---|---|
| Persona | defines stable preferences, style, response length, flexibility, and reasons |
| Prompt | asks for one natural next message |
| Turn policy | chooses who speaks next |
| Verifier | checks whether the generated message is valid |
| Repair | regenerates once if invalid/repetitive |
| Moderator | moves phases and asks focused questions |
| Consensus | uses explicit votes + private acceptability |
| Evaluation | measures quality after the dialogue |

Do not let one component do the job of another.

Especially:

- Do not let evaluation metrics control phase transitions.
- Do not let the moderator invent options.
- Do not let traits become a large dialogue-act planner.
- Do not infer full stance from every casual sentence.
- Do not use challenge graphs as a requirement for progress.

---

## 3. What to keep

### 3.1 Keep persona traits

Keep:

```text
openness
conscientiousness
extraversion
agreeableness
neuroticism
response_length
```

These are useful for distinct simulated users and later evaluation.

But use them mainly to derive simpler behavioural controls:

```text
initiative
flexibility
directness
detail_level
warmth
target_response_length
```

The raw traits should remain logged. Runtime code should mostly use the derived controls.

### 3.2 Keep private belief states

Keep the current `AgentBeliefs` idea.

Each sim should have:

```text
preferred
acceptable
rejected
key_concern
reasons
reservation
would_reconsider_if
```

This is one of the strongest parts of the project. It gives sims something to discuss beyond repeating "I like A".

### 3.3 Keep moderation

The moderator is useful for:

- option setup;
- opening prompt;
- nudging stuck discussions;
- asking for votes;
- asking holdouts;
- confirming final candidate;
- honest force-close/failure.

But moderator output must be verified too.

### 3.4 Keep memory, but make it operational

Memory should exist to prevent repetition, not to simulate a full mind.

Keep:

```text
last own turn
recent own point signatures
recent dialogue turns
current phase
current candidate/votes if relevant
```

Avoid large summaries of other people's arguments unless needed.

### 3.5 Keep evaluation logging

Keep `_eval.json`.

The system should support later LLM evaluation of:

- persona consistency;
- response-length consistency;
- compromise plausibility;
- distinctness between speakers;
- repetition;
- option validity;
- final outcome validity.

---

## 4. What to remove or demote

### 4.1 Remove challenge-gated progress

Challenge-response tracking can exist as an optional metric, but it must not gate phase transitions.

Delete or disable runtime logic that says the group cannot narrow/vote until a challenge has been answered.

Reason:

- It makes agents perform disagreement.
- It creates artificial "but doesn't X outweigh Y?" loops.
- It makes casual chats feel like debate exercises.

### 4.2 Remove heavy dialogue-act planning

The act planner is too much control.

Delete or disable:

```text
TurnPlan
plan_turn()
large act-weight sampling
strategy cooldowns over abstract acts
```

Replace with simple phase obligations:

```text
opening -> state priority
discussion -> respond naturally
vote -> explicit vote
confirmation -> yes/no
closure -> sign off
```

### 4.3 Demote full stance inference

Do not build consensus from every option mention.

Use explicit votes and explicit confirmations.

Track casual option mentions for evaluation and context, but do not treat them as reliable support.

Important rule:

```text
"If C had X, I could accept it" is not support for C unless X is true or feasible.
```

### 4.4 Remove option filtering from participant prompts

Every participant should always see all options.

Delete or replace any prompt function that only shows "relevant" options.

This fixes invalid statements such as a participant claiming that a listed option is not available.

### 4.5 Reduce long backstories

Backstories should be short.

Keep them as one light sentence, not a long explanation.

Reason:

- Long backstories dominate the dialogue.
- Sims start reciting biography instead of discussing the current decision.

---

## 5. Add the missing piece: verifier + repair layer

This is the most important new implementation.

### 5.1 Add a new module: `verifier.py`

Create:

```text
src/verifier.py
```

Purpose:

- deterministic checks after each generated message;
- return validation errors;
- decide whether one repair attempt is needed.

This should not call an LLM. It should be fast and deterministic.

### 5.2 Main data structures

Add:

```python
@dataclass
class VerificationIssue:
    code: str
    severity: str  # "repair" | "warn" | "fatal"
    message: str
```

Add:

```python
@dataclass
class VerificationResult:
    ok: bool
    issues: list[VerificationIssue]
    needs_repair: bool
```

### 5.3 Verifier functions

Implement these functions:

```python
verify_participant_turn(
    text: str,
    speaker_name: str,
    phase: str,
    options: list[str],
    history: list[str],
    persona_state: ParticipantState | None,
    resolver: OptionResolver,
    candidate: str | None = None,
) -> VerificationResult
```

```python
verify_moderator_turn(
    text: str,
    options: list[str],
    resolver: OptionResolver,
    allow_multi_option_solution: bool = False,
) -> VerificationResult
```

Helper checks:

```python
detect_invalid_option_reference(text, options, resolver)
detect_option_denial(text, options, resolver)
detect_invented_option_attribute(text, options, resolver)
detect_self_repetition(text, speaker_name, history, persona_state)
detect_missing_vote(text, phase, resolver)
detect_unclear_confirmation(text, phase, candidate)
detect_moderator_new_option(text, options, resolver)
detect_moderator_mixed_solution(text, allow_multi_option_solution)
```

### 5.4 Verification issue codes

Use explicit issue codes:

```text
INVALID_OPTION_REFERENCE
VALID_OPTION_DENIED
INVENTED_OPTION_FACT
SELF_REPETITION
MISSING_EXPLICIT_VOTE
UNCLEAR_CONFIRMATION
MODERATOR_NEW_OPTION
MODERATOR_MIXED_SOLUTION
NAME_PREFIX
TOO_LONG
EMPTY_OR_SILENCE
```

### 5.5 Repair rules

Repair once.

If the generated participant message fails verification:

1. Build a repair prompt.
2. Include the original bad message.
3. Include the specific issue.
4. Ask the LLM to rewrite the same turn, not continue the conversation.
5. Verify again.
6. If it still fails, fall back to a deterministic safe line for phase-required cases.

Example repair prompt:

```text
Your previous message was rejected because it repeated your earlier point.
Rewrite the same turn as one natural chat message.
Do not repeat the same reason.
React to the recent chat or move toward a decision.
No name prefix.
```

For vote phase:

```text
Your previous message did not clearly vote for one option.
Write one natural message that explicitly chooses exactly one of Option A-D.
No name prefix.
```

For invalid option denial:

```text
Your previous message incorrectly claimed that a listed option is unavailable.
Rewrite it while respecting that all listed options are available.
No name prefix.
```

### 5.6 Where verifier is called

In `simulator.py`:

```text
generate raw turn
clean raw turn
verify participant turn
if repair needed:
    repair once
verify again
return final turn
```

In `moderation.py` or `orchestrator.py`:

```text
generate or template moderator line
verify moderator line
if invalid:
    use deterministic fallback template
```

---

## 6. Revised file-level plan

### 6.1 `main.py`

Keep.

Minor changes only:

- keep entry point;
- keep batch mode;
- optionally print shorter persona summaries;
- make debug output configurable.

No large refactor here.

### 6.2 `config_loader.py`

Keep.

Update declared config sections after config simplification.

Do not overwork this file.

### 6.3 `llm_client.py`

Keep.

No major change.

Config should reduce temperature if currently high.

Recommended:

```yaml
temperature: 0.55-0.70
top_p: 0.90
```

### 6.4 `persona.py`

Keep and simplify.

Keep:

- `Persona`
- `AgentBeliefs`
- `SpeechSignature`
- `PersonaBuilder`
- trait generation
- belief generation
- diversity enforcement

Modify:

- add derived conversational traits;
- shorten backstories;
- ensure `acceptable` sets overlap in normal cases;
- ensure rare hard-blocker cases are truly rare;
- ensure `response_length` diversity;
- make `would_reconsider_if` feasible.

Do not remove traits. They are useful.

### 6.5 `policy.py`

Simplify.

Keep:

- speaker selection;
- direct-address priority;
- participation balancing;
- repetition pressure helpers if still useful.

Remove or disable:

- heavy act planning;
- `TurnPlan`;
- strategy cooldowns;
- large personality-bias act weights.

New core function:

```python
select_next_speaker(sims, state, history, structured_state) -> Simulator
```

Selection priority:

1. unanswered direct question target;
2. phase obligation target;
3. participant with low participation;
4. initiative bias;
5. avoid same speaker twice unless required.

### 6.6 `state.py`

Simplify, but do not destroy useful logging.

Keep:

- turn records;
- participant state;
- structured state;
- state tracker.

Remove or demote:

- full stance table as consensus source;
- challenge graph as runtime control;
- complex dialogue acts.

Track:

```text
turn_id
speaker
text
phase
mentioned_options
explicit_vote
direct_question_to
answered_question
word_count
verification_issues
repair_attempted
```

### 6.7 `prompt_context.py`

Do not delete immediately.

Short-term:

- keep if removing it would cause too much churn;
- change `build_relevant_options()` so it always returns all options;
- simplify memory block;
- remove challenge-specific prompt text.

Long-term:

- merge into `prompts.py` only if the code becomes easier, not as a symbolic cleanup.

The old plan said to delete it. This revised plan demotes that requirement. Stability matters more.

### 6.8 `prompts.py`

Keep as the central prompt registry.

Simplify participant prompts.

Participant turn prompt should contain:

```text
topic
all options
speaker card
current phase
recent chat
short memory
specific phase instruction
output rules
```

Avoid:

- huge theoretical instructions;
- too many possible acts;
- forcing disagreement;
- forcing every turn to include a reason.

Add repair prompts:

```python
repair_repetition(...)
repair_invalid_option(...)
repair_vote(...)
repair_confirmation(...)
repair_grounding(...)
```

### 6.9 `simulator.py`

Keep.

Add verifier integration.

New flow:

```text
_generate()
clean
verify
repair once if needed
verify again
enforce word budget
return
```

Important:

- do not repair endlessly;
- log repair attempts;
- deterministic fallback only for phase-required messages.

### 6.10 `moderation.py`

Keep, but constrain.

Make most moderator messages deterministic templates.

Allow LLM moderator lines only when:

- the output is verified;
- no new options are introduced;
- no mixed solution is introduced unless config allows it.

Add deterministic fallbacks for:

```text
ask_vote
ask_holdout
ask_confirmation
announce_success
announce_force_close
announce_failure
```

### 6.11 `orchestrator.py`

Simplify carefully.

Do not rewrite all at once.

Primary changes:

- make phase flow explicit;
- remove challenge-gated progress;
- use explicit votes for vote phase;
- use private acceptability for compromise;
- call verifier path through simulator/moderator;
- log verification issues.

Preferred phase flow:

```text
opening
discussion
vote
compromise
confirmation
closure
```

If keeping old phase names temporarily is easier, that is acceptable, but behaviour should match this flow.

### 6.12 `reasoning.py`

Keep but reduce runtime power.

Keep:

- scoped fact checking;
- consensus helper;
- evaluation metrics if useful.

Modify consensus:

- explicit vote first;
- private acceptability second;
- no vague stance inference as final authority.

Move Fisher/deliberation metrics to evaluation-only.

### 6.13 `logger.py`

Keep.

Add verifier metadata to `_eval.json`:

```json
"verification": {
  "issues": [],
  "repair_attempts": 0,
  "failed_repairs": 0
}
```

For each turn, log:

```json
{
  "speaker": "...",
  "text": "...",
  "phase": "...",
  "verification_issues": [],
  "repair_attempted": false
}
```

### 6.14 `utils.py`

Keep.

Add helpers only if needed by `verifier.py`.

Useful helpers:

```python
normalize_option_label(...)
option_denial_patterns(...)
extract_numbers_near_option(...)
```

Avoid turning `utils.py` into a second verifier.

### 6.15 `config.yaml`

Simplify gradually.

Add verifier config:

```yaml
verification:
  enabled: true
  repair_attempts: 1
  check_repetition: true
  check_option_validity: true
  check_moderator_options: true
  check_votes: true
  check_confirmation: true
```

Keep trait and persona sections.

Disable or remove later:

```yaml
act_planner
personality_bias
challenge_gating
```

If deleting sections would break too much code, first set them inactive.

---

## 7. Revised implementation order

### Phase 1 — Verification-first patch

Do this before major rewrites.

1. Create `src/verifier.py`.
2. Add participant verification in `simulator.py`.
3. Add moderator verification in `moderation.py` or `orchestrator.py`.
4. Make all options visible in every sim prompt.
5. Add repair-on-repetition.
6. Add repair-on-invalid-option/fact.
7. Add repair-on-missing-vote.
8. Log verifier issues.

This phase gives immediate quality improvement without huge architectural risk.

### Phase 2 — Reduce forced debate

1. Disable challenge-gated narrowing/progress.
2. Stop requiring answered challenges before moving forward.
3. Remove prompt wording that forces pushback.
4. Keep direct questions and answers.

Goal:

- agents should respond to each other naturally;
- disagreement should happen when useful, not because the system needs a challenge edge.

### Phase 3 — Simplify consensus

1. Track explicit votes.
2. Track confirmations.
3. Use private acceptability for compromise.
4. Stop using vague stance inference as the main decision source.
5. Treat impossible conditionals as non-support.

Goal:

- final decision must be valid and explainable.

### Phase 4 — Persona cleanup

1. Keep traits.
2. Add derived conversational controls.
3. Shorten backstories.
4. Improve response-length diversity.
5. Make acceptable sets overlap unless hard-blocker case.

Goal:

- preserve variety without unstable control flow.

### Phase 5 — Runtime simplification

1. Simplify `policy.py`.
2. Simplify `state.py`.
3. Simplify `orchestrator.py`.
4. Remove dead code only after tests pass.

Goal:

- easier debugging;
- fewer overlapping systems.

### Phase 6 — Evaluation polish

1. Add automatic metrics:
   - validity;
   - repetition;
   - response-length adherence;
   - participation balance;
   - outcome type;
   - repair frequency.
2. Keep full transcript and persona metadata.
3. Prepare for later LLM-based evaluation, but do not add it to runtime yet.

---

## 8. Concrete verifier behaviour

### 8.1 Repetition check

Compare generated message against:

- speaker's previous message;
- recent own point signatures;
- optionally last few group turns.

Repair threshold examples:

```yaml
repetition:
  own_last_turn_jaccard: 0.55
  own_points_similarity: 0.65
```

If repeated:

```text
repair once
```

If still repeated:

- accept if harmless short acknowledgement;
- otherwise use deterministic phase-safe fallback.

### 8.2 Option validity check

Detect:

- option letters outside A-D;
- claims that a listed option is unavailable;
- new option names introduced as choices.

Repair if found.

### 8.3 Grounding check

Detect numbers or concrete attributes near option mentions that are not in the option text.

Examples to repair:

```text
"B is only 20 minutes away"
"C costs less"
"D has online multiplayer"
```

unless those details appear in the option text or are general known facts not tied as attributes.

### 8.4 Vote check

During vote phase, participant message must contain exactly one clear vote.

If no clear vote:

```text
repair
```

If multiple votes:

```text
repair
```

### 8.5 Confirmation check

During confirmation, participant must be classifiable as:

```text
yes
no
conditional_yes
```

If unclear:

```text
repair
```

Conditional yes is accepted only if the condition is feasible and does not alter the option.

### 8.6 Moderator check

Moderator may not:

- introduce Option E;
- combine choices unless allowed;
- mention fake option attributes;
- claim consensus before confirmation;
- force-close and call it agreement.

Invalid moderator messages should be replaced by deterministic templates.

---

## 9. Acceptance criteria

Before considering the refactor successful, test at least 20 dialogues.

Required:

1. Every final option is A-D or outcome is honest failure.
2. No valid option is denied as unavailable.
3. Moderator never invents new options.
4. Every participant speaks at least twice in normal 3-person dialogues.
5. Every participant votes.
6. Force-close is rare and honest.
7. Repair attempts are logged.
8. Repetition is visibly reduced.
9. Response lengths differ across personas.
10. Chats sound casual, not formal debate scripts.

A good result is not perfect realism. A good result is:

```text
valid
bounded
varied
non-repetitive
easy to debug
```

---

## 10. What not to add

Do not add more research papers into runtime.

Do not add:

- new personality theories;
- full emotion models;
- full memory retrieval systems;
- multi-agent debate frameworks;
- complex social graphs;
- extra moderator roles;
- second LLM judge inside the generation loop.

If later evaluation needs an LLM judge, run it offline after dialogue generation.

---

## 11. Final target

The final simulator should be built around this loop:

```text
choose speaker
generate one message
verify message
repair once if needed
store turn
update simple state
move phase if needed
```

This is the main architectural target.

Naturalness should come from:

- compact personas;
- varied response lengths;
- local conversational prompting;
- short memory;
- moderate randomness;
- anti-repetition repair.

Validity should come from:

- full option visibility;
- deterministic verification;
- explicit votes;
- acceptability-based compromise;
- controlled moderator templates.

That is enough for the project.
