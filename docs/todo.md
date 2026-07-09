# TODO: v3 stabilization and simplification

This file is the active work queue. It should contain only open work.

The project remains an **option-grounded multi-user decision simulator** with exactly three outcomes:

- `successful`
- `majority`
- `unresolved`

## Required protocol

1. Work on one issue at a time.
2. Move all existing logs into logs/archive.
3. Prefer replacing old logic over adding parallel logic.
4. Run static checks before packaging:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

5. Run some example chats with random topics. At least one n=3 run, then one or two more with n=2-7.
6. Inspect transcripts manually and verify changes were implemented correctly.
7. Pick the next todo from the list.
8. Once you're sure all todos are done and everything works, run `eval\run_eval_suite.py` locally.
9. Update `CLAUDE.md`, `README.md`, and `info/*.md` to reflect the current state of the code, then remove completed todos from this file.

## Target design

Use this split everywhere:

- **OCEAN traits** are hidden setup traits. They are only used to derive simulator parameters and plausible persona content.
- **Sim attributes** describe who the simulated user is and what they care about:
  - `id`
  - `name`
  - `age`
  - `background`
  - `private_goal`
  - `preferred_options`
  - `option_stances`
  - `speech_style`
  - `rejection`
  - `rejection_reason`
- **Simulator parameters** are the only numeric behavior controls:
  - `engagement`: expected speaker frequency / turn share
  - `verbosity`: average utterance length through numeric word budgets
  - `directness`: blunt vs soft wording
  - `stubbornness`: resistance to changing stance and strength of stance defense

Do not pass OCEAN traits into participant utterance prompts. Do not pass `engagement` or `verbosity` as prose. `verbosity` should only become a word budget such as `Max words: 24`.

## Todo 1: Simplify simulator parameter model

Replace the current `SimulatorParameters` fields with exactly:

```python
engagement: float
verbosity: float
directness: float
stubbornness: float
```

Remove these fields from `SimulatorParameters`:

```python
initiative
responsiveness
compromise_threshold
```

Keep `Persona.sim_params: SimulatorParameters`. This is still the central place where behavior parameters are attached to a persona.

Update all dataclass construction, clipping, serialization, logging, manual config overrides, and eval fixtures so no old parameter names remain.

Expected result:

- No code path accesses `sim_params.initiative`.
- No code path accesses `sim_params.responsiveness`.
- No code path accesses `sim_params.compromise_threshold`.
- Manual profile `parameters` only accepts `engagement`, `verbosity`, `directness`, and `stubbornness`.

## Todo 2: Replace OCEAN-to-parameter derivation

Update `src/simulator.py::derive_simulator_parameters()` so OCEAN remains hidden and derives only the four final parameters.

Use normalized OCEAN values:

```python
open01 = (traits.openness - 1) / 4
consc01 = (traits.conscientiousness - 1) / 4
extra01 = (traits.extraversion - 1) / 4
agree01 = (traits.agreeableness - 1) / 4
neuro01 = (traits.neuroticism - 1) / 4
```

Then derive:

```python
engagement = 0.25 + 0.60 * extra01 + 0.15 * consc01
verbosity = 0.20 + 0.55 * extra01 + 0.25 * open01
directness = 0.25 + 0.35 * consc01 + 0.25 * extra01 + 0.15 * (1.0 - agree01)
stubbornness = (
    0.45 * (1.0 - agree01)
  + 0.25 * neuro01
  + 0.20 * (1.0 - open01)
  + 0.10 * consc01
)
```

Keep `.clipped()` so every parameter stays in `[0.0, 1.0]`.

Expected result:

- `TraitProfile.compromise_willingness` is no longer needed by runtime logic. Remove it if no remaining code uses it.
- High agreeableness generally lowers stubbornness.
- Low agreeableness, high neuroticism, low openness, and high conscientiousness increase stubbornness.
- Hard blockers still come only from `rejection`, not from `stubbornness == 1.0`.

## Todo 3: Rename `style` to `speech_style`

Rename persona field:

```python
style -> speech_style
```

Update all code and config paths that currently use `style` for persona wording, including:

- `src/models.py`
- `src/builders.py`
- `src/config_loader.py`
- `src/prompts.py`
- `src/logger.py`
- `eval/run_eval_suite.py`
- `config.yaml`
- docs and info files

Do not keep both names long-term. Replace old usage instead of maintaining parallel fields.

Expected result:

- Transcript metadata says `age/speech_style`, not `age/style`.
- Manual profiles use `speech_style`, not `style`.
- No runtime code accesses `persona.style`.

## Todo 4: Simplify age-to-speech-style generation

Replace the current long style strings with four compact age bands:

```text
18-27 -> young casual wording
28-40 -> relaxed practical wording
41-58 -> direct workplace wording
59+   -> measured traditional wording
```

Keep age plausible for the background and goal. Default generated age range should remain adult, e.g. `18-75`. Only include minors if the scenario explicitly fits minors.

Expected result:

- `speech_style` is only small register coloring.
- No phrase palette is introduced.
- No `speech_style` string should encode preferences, decision behavior, turn length, compromise behavior, or directness.

## Todo 5: Make engagement the only participation-share parameter

Update `src/simulator.py::expected_turn_share()` so expected participation share depends only on `engagement` plus a small floor.

Conceptual target:

```python
raw = {p.id: 0.30 + p.sim_params.engagement for p in personas}
```

Then normalize the raw values to shares.

Update speaker selection in `src/policy.py` so general turn frequency uses:

- engagement-based expected share
- actual share versus expected share
- recent-speaker penalty
- anti-monopoly dampening
- small randomness where already used

Do not equalize turn counts. A low-engagement sim should not be pulled up to the same number of turns as a high-engagement sim. Compare each sim to their own expected share.

Expected result:

- High engagement usually produces more participant turns.
- Quiet sims still appear, but they do not get artificially equalized.
- No initiative/responsiveness terms remain in speaker-selection scores.

## Todo 6: Make verbosity control only word budgets

Update `_word_bounds()` in `src/policy.py` and `_expected_words()` in `eval/eval.py` so verbosity is the only persona parameter affecting average length.

Remove engagement from word-budget calculation.

Conceptual target:

```python
factor = 0.45 + 0.85 * p.verbosity
```

Keep the existing idea that verbosity is an average, not a fixed template:

- every sim may sometimes produce a short beat
- low verbosity should produce more short beats
- high verbosity should have higher average words per utterance

Expected result:

- `verbosity` is not described in the prompt as prose.
- Prompt receives only the computed word range / max words.
- Eval expectation for verbosity matches the controller's word-budget formula.

## Todo 7: Replace responsiveness with deterministic question/answer obligations

Remove responsiveness-based delay for directly addressed questions.

Target behavior:

```text
If A directly asks B a question, B answers next unless a stronger validation/safety condition prevents it.
```

For group-directed questions, choose a respondent by a weighted score using:

- relevance to the question / option focus
- engagement
- expected-share deficit relative to that sim's own expected share
- recent-speaker penalty
- small randomness

Do not choose the absolutely quietest person by default. Do not use `responsiveness`.

Update at least:

- `src/policy.py::_route_discussion_turn()`
- `src/observer.py::_pick_group_respondent()`
- any dialogue helper that selects a respondent using `responsiveness`

Expected result:

- Directly addressed questions are answered promptly.
- Group questions are answered by plausible participants without flattening engagement differences.
- No code path accesses `sim_params.responsiveness`.

## Todo 8: Replace initiative with dialogue-state move logic

Do not replace `initiative` with `engagement` everywhere. That would overload engagement.

Instead, remove initiative-based behavior and route proactive moves from dialogue state.

Examples:

- A question should happen because there is an unresolved issue or useful comparison to ask about.
- A process move should happen because the discussion is stuck, a quiet relevant participant should be invited, or narrowing is needed.
- A compromise move should happen because support is concentrating or a split needs testing.
- Opening order can use engagement plus light randomness, or just randomized order with optional engagement bias.

Update at least:

- `src/policy.py::_choose_discussion_act()`
- `src/policy.py::_choose_speaker()`
- `src/policy.py::_opening_order()`
- `src/dialogue.py::_procedural_speaker()`
- participant-owned narrowing/probe helpers in `src/dialogue.py`

Expected result:

- No code path accesses `sim_params.initiative`.
- Engagement remains speaker frequency, not a generic replacement for initiative.
- Proposals, questions, process moves, and compromise attempts are driven by conversation state.

## Todo 9: Merge compromise threshold into stubbornness

Replace all switching, compromise, holdout, pacing, and resistance logic that currently uses `compromise_threshold` with `stubbornness`.

Use this meaning consistently:

```text
stubbornness high = very resistant, but theoretically movable
rejection true = hard blocker / cannot accept rejected option
```

Suggested replacements:

- `1.0 - compromise_threshold` becomes `1.0 - stubbornness`
- `compromise_threshold >= X` becomes `stubbornness >= X`
- combined formulas using both `stubbornness` and `compromise_threshold` should be simplified, not duplicated

Update at least:

- `src/policy.py`
- `src/dialogue.py`
- `src/observer.py`
- `src/prompts.py::_trait_phrase_preferences()`
- `src/logger.py`
- eval fixtures in `eval/run_eval_suite.py`

Expected result:

- No code path accesses `sim_params.compromise_threshold`.
- High-stubbornness sims defend longer and switch less often.
- Low-stubbornness sims compromise and soften more easily.
- Hard blockers remain controlled by `rejection` and `option_stances` rank 1, not by stubbornness alone.

## Todo 10: Simplify participant utterance prompts

Update `src/prompts.py::sim_utterance()` so the prompt only receives useful visible/persona information.

Include:

- name
- age
- background
- private goal
- preferred/current stance
- relevant option stances
- rejection only if hardblocker/relevant
- `speech_style`
- `directness`
- `stubbornness`
- numeric word limit

Do not include:

- OCEAN traits
- engagement
- verbosity prose
- initiative
- responsiveness
- compromise threshold
- phrase palettes for speech style

Use compact wording, for example:

```text
Speech style: relaxed practical wording.
Directness: 4/5.
Stubbornness: 3/5.
Max words: 24.
```

If code keeps parameters in `[0.0, 1.0]`, either pass those directly or map once to `1-5`. Do not add low/medium/high labels unless they already exist and are needed.

Expected result:

- The prompt does not duplicate controller-owned behavior.
- `verbosity` is represented only through the word budget.
- `engagement` is not visible to the LLM.
- `directness` and `speech_style` remain separate: speech style is register; directness is bluntness.

## Todo 11: Update manual configs and eval fixtures

Update all manual profiles in `eval/run_eval_suite.py` and any sample config profiles so they use the final parameter and attribute names.

Manual `parameters` should contain only:

```yaml
engagement: ...
verbosity: ...
directness: ...
stubbornness: ...
```

Manual profile style field should be:

```yaml
speech_style: ...
```

When old fixtures used both `stubbornness` and `compromise_threshold`, collapse them into one `stubbornness` value that preserves the intended behavior of the case.

Example:

```yaml
# old
stubbornness: 0.85
compromise_threshold: 0.80

# new
stubbornness: 0.85
```

Expected result:

- Eval suite runs without old parameter names.
- Fixture intent remains: active sims are still active, quiet sims are still quiet, hard holdouts still hold out, flexible bridge sims remain flexible.

## Todo 12: Update evaluation labels and metrics

Update eval code and run JSON/transcript expectations so they match the final parameter meanings.

Keep these core mappings:

```text
engagement -> turn share / free-discussion turn share
verbosity -> average words per participant turn
directness -> optional/manual or heuristic wording signal
stubbornness -> fewer/later switches and stronger stance defense
speech_style -> manual qualitative check only
```

Remove metric labels or comments that mention:

- initiative
- responsiveness
- compromise threshold

Update expected-word formulas to match the new word-budget code.

Expected result:

- Eval no longer claims to measure removed parameters.
- Engagement realization is based on the same expected-turn-share function the router uses.
- Verbosity realization is based on the same word-budget formula the controller uses.

## Todo 13: Update logs, transcript metadata, and docs

Update logging and documentation after code behavior is stable.

Required updates:

- `src/logger.py`: transcript should list only the final four sim parameters.
- `README.md`: explain the simplified sim generation model.
- `CLAUDE.md`: update implementation guidance and run protocol.
- `info/02_sim_generation.md`: explain OCEAN -> parameters -> attributes.
- `info/03_routing_and_turn_taking.md`: explain engagement-based routing and question obligations.
- `info/05_discussion_and_decision.md`: explain stubbornness/rejection distinction.
- `info/07_evaluation_and_logging.md`: update eval metric meanings.
- `info/08_configuration_and_running.md`: update manual profile field names.

Expected result:

- Documentation matches the actual code.
- No docs describe initiative, responsiveness, or compromise threshold as active parameters.
- No docs use `style` when they mean `speech_style`.
