# Option-grounded multi-user decision simulator

This repository generates configurable multi-user decision discussions with LLM-driven simulated participants.

The project scope is deliberately narrow:

```text
fixed option board + simulated participants + controller-routed discussion + visible decision outcome
```

It is not a generic chatbot, full society simulation, or full Generative-Agents-style memory system. The option board is the factual source of truth. Sims may compare options, ask questions, raise concerns, soften, resist, compromise, and vote, but they must not invent concrete facts outside the configured environment.

## Current architecture

The simulator uses a hybrid dialogue-system design:

```text
symbolic controller + LLM utterance renderer + parser/validator feedback loop
```

The controller owns phase logic, speaker choice, macro-act choice, option focus, narrowing, and outcome rules. The LLM only renders one natural utterance for the controller's current intent.

## Stance model

Private stance is stored as one central per-sim/per-option rank table:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from ranks. There are no separate runtime preference/rejection containers.

The persona setup may also provide a compact compatibility table for each sim and option:

```text
option id -> rank, short reason_for, short reason_against
```

Most options should remain neutral or acceptable. Strong dislikes and hard rejects should be rare and grounded.

## Personas, hidden traits, age, and speech style

Sim generation follows one split:

- **OCEAN traits are hidden setup traits.** They are only used to derive simulator parameters and plausible persona content; they never appear in utterance prompts or routing.
- **Sim attributes** describe who the simulated user is: `id`, `name`, `age`, `background`, `private_goal`, `preferred_options`, `option_stances`, `speech_style`, `rejection`, `rejection_reason`.
- **Simulator parameters** are the only numeric behavior controls:
  - `engagement`: expected speaker frequency / turn share;
  - `verbosity`: average utterance length, realized only as numeric word budgets;
  - `directness`: blunt vs soft wording;
  - `stubbornness`: resistance to changing stance and strength of stance defense.

`speech_style` is small register coloring derived from age (four compact bands: young casual / relaxed practical / direct workplace / measured traditional wording). It changes wording only and must not override stance, vote choice, willingness to compromise, or turn-taking behavior. Hard blockers come only from `rejection`, never from high stubbornness alone.

Generated and manual profiles are checked for obvious age/profile contradictions. For example, a very young participant should not receive a senior-executive biography, a mortgage-heavy family profile, or decades of experience.

## Chat-level discussion agenda

The project no longer uses per-sim scripted agendas to steer the discussion. Remaining pre-vote work is tracked as a chat-level `DialogueState.discussion_agenda` checklist. It covers global discussion needs such as option coverage and major tradeoff coverage.

Persona-specific reasons still exist through `OptionStance.reason_for` and `OptionStance.reason_against`. This keeps personal perspectives without adding a second agenda system.

## Controller / LLM separation

The controller owns the intended move:

```text
speaker + macro act + target/addressee + option focus + reason + intended stance effect
```

The LLM renders one natural message. Validation checks whether the line visibly matches the intended move and stays grounded. State changes are applied through the rank table only after validation.

The compact macro-act vocabulary is:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Only these macro acts are used by routing, prompts, and logs.

## Outcomes

A run ends in exactly one of three outcome states:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement remains after bounded narrowing.

Outcomes are derived from visible transcript evidence only: explicit votes, acceptances, and parsed visible commitments. Private stance ranks guide routing, but they do not directly decide the final result.

## High-level pipeline

```text
CLI topic or manual environment
  -> scenario / option board
  -> automatic or manual simulated participants
  -> age/profile/speech-style plausibility checks
  -> initial per-sim option ranks
  -> chat-level discussion agenda
  -> controller chooses speaker, target, macro act, and option focus
  -> LLM renders one utterance
  -> validation checks intent alignment and grounding
  -> observer updates visible state, rank table, and agenda progress
  -> controller routes follow-ups, concerns, narrowing, votes, and closure
  -> consensus manager computes successful / majority / unresolved
  -> transcript.md, run.json, metrics.csv are written
```

## Main modules

- `main.py`: CLI entrypoint for one topic, a topic file, piped topics, or configured manual environment.
- `eval/run_eval_suite.py`: sequential regression suite for important mode combinations and edge cases. Manual eval personas include age/speech-style/profile variation.
- `config.yaml`: provider, environment, participant, pacing, routing, validation, and output settings.
- `src/builders.py`: builds automatic/manual scenarios and participants, including age/speech-style/profile validation and initial option-rank compatibility.
- `src/models.py`: dataclasses, compact macro acts, age/speech-style/profile fields, chat-level agenda items, and per-option stance ranks.
- `src/simulator.py`: converts hidden OCEAN traits into the four simulator parameters and the engagement-based expected turn share.
- `src/dialogue.py`: orchestration loop for opening, discussion, voting, split narrowing, and closure.
- `src/policy.py`: speaker choice, macro-act choice, addressee choice, vote readiness, and procedural routing.
- `src/observer.py`: validated visible-state updates, rank movements, question/concern tracking, and agenda progress.
- `src/parsing.py`: option references, commitments, votes, rejections, and parser-safe phrase families.
- `src/validation.py`: turn validation, parser/intent alignment, minimal fallback protection, and grounding checks.
- `src/prompts.py`: setup, utterance, moderator, repair, and grounding prompts.
- `src/consensus.py`: final outcome computation from visible evidence.
- `src/logger.py` / `eval/eval.py`: transcripts, structured traces, stance-rank metrics, age/speech-style/profile logging, and token diagnostics.

## Running

Activate the existing project environment, then run:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For eval cases (the suite always runs all cases):

```powershell
py .\eval\run_eval_suite.py
```

Static check:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```
