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

## Scenario schema

A scenario is exactly `topic` + `shared_context` + `options`. Shared context is the public source of truth: facts every participant knows (group constraints, hard caps, timing). Each option card has `id`, `name`, `short_name`, `attrs`, `upside`, and `concern` — no `decision_kind`, generated `opening_question`, `tradeoff`, or `best_for` fields exist.

Attributes are topic-specific and chosen by the setup LLM; the prompt gives no example dimensions and the code hard-codes no preferred ones. `short_name` is a required concise natural alias (unique, copied from the name, never derived by clipping). The moderator opening is fixed and neutral: board + context, then "Let's discuss which option fits best overall."

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
  - `stubbornness`: strength of stance defense during the discussion;
  - `switch_resistance`: resistance to final movement — candidate switches, compromise acceptance, holdout concession, and vote/repair behavior.

`speech_style` is small register coloring derived from age (four compact bands: young casual / relaxed practical / direct workplace / measured traditional wording). It changes wording only and must not override stance, vote choice, willingness to compromise, or turn-taking behavior. Hard blockers come only from `rejection`, never from high stubbornness alone.

Generated and manual profiles are checked for obvious age/profile contradictions. For example, a very young participant should not receive a senior-executive biography, a mortgage-heavy family profile, or decades of experience.

## Phases and threads

There is no content agenda. Global progress is explicit controller phase state:

```text
opening -> discussion -> narrowing -> voting -> closing
narrowing -> discussion            (at most once, when the candidate collapses)
voting -> compromise_repair -> voting | closing
```

Local interaction is tracked as deterministic threads (`question`, `concern`, `blocker`, `comparison`, `repair`) with statuses `hot / cooling / resolved / stale`, option-specific deterministic issue keys, and one deterministic primary thread driving routing. Coverage ("was each option socially processed once?") runs only when no hot thread needs attention. Persona-specific reasons live in `OptionStance.reason_for` / `reason_against`.

## Controller / LLM separation

The controller owns the intended move (`MoveIntent`):

```text
speaker + macro act + route source + target/addressee + option focus + reason
```

The LLM renders one natural message. Validation checks whether the line visibly matches the intended move and stays grounded; routing is read-only, and only the final accepted, parsed turn changes dialogue state (observer). A routed answer, concern response, coverage turn, or vote counts only when the final text visibly realizes it.

The compact macro-act vocabulary is:

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Normal discussion sampling is limited to `support, concern, ask, compare, comment`. `answer` is route-driven by question threads; `process`/`compromise` belong to narrowing and repair; softening is an observed stance effect parsed from visible text, never a routed act.

## Voting and repair

Only formal commitments made during `voting`/`compromise_repair` count toward the outcome; opening leans and discussion support move public stance but never silently become final votes. After vote collection, one bounded repair state machine handles (in priority order) unclear votes, majority holdouts, split votes, and two-person deadlocks — each reason at most once per run, with `switch_resistance` governing final movement and hard blockers never pressured into fake agreement.

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
  -> controller routes: required answer > hot thread > cooling thread > coverage > continuation > normal act
  -> LLM renders one utterance
  -> parse -> validate -> repair/fallback -> append final turn
  -> observer updates threads, coverage, rank table, and progress signature
  -> flow: explicit phases, bounded narrowing, formal votes, one repair state machine
  -> consensus manager computes successful / majority / unresolved from formal commitments
  -> transcript.md, run.json (incl. controller trace), metrics.csv are written
```

## Main modules

- `main.py`: CLI entrypoint for one topic, a topic file, piped topics, or configured manual environment.
- `eval/run_eval_suite.py`: sequential regression suite for important mode combinations and edge cases. Manual eval personas include age/speech-style/profile variation.
- `config.yaml`: provider, environment, participant, pacing, threads, narrowing, routing, validation, and output settings.
- `src/builders.py`: builds automatic/manual scenarios and participants, including age/speech-style/profile validation and initial option-rank compatibility.
- `src/models.py`: stable domain dataclasses (scenario, personas, acts, turns, DialogueState) plus re-exports of the controller state types.
- `src/simulator.py`: converts hidden OCEAN traits into the five simulator parameters and the engagement-based expected turn share.
- `src/dialogue.py`: run orchestration and the generate→parse→validate→repair→append pipeline, turn/trace appends, logging.
- `src/controller/state.py`: controller runtime dataclasses — phases, thread state, repair state.
- `src/controller/threads.py`: deterministic issue keys, thread lifecycle transitions, primary-thread selection.
- `src/controller/policy.py`: read-only route/speaker/act/option/addressee selection returning `MoveIntent`.
- `src/controller/flow.py`: phase transition graph, narrowing readiness/behavior, formal voting, and the repair state machine.
- `src/observer.py`: the single post-turn semantic state-update entry point (threads via the engine, coverage, ranks, progress).
- `src/parsing.py`: pure visible-semantics layer — option references, commitments, question scope, blockers, softening.
- `src/validation.py`: side-effect-free turn validation, thread-aware realization checks, grounding, deterministic fallback.
- `src/prompts.py`: setup, utterance, moderator, repair, and grounding prompts.
- `src/consensus.py`: final outcome from formal visible commitments (voting/compromise_repair phases only).
- `src/logger.py` / `eval/eval.py`: transcripts, structured traces (controller trace, threads, repair history), metrics, and token diagnostics.
- `tests/`: deterministic controller tests (`py -m unittest discover -s tests`).

## Running

Activate the existing project environment, then run:

```powershell
py .\main.py "Choose a restaurant for a group dinner"
```

For eval cases (the suite always runs all cases):

```powershell
py .\eval\run_eval_suite.py
```

Deterministic tests and static check:

```powershell
py -m unittest discover -s tests
py -m compileall -q main.py src eval tests
```
