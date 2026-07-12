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
symbolic controller + dialogue-LLM utterance renderer + validator-LLM semantic interpreter
```

The controller owns phase logic, speaker choice, macro-act choice, option focus, narrowing, and outcome rules. Two independently configurable LLM roles exist (`llm.dialogue` and `llm.validator` in config.yaml; the same provider for both is fine, and no third checker exists):

- the **dialogue role** owns every generative call — scenario/persona setup, participant utterances, moderator lines, and repair rewrites;
- the **validator role** owns structured semantic interpretation of visible utterances, claim classification for grounding, and intended-move alignment; it never generates public dialogue text.

The governing authority order is:

```text
scenario/shared-context facts   authoritative for grounding
controller intent               authoritative for what the turn was asked to realize
visible utterance               authoritative for what was publicly said
validated visible evidence      authoritative for state updates
```

No generator self-report metadata exists and hidden controller intent never overrides contradictory visible text. Each candidate utterance is emitted inside an explicit `<utterance>` envelope, extracted conservatively (structural cleanup only — natural tails and clauses are never deleted), deterministically resolved (options, aliases, addressees, unambiguous public pronoun referents), interpreted by ONE validator call into a typed multi-label evidence object (support, concern, comparison, question, answer, softening, proposal, commitment, switch, blocker, atomic grounding claims — several may coexist in one line), verified deterministically (spans must occur in the utterance, critical votes/blockers must pass the conservative critical parser, claims are checked against a normalized option-attribute-value fact table), and then assessed into one explicit action: `ACCEPT`, `ACCEPT_WITH_METRIC`, `REPAIR`, `FALLBACK`, or `DROP`. Repair is one targeted dialogue-LLM rewrite fed with exact issue explanations and offending spans; fallback is a narrow act-specific deterministic family built only from known truthful information, revalidated through the same complete path; anything else is dropped rather than printed. Validator failures never fail open.

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

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from ranks. There are no separate runtime preference/rejection containers and no hidden commitment/confidence float: ranks (plus their short stored reasons) are the only persistent private stance state, and only accepted visible utterances move them. Public candidate scores used by narrowing/voting are group-level evidence from the transcript, never private preference values.

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
  - `verbosity`: average utterance length, realized only as numeric word budgets (soft generation targets — accepted utterances are never cut to length);
  - `directness`: blunt vs soft wording;
  - `stubbornness`: strength of stance defense during the discussion;
  - `switch_resistance`: resistance to final movement — candidate switches, compromise acceptance, holdout concession, and vote/repair behavior.

`speech_style` is small register coloring derived from age (four compact bands: young casual / relaxed practical / direct workplace / measured traditional wording). It changes wording only and must not override stance, vote choice, willingness to compromise, or turn-taking behavior.

Hard blockers come only from rank-1 rejections, never from high stubbornness alone. The configured `hard_blocker_probability` is a low **group-level** probability: when the event is sampled, exactly one participant becomes an exclusive hard blocker — one preferred option at rank 5, every other option hard-rejected at rank 1 with a grounded reason, and a background/goal stating the one non-negotiable requirement (they may still speak politely). When it is not sampled, every participant stays movable according to ranks and traits; manual profiles can still bind a single option via `rejection`.

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

The dialogue LLM renders one natural message against a compact realization contract: voice (age/register/directness/stubbornness cues), one act-specific semantic requirement, one turn objective, focus-only option facts, and a soft word range — returned inside an `<utterance>` envelope. Cleanup is structural only (envelope extraction, one speaker prefix, one quote pair, whitespace) and never deletes semantic content or clips to a word budget. Every intended move has the same visible-realization check on every route: what matters is whether the requested FUNCTION was realized, not whether the primary label matches (a comparative question realizes a requested comparison). Routing is read-only, and only the final accepted evidence object changes dialogue state (observer) — the observer never reparses text, updates are option-specific, and only a speaker's own accepted utterance can move that speaker's private ranks or vote.

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
CLI topic (always wins) or configured manual environment
  -> scenario / option board (invalid generated aliases get a small alias-only repair call)
  -> automatic or manual simulated participants
  -> age/profile/speech-style plausibility checks
  -> initial per-sim option ranks
  -> controller routes: required answer > hot thread > cooling thread > coverage > continuation > normal act
  -> dialogue LLM renders one enveloped utterance
  -> conservative extraction -> deterministic critical layer (mentions, strict
     commitments + post-checks, explicit blockers, genuine questions)
  -> selective validator LLM call ONLY when soft meaning can change state,
     requesting just the categories the intended move needs (+ grounding claims);
     simple fully-verifiable turns skip it via explicit deterministic fast paths
  -> deterministic verification of every span/id/binding + fact-table grounding
  -> assessment: ACCEPT / ACCEPT_WITH_METRIC / REPAIR / FALLBACK / DROP
     (repair only for blocking failures; truthful state-safe fallback families only)
  -> observer consumes the accepted evidence object: threads, coverage, rank table, progress
  -> consensus/public support consume the SAME accepted evidence (single semantic authority)
  -> flow: explicit phases, bounded narrowing, formal votes, one repair state machine
  -> transcript.md, run.json (incl. controller trace + validation telemetry), metrics.csv
```

## Main modules

- `main.py`: CLI entrypoint for one topic, a topic file, piped topics, or configured manual environment.
- `eval/run_eval_suite.py`: sequential regression suite for important mode combinations and edge cases. Manual eval personas include age/speech-style/profile variation.
- `config.yaml`: LLM roles (`llm.dialogue` / `llm.validator`), environment, participant, pacing, threads, narrowing, routing, validation mode (`validation.mode: selective | full`), and output settings. Safety-critical deterministic checks (commitments, blockers, grounding of accepted claims) are always active in both modes.
- `src/builders.py`: builds automatic/manual scenarios and participants, including age/speech-style/profile validation and initial option-rank compatibility.
- `src/models.py`: stable domain dataclasses (scenario, personas, acts, turns, DialogueState) plus re-exports of the controller state types.
- `src/simulator.py`: converts hidden OCEAN traits into the five simulator parameters and the engagement-based expected turn share.
- `src/dialogue.py`: run orchestration and the generate→parse→validate→repair→append pipeline, turn/trace appends, logging.
- `src/controller/state.py`: controller runtime dataclasses — phases, thread state, repair state.
- `src/controller/threads.py`: deterministic issue keys, thread lifecycle transitions, primary-thread selection.
- `src/controller/policy.py`: read-only route/speaker/act/option/addressee selection returning `MoveIntent`.
- `src/controller/flow.py`: phase transition graph, narrowing readiness/behavior, formal voting, and the repair state machine.
- `src/observer.py`: the single post-turn state-update entry point, consuming the accepted evidence object (threads via the engine, coverage, ranks, progress).
- `src/interpreter.py`: the deterministic critical layer, the selective validator-LLM call (intent-specific payload, explicit fast paths), deterministic verification of validator output, and the normalized fact table for claim-level grounding.
- `src/parsing.py`: deterministic critical parser/resolver — options, aliases, addressees, public pronoun referents, strict commitments with post-checks, strict blockers, genuine questions. Soft semantics belong to the validator role only.
- `src/validation.py`: evidence-based candidate assessment (repair only for blocking failures) and the truthful state-safe fallback families (vote/switch, blocker restatement, coverage request, factual comparison, listed/does-not-say answer).
- `src/prompts.py`: setup, utterance, moderator, repair, and validator-interpretation prompts.
- `src/consensus.py`: final outcome from formal visible commitments (voting/compromise_repair phases only).
- `src/logger.py` / `eval/eval.py`: transcripts, structured traces (controller trace, threads, repair history), metrics, and token diagnostics.
- `tests/`: deterministic controller tests (`py -m unittest discover -s tests`).

## Running

Activate the existing project environment, then run:

```powershell
py .\main.py                                          # interactive prompt (auto) / configured environment (manual)
py .\main.py "Choose a restaurant for a group dinner" # explicit topic
py .\main.py topics.txt                               # batch file (# comments and blank lines skipped)
"Choose a restaurant" | py .\main.py                  # piped topic(s)
```

An explicit CLI/piped topic always requests automatic scenario generation for
that topic, even when `environment.mode` is `manual`; explicit input is never
silently discarded. Without an explicit topic, manual mode runs the configured
environment once and auto mode prompts interactively. `participants.mode` is
independent: manual profiles combine freely with a CLI-generated scenario.

The full eval suite is a costly, explicitly approved operation (it runs every
case against the live LLM endpoints) — do not run it casually:

```powershell
py .\eval\run_eval_suite.py
```

Deterministic tests and static check:

```powershell
py -m unittest discover -s tests
py -m compileall -q main.py src eval tests
```
