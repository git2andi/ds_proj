# CLAUDE.md

Working instructions for coding agents in this repository.

## Project framing

This is a university project for an option-grounded multi-user decision simulator. The system simulates 2-7 LLM-driven participants discussing a fixed set of options. The goal is a configurable simulator whose participant parameters visibly affect turn-taking, stance movement, disagreement, compromise, and final outcomes.

The architecture is **simulator-driven**: each simulated user is the decision maker, not an LLM
voice realizing controller-authored moves. The authority split is:

```text
simulator policy (src/simulator.py) -> each sim decides its own willingness to speak and, if it
                                        claims the floor, one complete intended move
                                        (act/target/option-focus/reason/vote/compromise)
floor manager (controller/floor.py)  -> collects one bid per eligible sim, validates them
                                        structurally, adjusts floor access (recent-speaker /
                                        anti-monopoly / min-visibility), and selects a winner
                                        WITHOUT rewriting its act/focus/target/reason/vote
dialogue LLM                         -> realizes the winning intent as one visible utterance
observer (observer.py)               -> updates public state only from the final accepted text
flow (controller/flow.py)            -> phases, protocol obligations, bid rounds, termination
```

Per turn: build stimulus -> collect bids -> validate -> pick winner -> generate one utterance ->
conservative extraction -> deterministic critical interpretation/validation -> at most one critical
repair, else next-best bid / minimal safe fallback / drop -> observer updates state from accepted
text. Opening and voting are protocol-required turns and direct questions are mandatory adjacency
pairs, but the simulator still chooses the substance (opening position, answer direction, vote
target, compromise). Group questions use self-selection. A derived public participant ledger exposes only visible positions, concerns, recent acts, and question relationships. Narrowing/vote readiness, repair detection, and outcome calculation remain framework responsibilities.

The dialogue LLM renders utterances only. Runtime validation is deterministic and normally makes
zero validator-LLM calls (`validation.mode: critical`). No bid or intended move ever becomes public
support, acceptance, a vote, a switch, or a blocker — only accepted visible text does. The floor
manager applies engagement only through the simulator's willingness (never a second time); relevance
and personal stake can outweigh engagement. Selected SUPPORT, CONCERN, ASK, targeted ANSWER, COMPARE, COMPROMISE, and VOTE functions are correctness-critical: they must be visible after at most one repair. Formal commitments, blockers, option identity, and final outcome invariants are strict. The full eval suite
(`py .\eval\run_eval_suite.py`) uses live endpoints and is costly.

## Scenario model

A scenario is exactly `topic` + `shared_context` + `options`; option cards carry `id`, `name`, `short_name`, `attrs`, `upside`, `concern`. There is no `decision_kind`, generated `opening_question`, `tradeoff`, or `best_for`. `shared_context` is the public source of truth known by all participants; personas must align with it. Attributes are topic-natural and chosen by the setup LLM — do not add example dimensions to the setup prompt or hard-code preferred dimensions. `short_name` is required, unique, and never derived by clipping; an invalid generated one rejects the attempt (retry), an invalid manual one is a config error. The moderator opening is fixed and neutral ("Let's discuss which option fits best overall.").

## Current stance model

Private stance is stored as one per-sim/per-option rank table:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

Use `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` as computed helpers over the rank table. Do not add a second independent preference model.

Participant setup may provide short `reason_for` / `reason_against` fields for each option. Keep these short. Leave neutral options neutral. Hard rejects should be rare and grounded.

## Personas, hidden traits, age, and speech style

OCEAN traits are hidden setup traits. They only derive the five simulator parameters and plausible persona content; they are never passed into participant utterance prompts or routing.

The simulator parameters are the only numeric behavior controls:

```text
engagement        -> expected speaker frequency / turn share
verbosity         -> average utterance length (numeric word budgets only, never prose)
directness        -> blunt vs soft wording
stubbornness      -> strength of stance defense during discussion
switch_resistance -> final switching, compromise acceptance, holdout behavior,
                     and voting/decision-repair resistance
```

Age and `speech_style` are surface-realization metadata. `speech_style` is derived from age in four compact bands (18-27 young casual, 28-40 relaxed practical, 41-58 direct workplace, 59+ measured traditional wording) unless manually overridden. It may alter wording, formality, and conversational flavor, but must not alter routing, stance strength, vote choice, compromise willingness, or outcome logic. Hard blockers come only from `rejection`, not from `stubbornness`.

Profiles/backgrounds should be plausible for the generated or manually configured age. The builder performs deterministic checks for obvious contradictions such as a very young participant being described as a senior executive, long-term homeowner, married parent, or having decades of experience.

Manual participant profiles may include `age` and `speech_style`. If omitted, age/speech_style are filled by the builder. Manual `parameters` accept only `engagement`, `verbosity`, `directness`, `stubbornness`, and `switch_resistance`.

## Phases, threads, and no content agenda

There is no discussion agenda of any kind (per-sim or chat-level). Do not reintroduce one. Global progress is explicit controller phase state with a validated transition graph:

```text
opening -> discussion -> narrowing -> voting -> closing
narrowing -> discussion            (at most once, on candidate collapse)
voting -> compromise_repair -> voting | closing
```

Local interaction is thread state (`src/controller/threads.py` is the single owner of thread lifecycle): `question`, `concern`, `blocker`, `comparison` threads with statuses `hot/cooling/resolved/stale`, deterministic option-specific issue keys, and per-thread contribution caps. Threads are **public stimuli, not scripts**: a hot thread influences the participant-local scores in each simulator's bid, but the thread engine never prescribes which sim speaks or which reaction it makes (each sim picks the thread it reacts to). Floor arbitration and simulator policy are both read-only over state; only the final accepted, parsed turn changes dialogue state (observer). Only formal votes from the voting/compromise_repair phases count toward the outcome.

Persona-specific perspectives belong in:

```text
Persona.private_goal
OptionStance.reason_for
OptionStance.reason_against
```

## Compact act vocabulary

Each simulator policy chooses over macro acts:

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Open-floor self-selection samples `answer, support, concern, ask, compare, comment, compromise` only when a concrete public-state contribution exists; `process` is available only under an explicit stall stimulus. COMMENT has no filler baseline and silence is valid. `opening`, `answer`, and `vote` are obligation/protocol acts a simulator produces when the framework schedules that turn — the simulator still owns their substance. Softening is an observed stance effect parsed from visible text, never a chosen act. Do not grow the act list unless there is strong log evidence that a new act cannot be represented by the macro set.

## Implementation principles

- Prefer deterministic simulator/state logic over new LLM calls. There is exactly one participant
  LLM call per ordinary turn: the utterance realization of the winning bid. Do not add a per-sim
  per-turn LLM policy call — the bid policy is Python/symbolic-probabilistic.
- Participant behavior (act/target/option-focus/reason/vote/compromise) is owned by the simulator
  policy in `src/simulator.py`. The floor manager may reject or reorder complete bids but must never
  rewrite one. Do not move behavioral authority back into the controller/floor layer.
- A simulator reads only its own persona/runtime/private ranks plus shared scenario and accepted
  public state — never another simulator's private goal, hidden ranks, or hidden reasons.
- Public candidate selection uses accepted backing, formal votes, positive evidence, proposals, and objection load only. Stable contribution keys are tracked across the full accepted history to prevent recycled points after they leave the prompt window.
- Keep prompts act-specific and compact; the prompt realizes an already-decided intent and must not
  ask the LLM to choose the speaker, act, stance direction, vote, or compromise.
- Use the option-rank table as the stance source of truth.
- Keep the five simulator parameters behavior-relevant; keep speech_style wording-only. Engagement
  drives willingness (once); verbosity affects only length; switch_resistance affects only final
  movement; stubbornness affects only discussion-phase defense/concession.
- Do not add more personality traits or simulator parameters unless required.
- Do not reintroduce any content agenda, per-sim agenda steering, or a centralized speaker/act router.
- Floor arbitration and simulator policy stay read-only; persistent state changes only after the
  final accepted turn is observed.
- Thread lifecycle transitions go through `src/controller/threads.py` only — never assign `thread.status` elsewhere.
- Phase changes go through `_mark_phase` only; the transition graph is validated.
- Normal auto-generated sims should have movable preferences, not categorical blockers.
- Manual hard constraints must remain binding.
- Unresolved outcomes are allowed, but must be earned after narrowing/repair attempts.
- Decision turns should normally parse without repair.
- Invalid lines must not become transcript evidence.

## Development workflow

1. Work on one issue at a time.
2. Inspect the current code and latest logs before editing.
3. Make the smallest coherent change.
4. Run the deterministic tests and static checks:

```powershell
py -m unittest discover -s tests
py -m compileall -q main.py src eval tests
```

5. Before claiming stable behavior, run the eval suite (it always runs all cases) if LLM access is available:

```powershell
py .\eval\run_eval_suite.py
```

6. Inspect `transcript.md` and `run.json` (including `controller_trace`, `threads`, and `repair_history`), not only CSV metrics.
7. Update `README.md`, `docs/todo.md`, and relevant `info/*.md` files when behavior or workflow changes.

## What to inspect in logs

- Does the transcript read like a plausible group decision?
- Are turns grounded in the option board?
- Are direct questions answered by the addressed sim (question threads answered, not stale)?
- Do concern/blocker threads get issue-relevant responses instead of option-name mentions?
- Does the route-source distribution look sane (threads driving local moves, coverage only in quiet moments)?
- Do stance switches have visible reasons, and does `switch_resistance` shape final movement?
- Do final votes respect the sim's rank/visible stance, and are outcomes derived from voting/repair-phase commitments only?
- Are blockers preventing false unanimity?
- Do phase transitions and repair entries in the controller trace explain the run?
- Are age/profile/speech_style fields present and plausible?
- Is speech_style visible without overriding parameter-driven behavior?
- Are repair/fallback/token counts reasonable?
