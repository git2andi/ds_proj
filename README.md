# Option-grounded multi-user decision simulator

This repository generates configurable multi-user decision discussions with LLM-driven simulated participants.

The project scope is deliberately narrow:

```text
fixed option board + simulated participants + simulator-driven discussion + visible decision outcome
```

It is not a generic chatbot, full society simulation, or full Generative-Agents-style memory system. The option board is the factual source of truth. Sims may compare options, ask questions, raise concerns, soften, resist, compromise, and vote, but they must not invent concrete facts outside the configured environment.

## Current architecture

The simulator uses a controlled hybrid design — a Python simulator policy per
participant, a floor manager, and a dialogue-LLM renderer — **not** unrestricted
autonomous agents:

```text
simulator policy (per sim) + floor manager + dialogue-LLM utterance renderer + deterministic critical interpretation
```

Each simulated user is the behavioral decision maker. Given its own persona,
private stance, and the public dialogue state, a simulator decides whether it
wants to speak, computes its own willingness, and — if it claims the floor —
chooses one complete intended move (act, target, option focus, direction,
reason, and vote/compromise). The **floor manager** (`controller/floor.py`)
collects one bid per eligible simulator, validates them structurally, adjusts
floor access (recent-speaker penalty, anti-monopoly damping, minimum-visibility
correction — never engagement twice), and selects the highest-scoring valid
bid **without rewriting its act, focus, target, reason, or vote**. The **flow**
(`controller/flow.py`) owns phases, protocol obligations, bid-round
orchestration, narrowing/vote readiness, the bounded repair machine, and
termination. The dialogue LLM owns all generative calls (scenario/persona setup,
one utterance per winning bid, moderator lines, one bounded repair rewrite);
there is exactly one participant LLM call per ordinary turn. Normal runtime
validation is deterministic; the runner never constructs or calls a validator
LLM in `validation.mode: critical`.

Opening turns and formal votes are protocol-required and a direct question is a
mandatory adjacency pair, but the simulator still chooses the substance (opening
position, answer direction, vote target, compromise). Group questions are
answered by self-selection. Threads are public stimuli that shape simulator bids, never scripts that dictate who reacts or how. A compact public participant ledger gives each simulator the other participants' visible positions, concerns, recent acts, and question relationships without exposing private ranks, goals, reasons, or parameters.

The governing authority order is:

```text
scenario/shared-context facts   authoritative for listed facts
simulator intent                authoritative only for the requested function of the winning bid
visible utterance               authoritative for what was publicly said
accepted deterministic evidence authoritative for state updates
```

A bid or intended move never creates public support, a vote, a switch, or a
blocker. Each candidate utterance is extracted conservatively, resolved against
known option aliases, interpreted into the small visible-evidence model, and
checked only for correctness-critical failures: malformed output, invalid option
references, missing required question/focus, ambiguous formal commitment,
invalid switch, blocked-option acceptance, hybrid compromise, or exact
cross-option value contradictions. Ordinary opinions, support, concerns,
comparisons, and reasonable implications are not sent through a second semantic
model. At most one repair is attempted; only narrow truthful fallbacks remain.

## Scenario schema

A scenario is exactly `topic` + `shared_context` + `options`. Shared context is the public source of truth: facts every participant knows (group constraints, hard caps, timing). Each option card has `id`, `name`, `short_name`, `attrs`, `upside`, and `concern` — no `decision_kind`, generated `opening_question`, `tradeoff`, or `best_for` fields exist.

Attributes are topic-specific and chosen by the setup LLM; the prompt gives no example dimensions and the code hard-codes no preferred ones. `short_name` is a required concise natural alias (unique, copied from the name, never derived by clipping). Every option attribute, upside, and concern available to participant generation is printed on the public board; no hidden fourth attribute or clipped card fact may influence a turn. The moderator opening is fixed and neutral: board + context, then "Let's discuss which option fits best overall."

## Stance model

Private stance is stored as one central per-sim/per-option rank table:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from ranks. There are no separate runtime preference/rejection containers and no hidden commitment/confidence float: ranks (plus their short stored reasons) are the only persistent private stance state, and only accepted visible utterances move them. Public candidate scores used by narrowing/voting come only from accepted visible backing, formal votes, positive evidence, existing-option proposals, and objection load; private preference values never fill missing evidence.

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

Local interaction is tracked as deterministic threads (`question`, `concern`, `blocker`, `comparison`) with statuses `hot / cooling / resolved / stale`, option-specific deterministic issue keys, and per-thread contribution caps. Threads are public stimuli: a hot thread raises the relevant participant-local scores inside each simulator's bid, but the thread engine never picks who reacts or which reaction they make. Coverage ("was each option socially processed once?") becomes a relevance bonus to simulators that actually care about the ignored option and, if still uncovered, a moderator group question — never a forced participant turn. Persona-specific reasons live in `OptionStance.reason_for` / `reason_against`. Repeated empty bid rounds cannot bypass the minimum discussion gate: before narrowing, the moderator asks a concise public group question or a stronger public stall/coverage stimulus is offered; only the hard interaction cap can terminate an all-silent discussion early.

## Simulator / floor / LLM separation

Each simulator produces a `SimulatorBid`: whether it wants to speak, a normalized willingness, and — when it claims the floor — one complete `MoveIntent`:

```text
speaker + macro act + authority source + target/addressee + option focus + reason (+ vote/compromise)
```

The floor manager selects among complete bids and may reject or reorder them (recent-speaker penalty, anti-monopoly damping, minimum-visibility correction, structural validity), but it never rewrites a bid's act, focus, target, reason, or vote. The dialogue LLM then renders one natural message for the winning intent against a compact realization contract: voice (age/register/directness/stubbornness cues), one act-specific semantic requirement, one turn objective, focus-only option facts, and a soft word range — returned inside an `<utterance>` envelope. Cleanup is structural only (envelope extraction, one speaker prefix, one quote pair, whitespace) and never deletes semantic content or clips to a word budget. Every selected SUPPORT, CONCERN, ASK, targeted ANSWER, COMPARE, COMPROMISE, and VOTE must be visibly realized with its selected focus; otherwise one bounded repair runs and the turn is dropped if still invalid. COMMENT and PROCESS remain semantically flexible but cannot fabricate evidence. Bidding and floor arbitration are read-only, and only the final accepted evidence object changes dialogue state (observer) — the observer never reparses text, updates are option-specific, and only a speaker's own accepted utterance can move that speaker's private ranks or vote.

The compact macro-act vocabulary is:

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Open-floor self-selection samples `answer, support, concern, ask, compare, comment, compromise` when public state provides a concrete contribution; `process` is available only under a stall stimulus. COMMENT has no generic filler baseline, and silence is valid. `opening`, `answer`, and `vote` are obligation/protocol acts whose substance the simulator still owns; softening is an observed stance effect parsed from visible text, never a chosen act.

## Voting and repair

Only formal commitments made during `voting`/`compromise_repair` count toward the outcome; opening leans and discussion support never silently become final votes. Narrowing creates one public group stimulus; any relevant simulator may react, but silence does not authorize the framework to invent a response. After one complete vote round, unanimity and clear majorities close immediately. A bare one-vote majority receives one bounded concern/response/re-vote round. A no-majority split tests one existing option once and gives visible dissenters one bounded reservation/re-vote opportunity. Candidate selection uses visible votes, positive discussion evidence, proposals, and objection load with deterministic tie-breaking. Each simulator may stay or switch to the tested candidate; repair cannot introduce a third option, and majority or unresolved outcomes remain valid.

## Outcomes

A run ends in exactly one of three outcome states:

- `successful`: all visible final stances support the same option.
- `majority`: a majority visibly supports the winning option.
- `unresolved`: no sufficient agreement remains after bounded narrowing.

Outcomes are derived from visible transcript evidence only: explicit votes, acceptances, and parsed visible commitments. Private stance ranks guide each simulator's own decisions, but they do not directly decide the final result, and the framework never engineers consensus by choosing a participant's vote — majority and unresolved outcomes remain legitimate.

## High-level pipeline

```text
CLI topic or configured manual environment
  -> scenario and participant setup
  -> every eligible simulator submits a bid; the floor manager validates and selects a winner
     (or the framework imposes a protocol obligation: opening, direct answer, vote)
  -> dialogue LLM renders one visible utterance for the winning simulator intent
  -> conservative extraction
  -> deterministic critical interpretation and validation
  -> at most one repair for a correctness-critical failure
  -> minimal vote/switch/unknown-information fallback when safe
  -> observer updates threads, coverage, visible stance, ranks, commitments, and blockers
  -> bounded narrowing and one complete formal vote round
  -> immediate closure for unanimity or a clear majority
  -> one bounded repair round for a bare majority or unresolved split
  -> deterministic status-correct closure
  -> transcript.md, run.json, and concise metrics.csv
```

A bid or hidden simulator intent never counts as public evidence. Ordinary support, concerns, opinions,
and reasonable inferences are not sent through a runtime validator LLM. Runtime validation is
strict only for malformed output, option references, required questions/focus, formal votes,
public switches, blockers, existing-option compromise, transferred exact values, unlisted exact quantities,
and explicit unlisted feature/location claims.

## Main modules

- `main.py`: CLI entrypoint.
- `eval/run_eval_suite.py`: ten-case live regression suite with behavioral and efficiency flags.
- `config.yaml`: dialogue LLM, environment, participants, pacing, threads, floor arbitration, critical validation, and output settings.
- `src/builders.py`: scenario and participant construction.
- `src/models.py`: domain and runtime dataclasses (incl. `SimulatorBid`, `TurnObligation`, `DiscussionStimulus`).
- `src/simulator.py`: the simulator policy — OCEAN→parameter derivation, per-sim willingness, act scoring, and complete bid selection (owns participant behavior).
- `src/dialogue.py`: orchestration and bounded generate→validate→repair→append lifecycle.
- `src/controller/state.py`: phases, threads, and repair state.
- `src/controller/threads.py`: deterministic thread lifecycle (threads as stimuli).
- `src/controller/floor.py`: floor arbitration — collect/validate/score/select bids without rewriting them; framework public-evidence readers.
- `src/controller/flow.py`: phases, protocol obligations, bid-round orchestration, narrowing, voting, one-round majority/split repair, and closure.
- `src/interpreter.py`: deterministic visible-evidence interpretation; no LLM calls.
- `src/parsing.py`: option/alias, commitment, blocker, switch, and question parsing.
- `src/validation.py`: correctness-critical assessment and minimal safe fallbacks.
- `src/observer.py`: the single accepted-turn state mutation path.
- `src/consensus.py`: current public backing and transcript-derived formal tally.
- `src/prompts.py`: setup, moderator, utterance, and critical repair prompts.
- `src/logger.py` / `eval/eval.py`: trace artifacts and concise grouped metrics.
- `tests/`: deterministic regression tests (`py -m pytest -q`).

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

### Realization and protocol integrity

Open-floor silence and wording/validation failure are tracked separately. A round with valid simulator bids that cannot be realized does not count as a no-claim round and cannot accelerate silence-based narrowing. Required openings, direct answers, and formal decisions use bounded realization retries and fail explicitly instead of being silently skipped.

Natural support, concern, and conditional-compromise paraphrases are interpreted into option-specific public evidence. Accepted simulator-owned conditional willingness persists as an acceptable runtime option, while stronger explicit movement may change the preferred option. Ordinary contribution keys are enforced across each simulator's complete accepted history to prevent non-consecutive repetition.
