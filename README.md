# Autonomous Multi-User Decision Simulator

University project: **Implementation of User Simulators in Multi-User Conversational AI**.

The program generates a fixed public option board and persona cards, then simulates a small group discussing the options and voting. Its central design choice is simulator authority: every participant independently decides whether it wants to speak and submits one complete structured action. The controller schedules bids and manages phases, but it does not invent participant decisions.

```text
topic
→ scenario and option cards
→ validated natural option aliases
→ persona cards and private stances
→ simulator-owned bids
→ floor selection
→ LLM wording
→ hard validation
→ public state update
→ pre-vote narrowing and one final vote
```

## Scope

This is a compact option-grounded simulator, not a general model of human group behavior. It aims to produce coherent synthetic discussions with visible trait differences, bounded sub-discussions, possible preference movement, and deterministic outcomes. It does not claim psychological realism, complete factual grounding, or unrestricted natural conversation.

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

Any compromise or preference movement is handled before the single authoritative voting round.

## Simulator authority

Each `UserSimulator`:

- constructs actions valid for its persona and the visible state;
- probabilistically decides whether to volunteer, based mainly on engagement;
- selects one intact action and submits it as a bid;
- owns any acceptance, preference switch, and final vote.

Ordinary actions are `REACT`, `SUPPORT`, `OBJECT`, `COMPARE`, `ASK`, and `ACCEPT`. Protocol actions are `OPENING`, `ANSWER`, and `VOTE`.

The `FloorManager` selects one intact bid using categorical priority and light turn balancing. It never rewrites the selected action. Required direct answers and votes bypass voluntary willingness.

The LLM receives the selected action, grounded source, persona voice, active thread, and recent turns. It realizes discussion wording only; it does not choose the speaker, action, movement, or vote. Formal vote wording is deterministic.

## Discussion and threads

The ordinary floor collects one bid from every willing simulator. Questions are eligible only when they address another visible position or trade-off; they are not generated simply because an unused attribute exists.

At most one `DiscussionThread` is active:

- a direct question requires the named participant to answer next;
- a group question allows one eligible participant to answer;
- a concern allows another participant to respond;
- later related reactions, objections, comparisons, or acceptances remain voluntary;
- the thread closes when no related bid exists or after the configured turn cap.

Structured point keys `(option, attribute)` prevent a participant from repeatedly using the same argument and prevent an already public point from opening another question. Later thread turns must add a new point, compare, or visibly move; otherwise the thread closes.

Discussion ends at the hard budget, after the target budget when no novel bid remains, or after minimum participation when public preferences have converged.

## Traits

The simulator parameters are:

- **engagement**: probability of submitting a voluntary bid;
- **verbosity**: utterance word budget;
- **directness**: lexical instruction for hedging or firmness;
- **stubbornness**: probability of accepting or switching after a valid public trigger.

A rare hard blocker rejects every nonpreferred option and never moves.

## Setup

Automatic setup has three small LLM stages:

1. generate shared context and four option cards;
2. generate one or two natural aliases per option and one fixed first name per participant;
3. generate persona cards and private option stances using those fixed names.

Scenario validation is intentionally structural. It checks exact option IDs, unique full names, required public attributes, upsides, concerns, and compact shared context. It does not infer generic world semantics, superlatives, or missing attributes.

A scenario receives up to three complete generation attempts, with validation feedback supplied after each failure. If all three fail, setup is recorded as an error. The lightweight metadata call never regenerates an already valid scenario: malformed aliases are discarded, and invalid or missing participant names receive unique local fallbacks. Fixed names are propagated through backgrounds, private goals, rejection text, and stance reasons.

An option may be referenced by:

- its full name;
- a validated generated alias, such as `Chicago City` for `Chicago City Stay`;
- `Option A`, `Option B`, and so on.

Aliases must be formed from words in the full name and remain unique after normalization. Generated aliases contain two or more words, contain no numbers, and cannot end in incomplete connectors such as `to` or `with`. Manual scenarios and manual persona profiles remain supported through `config.yaml`.

## Visible evidence and language realization

Openings, visible preference movements, and votes must explicitly identify their option through a validated reference. Missing an exact alias is treated as a minor realization issue for ordinary discussion turns, including questions, reactions, objections, answers, and comparisons; those turns are not dropped solely for that reason. This allows natural contextual wording such as “That distance would bother me too” without making broad fuzzy matching part of the runtime.

The realization prompt encourages speakers to:

- connect to the preceding message;
- vary sentence openings;
- avoid copying their recent wording or structure;
- place the option reference naturally rather than always first;
- use short acknowledgments, pronouns, contractions, and `we`/`us` when appropriate.

The compact prompt encourages contextual continuation and asks for an option reference when clarity needs one, without forcing every ordinary turn to repeat a full name. It also discourages routinely starting with an option name, participant name, or `I`.

Supporting actions use the persona's grounded `reason_for` or the option upside. Objections and concern questions use `reason_against` or the option concern. Neutral attributes are reserved for reactions and comparisons, preventing arbitrary facts such as capacity or duration from being reframed as unsupported objections. Every comparison uses the same named public attribute from both options. The two values are labeled by option, while wording remains conversational and free-form. Comparisons have a low policy weight and cannot be selected in close succession.

## Validation and fallbacks

Runtime validation rejects only hard correctness failures, including:

- empty or malformed output;
- invalid structured option targets;
- unsupported numeric values;
- an absent direct addressee;
- an unclear or mismatched formal vote representation;
- a movement that does not visibly identify its target option;
- a hard-blocker contradiction.

Only openings receive one repair attempt and a deterministic last-resort fallback. Required answers keep the generated wording and are not semantically rescored, repaired, or replaced by generic fallbacks; they are subject only to the same hard surface checks as other turns. Invalid voluntary contributions are dropped and flagged rather than inserted into public history.

Formal vote text is deterministic from the start, using short variants such as `My final vote is X`. Voting therefore uses no dialogue-generation or repair calls.

No LLM validator or judge is called during dialogue generation.

## Narrowing and outcomes

Public preferences determine narrowing before voting:

- unanimity or a decisive majority proceeds directly to voting;
- only narrow 2–1 and 3–2 majorities receive one bounded outlier opportunity;
- a split without a majority receives one short compromise prompt followed by up to two autonomous movement-bid rounds.

During ordinary discussion, a non-hard-blocking simulator may also make another visible option acceptable. It may switch preference before narrowing when that option has been discussed recently, has more public support than its current choice, and the participant's stubbornness draw permits movement. This keeps the change grounded in the public exchange rather than forcing convergence.

Every participant then casts one explicit final vote. Outcomes are deterministic:

- `successful`: unanimous vote;
- `majority`: one option has a strict configured majority;
- `unresolved`: no option reaches a majority.

## Running

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Configure providers and runtime values in `config.yaml`, then run:

```powershell
py .\main.py
```

Run deterministic development tests:

```powershell
py -m pytest -q
```

Run a small scenario batch:

```powershell
py .\eval\run_scenarios.py --limit 10 --seed 500 --clean
py .\eval\summarize_runs.py --logs .\eval\logs_scenarios
```

Run the independent post-hoc transcript judges separately:

```powershell
py .\eval\judge_transcripts.py --logs .\eval\logs_scenarios --judges 3 --provider uni
```

## Repository layout

```text
src/                    runtime, setup, policy, validation, logging
config.yaml             providers and compact runtime configuration
main.py                 interactive entry point
eval/                   focused suite, scenario batch, summaries, judges
tests/                  deterministic development tests
info/                   implementation documentation
info/papers/            supplied literature notes and papers
SIMPLIFICATION_ACTION_PLAN.md
```

## Logged artifacts

Each run stores a transcript and `run.json`. The structured artifact includes:

- scenario, aliases, and complete persona cards;
- provenance and actual seed;
- visible turns and optional action traces;
- generation attempts and validation failures;
- public point counts, recent point keys, preferences, votes, and outcome;
- repairs, drops, fallbacks, review flags, and token usage.

The report-facing evaluator intentionally exposes only a compact set of reliability, process, cost, and trait metrics. Detailed runtime evidence remains in `run.json` for debugging.
