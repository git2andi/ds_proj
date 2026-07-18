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

The LLM receives the selected action, relevant option facts, persona voice, active thread, and recent turns. It realizes wording only; it does not choose the speaker, action, movement, or vote.

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
2. generate one to three natural aliases for each fixed option name;
3. generate persona cards and private option stances.

Scenario validation is intentionally structural. It checks exact option IDs, unique full names, required public attributes, upsides, concerns, and compact shared context. It does not infer generic world semantics, superlatives, or missing attributes.

If the first scenario is invalid, the validation error is included in one complete regeneration request. If that result is still invalid, setup fails. Alias generation never regenerates an already valid scenario: malformed or colliding aliases are discarded, and the full option name remains valid. Invalid generated persona names receive a unique local fallback instead of invalidating the complete setup.

An option may be referenced by:

- its full name;
- a validated generated alias, such as `Chicago` for `Chicago City Stay`;
- `Option A`, `Option B`, and so on.

Aliases must be formed from words in the full name and remain unique after normalization. Manual scenarios and manual persona profiles remain supported through `config.yaml`.

## Visible evidence and language realization

Openings, new questions, comparisons, movements, and votes must explicitly identify their option through a validated reference. Reactions and answers may use a local contextual reference when the immediately preceding turn or active thread identifies exactly one option. This allows natural wording such as “That distance would bother me too” without allowing arbitrary fuzzy option matching.

The realization prompt encourages speakers to:

- connect to the preceding message;
- vary sentence openings;
- avoid copying their recent wording or structure;
- place the option reference naturally rather than always first;
- use short acknowledgments, pronouns, contractions, and `we`/`us` when appropriate.

## Validation and fallbacks

Runtime validation rejects only hard correctness failures, including:

- empty or malformed output;
- wrong or nonexistent option references;
- unsupported numeric values;
- an absent direct addressee;
- an answer unrelated to the active question or point;
- an unclear or mismatched formal vote;
- movement toward the wrong option;
- a hard-blocker contradiction.

Openings, required answers, and votes receive one repair attempt. A deterministic opening is only a last-resort protocol fallback after both generation and repair fail; natural shorthand such as `Chicago` should normally pass through a generated alias. Required-answer and vote fallbacks use natural text and never expose raw schema keys. Invalid voluntary contributions are dropped and flagged rather than inserted into public history.

No LLM validator or judge is called during dialogue generation.

## Narrowing and outcomes

Public preferences determine narrowing before voting:

- unanimity or a decisive majority proceeds directly to voting;
- only narrow 2–1 and 3–2 majorities receive one bounded outlier opportunity;
- a split without a majority receives one short compromise window.

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
