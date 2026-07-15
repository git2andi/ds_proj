# Autonomous Multi-User Decision Simulator

University project: **Implementation of User Simulators in Multi-User Conversational AI**.

The program simulates several configurable users discussing a fixed public option board. Each simulator independently decides whether to speak, what communicative action to perform, which option or participant to address, and whether to change stance. A lightweight environment coordinates the floor and guides the group through:

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

A single bounded return from voting to narrowing is allowed only when the first vote has no majority and the final discussion produces visible acceptance or switching. Otherwise the run closes unresolved without a pointless duplicate vote.

## Core ownership

### User simulator

Each `UserSimulator` owns:

- voluntary participation;
- the complete structured `UserAction`;
- support, concern, concern-based question, answer, soft comparison, final-position, and compromise choices;
- option focus and addressee;
- personal reason;
- visible stance movement;
- clear visible vote.

### Environment

The environment owns:

- mandatory opening order;
- direct-answer obligations;
- categorical floor arbitration;
- broad turn budgets;
- one active issue;
- soft option coverage;
- public narrowing;
- vote collection and outcome computation.

### LLM

The LLM only realizes one selected action as natural language. It does not choose the speaker, action, stance change, or vote.

Ordinary pro/con facts already established publicly are not offered again as new standalone contributions; required answers, issue reactions, and stance movement remain available.

## Simplified policy

There are no urgency scores, floor multipliers, candidate-score formulas, or public-pressure scores.

- Engagement maps to a configurable probability of submitting a voluntary bid.
- Stubbornness maps to a configurable movement probability after a concrete trigger. Movement may mean making an option acceptable or visibly switching preference.
- Every acceptance or switch stores one concrete movement reason from the persona stance or resolved issue. The acceptance must make that reason visible; a later vote may stay short because the rationale is already public.
- A rank-3 neutral option may become compromise material directly. A rank-2 disliked option becomes eligible only after that participant's concrete concern was visibly resolved or softened. Rank-1 and hard-blocked options never become acceptable.
- Verbosity maps to a configurable maximum word count.
- Directness maps to one short wording instruction.
- The floor uses categorical priority: required answer, concern-owner reaction, active-issue response, ordinary contribution.
- A direct question uses a configured semantic form (choice impact, trade-off, or optional condition), closes after the addressed participant answers, and is not answered repeatedly by the whole group.
- After a concern receives a response, its owner visibly accepts the trade-off, softens, or maintains the concern.
- Ties inside one category are resolved with seeded random selection.
- Ordinary discussion limits each simulator to one newly opened concern by default, avoiding systematic processing of every alternative.
- Stagnation exposes one simulator-owned compromise opportunity; it never forces a switch. A visible moderator compromise prompt is committed only together with a successfully realized participant response, so failed language generation cannot leave an unanswered nudge.
- Once a movement action wins the floor, language failure cannot erase it: after one focused repair the runtime commits a grounded minimal movement fallback and records that fallback explicitly.

## Reasons and grounding

Ordinary actions primarily use:

1. persona-specific `reason_for` and `reason_against`;
2. option `upside` and `concern` as fallback;
3. raw option attributes only when a direct question, active issue, persona reason, or concrete comparison needs them.

The structured action is authoritative. Validation blocks only hard failures such as unusable output, unknown options, unsupported concrete values, unrelated direct answers, genuinely ambiguous votes, invisible required stance changes, hard-blocker contradictions, and near-verbatim self-repetition. During voting, a short message that visibly names exactly the intended option is sufficient even without a fixed vote verb. An incomplete comparison is accepted as a useful one-sided contribution and is not recorded as public comparison evidence.

No validator LLM is used. Realization prompts explicitly forbid invented option subtypes, facilities, use cases, guarantees, and stronger or weaker mutations of supplied facts. Failed ordinary realizations remain diagnosable in `run.json`. A selected stance movement or formal vote can never disappear: after one generation and one focused repair fail, the runtime renders a minimal grounded statement for the simulator’s already-authoritative action.

## Running

Install dependencies for the configured LLM provider, then run:

```powershell
py .\main.py "Choose a Saturday study location"
```

With no CLI topic, the program uses the manual scenario when `environment.mode: manual`; otherwise it asks interactively for a topic.

## Deterministic tests

```powershell
py -m pytest -q
```

The deterministic suite covers configuration, simulator authority, opening variation, concern-based questions, issue closure, rank-2 concern gating, adaptive narrowing, optional compromise, authoritative movement and vote fallback, deterministic tied choices, hard blockers, logging, and bounded pacing from two through seven participants.

## LLM-backed evaluation

List cases without contacting the endpoint:

```powershell
py .\eval\run_eval_suite.py --list
```

Run all 15 cases across 10 topics and groups of 2–7 participants:

```powershell
py .\eval\run_eval_suite.py
```

Run selected cases:

```powershell
py .\eval\run_eval_suite.py --case grounding_sensitive_flight_n4 --case engagement_spread_meeting_n5
```

Outputs are written to `eval/logs_eval_suite/` and zipped as `eval/logs_eval_suite.zip`.

## Logs

Each normal run writes:

- `transcript.md`: option board, participant trait summary, visible chat, outcome, compact metrics;
- `run.json`: structured run state, core diagnostics, and compact failed-generation records;
- `metrics.csv`: one flat comparison row per run.

Deep generation attempts and action traces are optional:

```yaml
output:
  debug_metrics: false
  write_action_trace: false
```

## Main files

```text
src/builders.py       scenario and persona setup
src/simulator.py      participant-local policy and floor selection
src/dialogue.py       phase loop and public state updates
src/prompts.py        compact action-to-language prompts
src/validation.py     minimal deterministic validation/grounding
src/consensus.py      public narrowing and vote outcomes
src/logger.py         compact transcript and structured logging
```

See `info/` for the detailed architecture.
