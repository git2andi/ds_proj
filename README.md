# Autonomous Multi-User Decision Simulator

University project: **Implementation of User Simulators in Multi-User Conversational AI**.

The program turns a topic into a fixed public option board and simulates several configurable users discussing it. Each simulator owns a structured decision about whether to speak, what to do, which option or person to address, which grounded reason to use, whether to change stance, and how to vote. A lightweight environment manages only the shared conversation protocol.

```text
topic
→ scenario and options
→ personas and private option stances
→ simulator bids
→ floor selection
→ LLM utterance realization
→ deterministic validation/grounding
→ public state update
→ narrowing and voting
→ outcome and logs
```

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

A single return from voting to narrowing is allowed when the first vote has no majority and the intervening discussion produces visible acceptance or switching. Otherwise the run closes unresolved rather than repeating the same vote.

## Ownership

### User simulator

Each `UserSimulator` owns:

- voluntary floor entry;
- the complete structured `UserAction`;
- action type, option focus, addressee, and grounded reason source;
- support, concern, question, answer, comparison, acknowledgement, compromise, and final-position behavior;
- proposed acceptance or preference switching;
- the formal vote.

Engagement controls voluntary bid probability. Verbosity controls the realization word budget. Directness affects wording. Stubbornness controls the probability of accepting or switching after a concrete public trigger. Hard blockers use stubbornness 5 and never accept or vote for a nonpreferred option.

### Environment and floor

The environment owns:

- opening order;
- direct-answer obligations;
- categorical floor arbitration;
- broad participant-scaled turn budgets;
- one active question or concern;
- one soft option-coverage opportunity;
- phase transitions and public narrowing;
- vote collection and outcome computation.

The floor selects one complete bid and never rewrites it. It does not use expected-turn shares or equalize participation.

### LLM

Only the selected action is sent to the dialogue LLM. The LLM controls wording and sentence structure, not speaker selection, action choice, stance movement, or votes. Persona backstory and private goals are included selectively when they help realize the simulator-owned reason.

## Setup

Automatic setup uses two bounded LLM calls:

1. create one shared context and four public option cards;
2. create personas with direct traits, private goals, and one stance per option.

The builder validates option IDs, attributes, aliases, persona fields, trait ranges, preference shapes, hard blockers, and stance consistency. Setup sampling uses the run-local random generator, so the run seed controls Python-side setup and dialogue decisions. Invalid setup is retried only up to the configured attempt limit; it is not hidden by a fabricated scenario.

Manual scenarios and manual persona profiles remain supported through `config.yaml`.

## Discussion and issues

Every eligible simulator produces either silence or one complete bid. Bids use three categorical priorities:

1. required direct answer;
2. response to the active issue or moderator stimulus;
3. ordinary voluntary contribution.

Ties inside the highest available category are resolved with seeded random selection. The configured consecutive-turn limit prevents pathological monologues without enforcing equal participation.

The runtime keeps at most one active question or concern:

- a direct question creates one mandatory answer;
- one optional third-party reaction may follow;
- concern responses and owner reactions remain voluntary;
- unresolved issues become stale when the group moves on;
- the same semantic concern is deduplicated and may be reopened only within its configured bound.

Soft option coverage is also voluntary. The moderator may expose an untouched option once; if no simulator bids, the option is recorded as receiving no expressed interest rather than receiving manufactured dialogue.

## Narrowing and voting

Narrowing uses only public preferences, acceptances, unresolved concerns, and visible votes. It never reads another simulator's hidden stance to manufacture consensus.

- Public unanimity may skip unnecessary restatement.
- With one leader, only relevant dissenters or unresolved concern owners receive bounded final-position opportunities.
- A complete split exposes one simulator-owned compromise opportunity.
- A valid majority closes immediately; unanimity is not required.
- A second vote occurs only after visible movement during the one re-narrowing round.

## Realization, parsing, grounding, and repair

The structured action remains authoritative, but protocol-changing effects are committed only after the visible text satisfies the required check.

For each selected action:

1. validate the structured action and its option/reason references;
2. generate one utterance;
3. deterministically inspect option mentions, votes, direct-answer relevance, required movement wording, duplicate wording, and high-risk factual claims;
4. make one focused repair attempt when a hard check fails;
5. accept and update public state, or drop the ordinary contribution.

There is no validator LLM.

Grounding is deliberately narrow and explainable. It rejects clear problems such as unknown options, values copied from the wrong option, invented prices/times/distances/capacities, contradictions of the public card, and unsupported objective guarantees. Subjective opinions and personal context remain allowed. Validation does not require every ordinary utterance to reproduce its intended action label exactly.

Voluntary stance-changing contributions that still fail after repair are dropped and logged instead of being replaced by scripted language. Protocol-critical votes, and mandatory movement statements when required by the phase, may use a concise grounded fallback so the run can close; every fallback is explicitly recorded. Token accounting includes both the original and repair call.

## Running

Install the dependencies for the configured provider and place required API keys in `.env`.

```powershell
py .\main.py "Choose a Saturday study location"
```

With no CLI topic, manual environment mode uses the configured board; otherwise the program asks for a topic.

## Deterministic tests

```powershell
py -m pytest -q
```

The offline suite covers setup validation, seeded sampling, simulator authority, floor behavior, issue handling, grounding, repair accounting, hard blockers, stance movement, voting, logging, and evaluation-script definitions.

## Evaluation

The active evaluation implementation is under `eval2/`. All default paths are relative to that folder, and configuration overrides are applied in memory only.

```powershell
py .\eval2\run_eval_suite.py
py .\eval2\run_scenarios.py --limit 40
py .\eval2\evaluate_runs.py
py .\eval2\judge_transcripts.py
py .\eval2\validate_judge.py
py .\eval2\run_config_sweep.py
py .\eval2\run_config_confirmation.py
```

Purpose of the scripts:

- `run_eval_suite.py`: focused pinned protocol/regression cases;
- `run_scenarios.py`: broader topic batch generation;
- `evaluate_runs.py`: deterministic post-hoc metrics over `run.json` files;
- `judge_transcripts.py`: three-role LLM transcript evaluation on naturalness, coherence, groundedness, persona consistency, and deliberation quality;
- `validate_judge.py`: controlled corruption checks for the judge;
- `run_config_sweep.py`: small sweep over duplicate detection, issue follow-up depth, consecutive turns, and small-group closure;
- `run_config_confirmation.py`: cumulative confirmation on several topics.

Configuration comparisons reuse one generated scenario/persona setup for every candidate within a seed. This prevents stochastic setup differences from being mistaken for a configuration effect and avoids repeated setup calls.

The old `eval/` folder contains historical outputs, including the intentionally interrupted scenario batch. It is not the active script location.

See [`eval2/README.md`](eval2/README.md) and [`info/07_evaluation_and_logging.md`](info/07_evaluation_and_logging.md).

## Main files

```text
main.py                 command-line entry point
config.yaml             providers, setup, traits, policy, language, and output settings
src/builders.py         scenario/persona construction and validation
src/simulator.py        simulator policy and floor arbitration
src/dialogue.py         phase loop, issue protocol, realization, and state updates
src/prompts.py          setup, realization, and focused repair prompts
src/validation.py       deterministic action, grounding, and realization checks
src/consensus.py        narrowing and final outcome derivation
src/logger.py           transcript, run.json, prompts, and metrics
src/eval.py             runtime metric flattening
eval2/                  active evaluation scripts
eval/                   preserved historical evaluation outputs
tests/                  deterministic offline tests
```
