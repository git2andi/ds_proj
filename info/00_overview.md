# System overview

The project simulates small option-grounded group discussions. It is intentionally narrower than a general multi-agent dialogue system.

## Data flow

```text
topic
→ public scenario and four option cards
→ validated natural aliases
→ persona cards and private stances
→ autonomous simulator bids
→ floor selection
→ LLM realization
→ hard validation
→ public state update
→ pre-vote narrowing, one final vote, outcome
```

The core contribution is the ownership split. Each simulator decides whether to speak and selects a complete action. The floor only schedules intact bids. The LLM supplies opening and discussion language, while deterministic code controls phases, visible movement commitment, formal vote wording, and outcomes.

## Runtime state

The shared state contains visible turns, public preferences and acceptances, one optional bounded thread, compact public point history, response obligations, final votes, and runtime counters. It does not maintain a general issue graph or infer hidden dialogue semantics from arbitrary text.

## Phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

Discussion supports direct questions, group questions, concerns, third-party reactions, comparisons, and acceptance. Any compromise occurs before one final formal vote.

## Intended claims

The implementation supports claims about simulator authority, configurable behavioral parameters, bounded discussion control, visible preference movement, and deterministic voting. It does not establish human realism, psychological validity, or complete semantic grounding.
