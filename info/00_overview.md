# System overview

This project simulates bounded, option-grounded group discussions between two and seven user simulators. It is intentionally narrower than a general multi-agent dialogue system.

## Data flow

```text
topic and YAML configuration
→ shared context and four public option cards
→ validated option aliases and participant names
→ persona cards and private stances
→ simulator-owned structured bids
→ floor selection
→ LLM utterance realization
→ deterministic validation
→ public state update
→ optional narrowing, one final vote, outcome
```

The central design choice is the separation of responsibilities. Each `UserSimulator` constructs a complete `UserAction` from its private state and the visible dialogue. The floor manager selects among intact bids but does not rewrite them. The LLM supplies wording for openings and discussion turns, while deterministic code controls phases, state commitment, final vote wording, and outcome calculation.

## Runtime state

Private participant state contains the persona, traits, option ranks, reasons, current preference, and acceptable or rejected alternatives. Shared state contains only public evidence: visible turns, stated preferences and acceptances, one optional bounded thread, response obligations, point-use records, narrowing state, votes, and runtime counters.

## Phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

Discussion supports reactions, support, objections, comparisons, direct and group questions, answers, and visible acceptance or switching. Narrowing occurs only before the single final voting round.

## Scope

The implementation supports claims about simulator authority, configurable behavioral controls, bounded turn taking, visible stance movement, and deterministic outcomes. It does not claim human realism, psychological validity, unrestricted conversation, or complete semantic grounding.
