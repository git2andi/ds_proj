# Evaluation summary

Attempted scenarios: 198
Completed runs: 196
Setup/runtime failures: 2
Outcomes: {'majority': 118, 'successful': 19, 'unresolved': 59}
Protocol pass: 196/196
Runs needing review: 36

## Runtime quality

- Mean participant turns: 30.8
- Mean moderator ratio: 0.125
- Repairs per 100 participant turns: 0.63
- Drops per 100 participant turns: 0.74
- Fallbacks per 100 participant turns: 0.28
- Total input tokens: 3312079
- Total output tokens: 477473

## Trait realization (Spearman)

- engagement_vs_voluntary_share: 0.675
- verbosity_vs_avg_words: 0.916
- stubbornness_vs_flexibility: -0.205
- directness_vs_inverse_hedge_rate: 0.582

## Failed scenarios

- Pick a neighborhood for a short stay in Barcelona: RuntimeError: mandatory opening failed for p1: ['focused option is not visible']
- Pick a co-op video game for a weekend playthrough: RuntimeError: invalid deterministic vote for p1: ['unsupported numeric claim: 2 is']
