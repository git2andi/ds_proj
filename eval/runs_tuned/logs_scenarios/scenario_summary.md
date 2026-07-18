# Scenario batch summary

Runs: 198 (2 errors)
Outcomes: {'error': 2, 'majority': 118, 'successful': 19, 'unresolved': 59}

| Participants | Runs | Successful | Majority | Unresolved | Avg participant turns | Avg input tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 11 | 0 | 21 | 14.6 | 8820 |
| 3 | 32 | 5 | 22 | 5 | 21.8 | 12483 |
| 4 | 33 | 1 | 17 | 15 | 29.8 | 16477 |
| 5 | 33 | 2 | 25 | 6 | 35.8 | 19526 |
| 6 | 33 | 0 | 24 | 9 | 39.7 | 21305 |
| 7 | 33 | 0 | 30 | 3 | 42.5 | 22400 |

## Errors

- case 122 ('Pick a neighborhood for a short stay in Barcelona'): RuntimeError: mandatory opening failed for p1: ['focused option is not visible']
- case 145 ('Pick a co-op video game for a weekend playthrough'): RuntimeError: invalid deterministic vote for p1: ['unsupported numeric claim: 2 is']
