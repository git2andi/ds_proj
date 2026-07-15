# Focused LLM-backed evaluation suite

| Case | Outcome | Turns | Narrow | Move | Comp. | Concerns R/S | Re-vote | Repairs | Tokens | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two_person_weekend_consensus | successful | 5 | 0 | 0 | 0/0 | 0/0 | 1 | 0 | 1784 | yes |
| three_way_split_workspace | unresolved | 13 | 0 | 0 | 0/0 | 0/1 | 1 | 0 | 4953 | yes |
| compromise_presentation | successful | 14 | 2 | 1 | 1/1 | 0/0 | 1 | 2 | 6050 | yes |
| majority_no_moderator_dinner | majority | 17 | 3 | 0 | 0/0 | 0/1 | 1 | 2 | 6958 | yes |
| hard_blocker_cleaning | majority | 12 | 1 | 0 | 0/0 | 0/0 | 1 | 0 | 4636 | yes |
| direct_question_hike | majority | 19 | 2 | 1 | 1/1 | 0/0 | 2 | 2 | 7958 | yes |
| concern_resolution_book_club | successful | 15 | 2 | 1 | 2/1 | 1/0 | 1 | 1 | 6170 | yes |
| grounding_sensitive_flight_n4 | successful | 21 | 4 | 2 | 2/2 | 0/0 | 1 | 2 | 8727 | yes |
| engagement_spread_meeting_n5 | successful | 29 | 7 | 3 | 3/3 | 1/0 | 1 | 2 | 12258 | yes |
| language_style_spread_laptop | successful | 30 | 6 | 3 | 3/3 | 0/0 | 2 | 3 | 12988 | yes |
| six_person_weekend_participation | unresolved | 34 | 4 | 2 | 2/2 | 0/0 | 2 | 2 | 14081 | yes |
| seven_person_workspace_scale | majority | 35 | 9 | 4 | 4/3 | 0/1 | 1 | 5 | 15102 | yes |
| two_person_presentation_deadlock | unresolved | 7 | 0 | 0 | 0/0 | 0/1 | 1 | 1 | 2814 | yes |
| five_person_restaurant_compromise | successful | 29 | 7 | 4 | 4/3 | 1/0 | 1 | 3 | 12468 | yes |
| six_person_cleaning_mixed | majority | 27 | 5 | 2 | 2/2 | 0/0 | 1 | 1 | 11238 | yes |

Structural passes: 15/15
Quality-expectation passes: 15/15
Case passes: 15/15
Total input tokens: 128185
Mean input tokens per case: 8545.7

A second vote is permitted only when the preceding re-narrowing produced visible acceptance or switching.
Comp. reports compromise proposals/acceptances; every selected movement must commit and carry a stored grounded reason; repair-rate quality threshold is 25%.

## Policy calibration

```json
{
  "bid_probability_by_engagement": {
    "1": 0.2,
    "2": 0.35,
    "3": 0.5,
    "4": 0.7,
    "5": 0.9
  },
  "movement_probability_by_stubbornness": {
    "1": 0.8,
    "2": 0.6,
    "3": 0.4,
    "4": 0.2,
    "5": 0.0
  },
  "conversation_budgets_n3": [
    6,
    12,
    18
  ],
  "conversation_budgets_n4": [
    8,
    16,
    24
  ],
  "conversation_budgets_n7": [
    14,
    22,
    30
  ]
}
```
