# Focused LLM-backed evaluation suite

| Case | Outcome | Turns | Narrow | Move | Comp. | Concerns R/S | Re-vote | Repairs | Tokens | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two_person_weekend_consensus | successful | 6 | 0 | 0 | 0/0 | 0/0 | 1 | 1 | 2180 | yes |
| three_way_split_workspace | unresolved | 18 | 0 | 0 | 0/0 | 0/3 | 1 | 1 | 6762 | yes |
| compromise_presentation | successful | 14 | 2 | 1 | 1/1 | 0/0 | 1 | 1 | 5398 | yes |
| majority_no_moderator_dinner | successful | 18 | 2 | 1 | 1/1 | 0/0 | 1 | 0 | 6567 | yes |
| hard_blocker_cleaning | majority | 12 | 1 | 0 | 0/0 | 0/0 | 1 | 0 | 4309 | yes |
| direct_question_hike | majority | 18 | 1 | 1 | 1/1 | 0/0 | 2 | 1 | 7756 | yes |
| concern_resolution_book_club | successful | 15 | 2 | 1 | 2/1 | 1/0 | 1 | 1 | 5620 | yes |
| grounding_sensitive_flight_n4 | majority | 22 | 5 | 1 | 1/1 | 0/1 | 1 | 1 | 8567 | yes |
| engagement_spread_meeting_n5 | successful | 29 | 7 | 3 | 3/3 | 1/0 | 1 | 0 | 11456 | yes |
| language_style_spread_laptop | unresolved | 20 | 0 | 0 | 0/0 | 0/0 | 1 | 0 | 8312 | yes |
| six_person_weekend_participation | unresolved | 24 | 0 | 0 | 0/0 | 0/0 | 1 | 0 | 10436 | yes |
| seven_person_workspace_scale | majority | 35 | 7 | 3 | 3/3 | 1/0 | 1 | 2 | 14063 | yes |
| two_person_presentation_deadlock | unresolved | 11 | 0 | 0 | 0/0 | 0/1 | 1 | 1 | 5393 | yes |
| five_person_restaurant_compromise | successful | 29 | 6 | 3 | 3/3 | 0/1 | 1 | 0 | 11085 | yes |
| six_person_cleaning_mixed | majority | 34 | 6 | 1 | 1/1 | 0/3 | 1 | 0 | 13094 | yes |

Structural passes: 15/15
Quality-expectation passes: 15/15
Case passes: 15/15
Total input tokens: 120998
Mean input tokens per case: 8066.5

A second vote is permitted only when the preceding re-narrowing produced visible acceptance or switching.
Comp. reports compromise proposals/acceptances; repair-rate quality threshold is 25%.

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
