# LLM-backed evaluation suite

Dialogue provider/model: gpt / gpt-4.1-mini
The LLM realizes authoritative actions; seeded Python policies choose and commit them.

| Case | Scenario | Outcome | Round | Turns | Voluntary | Repairs | Drops | Resolved issues | Switches | Vote protocol | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy_agreement | study | successful | 1 | 9 | 3 | 0 | 0 | 0 | 0 | valid | yes |
| three_way_split | study | successful | 1 | 30 | 21 | 0 | 1 | 3 | 2 | valid | yes |
| normal_compromise | study | successful | 1 | 30 | 21 | 2 | 0 | 3 | 1 | valid | yes |
| majority_holdout | study | majority | 1 | 30 | 21 | 1 | 0 | 1 | 0 | valid | yes |
| hard_blocker | cleaning | majority | 1 | 28 | 19 | 1 | 0 | 3 | 0 | valid | yes |
| direct_question_followup | study | successful | 1 | 30 | 20 | 1 | 0 | 4 | 1 | valid | yes |
| unresolved_concern | study | unresolved | 2 | 31 | 21 | 1 | 0 | 1 | 0 | valid | yes |
| concern_resolution | study | successful | 1 | 26 | 20 | 4 | 0 | 1 | 1 | valid | yes |
| no_moderator | restaurant | successful | 1 | 32 | 21 | 0 | 0 | 5 | 1 | valid | yes |
| grounding_sensitive | flight | successful | 1 | 31 | 21 | 3 | 1 | 4 | 2 | valid | yes |
| engagement_spread | study | majority | 1 | 34 | 21 | 2 | 0 | 5 | 2 | valid | yes |
| verbosity_spread | study | majority | 1 | 28 | 19 | 2 | 1 | 4 | 0 | DEGRADED | NO |
| visible_stance_switch | study | successful | 1 | 27 | 19 | 6 | 0 | 2 | 1 | valid | yes |
| persona_distinctness | study | unresolved | 2 | 38 | 23 | 1 | 0 | 3 | 1 | valid | yes |
| no_majority_revote | study | unresolved | 2 | 35 | 20 | 0 | 0 | 3 | 0 | valid | yes |

Structural passes: 14/15
Case-specific passes: 14/15
Total repairs: 24
Total dropped turns: 3
Vote-protocol degradations: 1
Rapid-switch violations: 0
Direct-answer phase-boundary violations: 0

## Policy and isolated realization calibration

```json
{
  "engagement_bid_counts": {
    "1": 254,
    "3": 654,
    "5": 1195
  },
  "engagement_monotonic": true,
  "switch_probabilities": {
    "1": 0.7360000000000001,
    "2": 0.6000000000000001,
    "3": 0.41600000000000004,
    "4": 0.22400000000000003
  },
  "stubbornness_monotonic": true,
  "distinct_question_keys": [
    "rationale|B|p2",
    "acceptability|B|p3",
    "rationale|C|p3",
    "impact|C|p3",
    "acceptability|C|p2",
    "acceptability|D|p2",
    "comparison|A,B|p2"
  ],
  "question_key_diversity": 7,
  "available_reason_sources": 5,
  "realization_diagnostics": {
    "verbosity_1": {
      "text": "I support Option A since it\u2019s quiet and predictable, which helps with focused work.",
      "word_count": 14
    },
    "verbosity_5": {
      "text": "I\u2019m leaning toward Option A, the Central Library, since it offers a quiet and predictable environment, which is really important for staying focused on project work, even though it might get a bit crowded at times.",
      "word_count": 36
    },
    "directness_1": {
      "text": "I\u2019d lean toward Option A since it\u2019s quiet and predictable, which should really help us stay focused on the project.",
      "word_count": 20
    },
    "directness_5": {
      "text": "I support Option A because the Central Library is quiet and predictable, which helps us focus on the project.",
      "word_count": 19
    },
    "style_young": {
      "text": "I\u2019d lean toward Option A since the Central Library is quiet and predictable, which helps with focused project work.",
      "word_count": 19
    },
    "style_measured": {
      "text": "I would support Option A, the Central Library, as it offers a quiet and predictable environment suited for focused project work.",
      "word_count": 21
    },
    "verbosity_monotonic": true,
    "directness_qualified_terms": 0,
    "directness_explicit_terms": 1,
    "formal_vote_switch": {
      "text": "I\u2019m switching my vote from Option B to Option A because the discussion shifted things, and the quietness at the Central Library seems better for our focused work.",
      "valid": true,
      "errors": []
    }
  }
}
```
