# Configuration and running

Important configuration sections:

```yaml
simulator:
  bid_probability_by_engagement: {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.70, 5: 0.90}
  question_modes: [choice_impact, tradeoff]
  unknown_information_question_probability: 0.08
  movement_probability_by_stubbornness: {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20, 5: 0.00}

language:
  max_words_by_verbosity: {1: 8, 2: 12, 3: 16, 4: 22, 5: 27}
  directness_instructions: {...}

conversation:
  min_voluntary_turns_per_participant: 2
  soft_target_voluntary_turns_per_participant: 5
  hard_max_voluntary_turns_per_participant: 7
  soft_target_voluntary_turn_cap: 22
  hard_max_voluntary_turn_cap: 30
  issue_follow_up_cap: 3
  direct_question_optional_follow_up_cap: 1
  concern_external_response_cap: 2
  max_concerns_per_participant: 1
  stagnation_no_bid_rounds: 1
  compromise_window_max_turns: 1
  narrowing_reaction_turn_cap: 2
  large_group_narrowing_final_position_cap: 3
  recent_turns_in_prompt: 7
  max_consecutive_turns: 2
```

Behavioral probabilities, word budgets, and primary turn caps are explicit configuration values. Two fixed protocol rules remain intentionally simple in code: small groups receive one extra ordinary no-bid retry and shared acceptability waits for three turns beyond the minimum before closing discussion.

Run:

```powershell
py .\main.py "Your topic"
```

Tests:

```powershell
py -m pytest -q
```

Evaluation:

```powershell
py .\eval\run_eval_suite.py
```

The LLM-backed suite contains 17 cases over 10 varied domains and covers every supported group size from 2 through 7 participants. Fifteen use normal defaults; two `long_*` cases apply isolated stress overrides.


`conversation.diagnostic_allow_reason_reuse` defaults to `false`. It exists only for deliberately long evaluation cases that need to expose repetition under extended pacing. It should remain disabled for normal simulations.
