# Configuration and running

Important configuration sections:

```yaml
simulator:
  bid_probability_by_engagement: {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.70, 5: 0.90}
  movement_probability_by_stubbornness: {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20, 5: 0.00}

language:
  max_words_by_verbosity: {1: 10, 2: 14, 3: 18, 4: 24, 5: 30}
  directness_instructions: {...}

conversation:
  min_voluntary_turns_per_participant: 2
  soft_target_voluntary_turns_per_participant: 4
  hard_max_voluntary_turns_per_participant: 6
  soft_target_voluntary_turn_cap: 22
  hard_max_voluntary_turn_cap: 30
  issue_follow_up_cap: 3
  max_concerns_per_participant: 1
  stagnation_no_bid_rounds: 1
  compromise_window_max_turns: 1
  narrowing_reaction_turn_cap: 2
  recent_turns_in_prompt: 5
  max_consecutive_turns: 2
```

All behavioral probabilities and limits are explicit configuration values. The code does not hide additional multiplier formulas.

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

The LLM-backed suite contains 15 cases over 10 topics and covers every supported group size from 2 through 7 participants.
