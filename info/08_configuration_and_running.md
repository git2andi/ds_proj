# Configuration and running

All behavioral probabilities, word budgets, and pacing caps live in `config.yaml` and are validated on load (`src/config_loader.py`): budget ordering, probability bounds, monotone level mappings, group-size boundaries, and preference-shape weights are all checked before a run starts.

Important configuration sections:

```yaml
simulator:
  bid_probability_by_engagement: {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.70, 5: 0.90}
  question_modes: [choice_impact, tradeoff]
  unknown_information_question_probability: 0.08
  movement_probability_by_stubbornness: {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20, 5: 0.00}

language:
  max_words_by_verbosity: {1: 8, 2: 12, 3: 16, 4: 22, 5: 27}
  action_max_words: {acknowledge: 12, ask: 18, answer: 18, final_position: 18, vote: 10, simple_vote: 8}
  near_duplicate_similarity_threshold: 0.92
  near_duplicate_recent_turns: 3
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
  max_concern_reopens: 1
  stagnation_no_bid_rounds: 1
  compromise_window_max_turns: 1
  narrowing_reaction_turn_cap: 2
  small_group_max_participants: 4
  small_group_extra_no_bid_rounds: 1
  small_group_shared_acceptance_extra_turns: 3
  unanimous_closure_min_voluntary_turns_per_participant: 1.0
  large_group_min_participants: 5
  large_group_optional_reaction_window_cap: 2
  large_group_narrowing_issue_turn_cap: 1
  large_group_narrowing_final_position_cap: 3
  recent_turns_in_prompt: 7
  max_consecutive_turns: 2
```

Group-size-scoped pacing is also configuration: groups up to `small_group_max_participants` receive `small_group_extra_no_bid_rounds` extra ordinary bidding rounds and `small_group_shared_acceptance_extra_turns` extra turns before shared acceptability closes discussion; groups of at least `large_group_min_participants` cap optional reaction windows, per-position issue turns, and required final positions during narrowing.

The LLM provider is selected by `llm.dialogue` (`uni | groq | gemini | gpt`) with one model per provider and three sampling profiles (`setup`, `dialogue`, `repair`). API keys come from `.env`. The `top_k` sampling value applies only to providers whose API supports it (the Ollama-based `uni` endpoint); OpenAI-compatible providers use temperature and `top_p`.

`environment.mode: manual` runs a fully user-authored option board; `participants.mode: manual` pins persona profiles (both validated strictly, at most one hard blocker). An explicit CLI topic always forces automatic scenario generation.

## Commands

Run one dialogue:

```powershell
py .\main.py "Your topic"
```

Deterministic tests:

```powershell
py -m pytest -q
```

Evaluation (see `07_evaluation_and_logging.md` for what each does):

```powershell
py .\eval\run_eval_suite.py            # 17 pinned LLM-backed cases
py .\eval\run_scenarios.py --limit 5   # scenarios.txt batch
py .\eval\run_config_sweep.py --list   # one-knob-at-a-time sensitivity sweep
py .\eval\judge_transcripts.py         # ChatEval-style transcript scoring
```

Every evaluation script overrides configuration in memory only; `config.yaml` on disk is never modified by an experiment.

`conversation.diagnostic_allow_reason_reuse` defaults to `false`. It exists only for deliberately long evaluation cases that need to expose repetition under extended pacing. It should remain disabled for normal simulations.
