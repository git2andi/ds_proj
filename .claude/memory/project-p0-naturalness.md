---
name: project-p0-naturalness
description: P0 local conversation and visible OCEAN behavior upgraded 2026-06-30
metadata:
  node_type: memory
  type: project
  originSessionId: bc16d197-d922-4442-885f-d07980d8a535
---

KF08/KF09 local-response and trait-expression upgrade completed 2026-06-30. Semantic repetition remains open under KF14/KF27; greeting/farewell templates remain KF26; commitment churn remains KF29.

**Why:** The router often had a response target, but `sim_utterance` simultaneously showed the full background/private goal and asked for another personal option reason. GPT therefore produced locally unrelated mini-pitches and repeated persona biographies. Generic behavior cues were truncated to the first three traits, so some OCEAN dimensions rarely affected a turn.

**Implementation:**

- `_local_response_turn_for()` prefers a recent turn sharing the focused option, then the latest other-speaker turn. It attaches context without changing the routed speaker or injecting a turn.
- Targeted prompts put the exact prior message first (bounded by `response_target_max_words`), omit background/private goal, and forbid restarting the option case. Repair prompts preserve the same target.
- OPENING retains full background and goal. Untargeted non-opening turns receive only a compact personal stake.
- `_turn_behavior()` derives at most two act-specific cues from OCEAN. No new persona fields were added.
- `_trait_adjusted_act_probabilities()` makes openness, conscientiousness, extraversion, agreeableness, and neuroticism materially affect free-discussion move weights through `routing.trait_act_slope`.
- Extraversion has a stronger but still bounded effect on turn share; SUPPORT weight was shifted toward REACT to reduce parallel option pitching.
- VOTE/ACCEPT guidance and repairs request plain chat choices rather than formal ballot language.

**Validation:** Fourteen deterministic tests pass. Eight GPT runs covering sizes 2-7 produced 252 decision turns. Exact response context was routed on 134/143 eligible turns (93.7%, archived baseline 71.7%), and manual review found discussion turns generally answered the targeted point. Response length visibly tracks traits: in the `n=6` run, the high-extraversion/high-length speaker averaged 21.1 words while three low-extraversion/short speakers averaged 9.5-9.8. The remaining 15 self-repetition warnings concentrate in long `n=5`/`n=7` runs and remain open with pacing/semantic exhaustion rather than being marked resolved.

**How to apply:** Preserve the hierarchy: exact local message, one routed job, act-specific trait behavior, then bounded facts/recent chat. Do not restore full biography on targeted turns or add generated initiative/directness state.
