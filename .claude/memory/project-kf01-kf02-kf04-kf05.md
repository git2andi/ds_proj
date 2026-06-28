---
name: project-kf01-kf02-kf04-kf05
description: "KF01/KF02/KF04/KF05 resolved 2026-06-28 — preference validation, outcome taxonomy, alias map, hard-blocker vote guard"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a9b7b55-4d66-4264-ac3a-d9d8c971642c
---

KF01 resolved: replaced `_apply_preference_plan()` with `_validate_preference_plan()`. Setup now fails-and-retries if same camp chose different options or all camps chose the same option. Prompt shows participant names alongside IDs. Safe postprocessing ensures `preferred_option` in `acceptable_options`. Minimum acceptable-options guarantee added: if non-stubborn persona still has <2 acceptable options after all cleanup, best-scoring non-hard-rejected option is added.

KF02 resolved: outcome taxonomy is now `successful / majority / unresolved`. `support_fraction()` counts only visible `explicit_vote` and `accepted_options`, not hidden `current_preference` or `preferred_option`. Updated `dialogue.py` and `prompts.py`.

KF04 resolved: `OptionResolver._build_aliases()` now also registers `OptionCard.short_name` if `len(short_name) >= 4`. Collision detection still prevents ambiguous aliases.

KF05 resolved: `MessageValidator._check_hard_blocker_vote()` added in `validation.py`. If `agreeableness == 1` and a parsed trailer (move.present=True) names a non-preferred option on a VOTE/ACCEPT turn, triggers `HARD_BLOCKER_WRONG_VOTE` repair.

**Why:** These were P0 setup and outcome-state correctness failures. KF03 (visible commitment text) remains open as next P0 item.

**How to apply:** Known failures priority now starts at KF03 then KF06.
