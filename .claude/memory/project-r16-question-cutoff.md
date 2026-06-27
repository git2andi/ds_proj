---
name: project-r16-question-cutoff
description: R16 fix — moderator cuts off open question at decision closure via two distinct paths
metadata:
  type: project
---

When a participant generated a question on an ACCEPT-routed turn in the confirmation phase, the moderator could close the discussion immediately after — cutting off the unanswered question.

**Root cause 1 (parsing.py):** `_INTENT_FALLBACK_STANCE[ActType.ACCEPT] = "accept"` — when the model produces no stance in its trailer, the fallback infers an accept. A question with no trailer triggered this, silently crediting the speaker as having accepted the candidate option and triggering premature consensus.

**Root cause 2 (dialogue.py):** The CONFIRMATION→CLOSURE timeout (`max_confirmation_turns`) in `DialogueController.update_phase()` had no guard against open questions, so even after fix 1, the timeout could still fire and close mid-question.

**Why:** The fallback-stance mechanism was designed for the common case (model omits trailer on a genuine accept), but didn't account for the model generating a question instead of committing. Narrowing already has an open_questions guard; confirmation timeout didn't.

**How to apply:** Both fixes work together:
1. `_resolve_move()`: after `_INTENT_FALLBACK_STANCE`, override stance to "neutral" if `question_target` is set — a question is never a binding commitment
2. `update_phase()`: confirmation timeout uses `if not state.open_questions: state.phase = Phase.CLOSURE` — `hard_max_turns` is the unconditional backstop

Related: [[project-r13-question-echo-backstop]], [[project-r15-backtoback-routing]]
