# Known Failures

Failures and quality issues tracked across evaluation passes. Each entry records
the symptom, root cause, fix status, and how to detect a regression.

## Fixed

### F1: Mid-discussion "accept" counted as binding commitment
- **Symptom**: Premature fake-unanimous outcomes while chat is still arguing (12/19 accepts in an n=7 run happened mid-discussion).
- **Root cause**: The model liberally tagged ordinary discussion lines `stance=accept`; these were recorded as real acceptances.
- **Fix**: Commitment only honoured on routed decision turns (narrowing vote / confirmation). `parsing._resolve_move`, gated by `_DECISION_ACTS`.
- **Regression signal**: `run.json` turns in discussion phase with `act_type=accept` that have `explicit_vote` set.

### F2: Hard blockers could be talked into accepting non-preferred options
- **Symptom**: A hard-blocker who preferred A "accepted" C then B, turning a deadlock into fake consensus.
- **Root cause**: No guard preventing hard-blocker stance shifts.
- **Fix**: Hard blocker only ever backs their preferred option; any vote/accept elsewhere ignored. `dialogue.StateTracker._update_runtime`, `_can_back`.
- **Regression signal**: In `run.json`, a persona with `is_hard_blocker=true` whose `explicit_vote` or `accepted_options` includes a non-preferred option.

### F3: "Wall of I" — every turn opens with self-anchored stance
- **Symptom**: "I'm drawn to X because...", "I worry that Y...", "I prefer Z..." — parallel monologues, not conversation.
- **Root cause**: Opening prompt said "say which option you're drawn to"; repetition guards were warn-only; no opener monotony check.
- **Fix**: Prompt rewrite (react-first), `validation._check_opener_variety` / `REPETITIVE_OPENER`, `REPEATED_START` promoted to repair.
- **Regression signal**: `evals/run_eval.py` `opener_variety` check: >50% of participant turns opening with I/I'm/we/our.

### F4: Possessive opener tic ("X's `<feature>`")
- **Symptom**: ~40% of turns in large-group runs opened with `<OptionName>'s <attribute>`.
- **Root cause**: Model default phrasing; no deterministic guard.
- **Fix**: `validation._check_robotic_phrasing` / `_possessive_openers`.
- **Regression signal**: `POSSESSIVE_SUBJECT` in validation issues; `evals/run_eval.py` robotic-template count.

### F5: Hedged acceptance closing the chat
- **Symptom**: "might be okay if there's free time" recorded as firm accept, closing prematurely.
- **Root cause**: No hedge detection on confirmation accepts.
- **Fix**: `parsing._HEDGED_ACCEPT` — tentative/conditional acceptance stays neutral.
- **Regression signal**: `test_parsing.py::test_hedged_accept_*`.

### F6: Moderator repeating stock lines verbatim
- **Symptom**: Two identical "anyone object or lock it in?" nudges in one run.
- **Root cause**: No prior-line dedup for moderator.
- **Fix**: Prior facilitator lines fed back into moderator prompt with "say it differently" rider.
- **Regression signal**: Duplicate moderator lines in transcript (checked by `evals/run_eval.py`).

### F7: Formulaic templates leaking through
- **Symptom**: "X outweighs Y", "X's point is valid, but", "makes me think...", "Given the discussion...", etc.
- **Root cause**: Model reaches for these despite prompt bans.
- **Fix**: `validation._ROBOTIC_TEMPLATES` deterministic catch.
- **Regression signal**: `ROBOTIC_TEMPLATE` in validation issues; `test_validation.py::test_robotic_template_*`.

### F8: Collective "we" for personal stances
- **Symptom**: "We consider...", "We prioritize..." — individual sounds like a committee.
- **Root cause**: Model default register.
- **Fix**: `validation.fix_collective_voice` deterministic rewrite + warn-level backstop.
- **Regression signal**: `test_validation.py::test_collective_voice_*`.

### F9: Card-reading — turns parrot option card text verbatim
- **Symptom**: "Beach relaxation offers a great comfort level with its high comfort attribute", "Cozy atmosphere and excellent service" quoted word-for-word from the card.
- **Root cause**: Model echoes the option card prose fields it sees in the prompt.
- **Fix**: `validation._check_card_reading` flags 4+ word verbatim matches from upside/tradeoff/concern/best_for fields. Prompt rule: "never parrot an option card's description".
- **Regression signal**: `CARD_READING` in validation issues; `test_validation.py::TestCardReading`.

### F10: Self-narration ("I should consider...")
- **Symptom**: "I should consider the extensive wine list", "I should prioritize a place with..." — nobody talks like this.
- **Root cause**: LLM default reasoning register.
- **Fix**: `validation._check_self_narration` flags "I should/need to consider/prioritize/think about/..." patterns. Prompt rule: "never narrate your own thinking".
- **Regression signal**: `SELF_NARRATION` in validation issues; `test_validation.py::TestSelfNarration`.

### F11: Same-speaker back-to-back turns
- **Symptom**: Ava speaks twice in a row with near-duplicate content.
- **Root cause**: `_pick_speaker` used windowed recent list for last-speaker check; moderator turns could push the real last participant out of the window. `_opening_intent` iterated personas in order without checking who just spoke (e.g. after greeting).
- **Fix**: `_pick_speaker` now uses `_last_participant_turn` for the actual last speaker. `_opening_intent` reorders candidates to avoid the last speaker.
- **Regression signal**: Adjacent participant turns with the same speaker_id in transcript.

### F12: Zero questions in discussions
- **Symptom**: question_density = 0.0 across multiple runs despite ask base probability being 0.16.
- **Root cause**: `ask` weight lost the weighted_choice against 6 other acts (combined ~0.84). Damping was also aggressive (0.25x).
- **Fix**: Raised ask base to 0.20, reduced damping to 0.40. ASK move guidance rewritten with casual example questions.
- **Regression signal**: question_density = 0.0 in metrics; no "?" in any participant turn.

## Open (lower priority)

### O1: Thematic content spread
- **Symptom**: A shared persona theme (e.g. "balance study and leisure") recurs across several speakers — lexically varied but same idea.
- **Status**: Acknowledged; pushing harder risks stilted text.
- **Detection**: Manual review; jaccard on theme-extracted keywords across speakers.

### O2: Softer template variants
- **Symptom**: "seems like a great choice", "makes sense to me", "wins out for me" — mild template feel.
- **Status**: Left uncaught; broadening the regex over-fires on natural usage.
- **Detection**: Manual review of vote/accept turns.

### O3: Rare template leak through repair (~1 per run)
- **Symptom**: A flagged template reproduced by the single repair attempt.
- **Status**: `max_repairs_per_turn: 1` limits fix attempts. More attempts = more LLM cost.
- **Detection**: `ROBOTIC_TEMPLATE` still present in `validation_issues` after repair.

### O4: Weak world/task grounding
- **Symptom**: Conversations stay at generic option labels and one or two static attributes. No embodied concerns ("I don't want to arrive at midnight").
- **Status**: By design — personas can't negotiate facts not on the option cards.
- **Detection**: Manual review of specificity in medium/long turns.

### O5: Under-explained stance changes
- **Symptom**: Participant shifts preference without visible persuasion trigger in the text.
- **Status**: Partially addressed by persuasion gating (min speaker turns, support margin). The *voicing* of the reason is still model-dependent.
- **Detection**: In `run.json`, look for turns where `current_preference` changes without a preceding argument for the new option.

### O6: Coverage imbalance
- **Symptom**: An option is eliminated without anyone explicitly dropping it.
- **Status**: Router ensures preferred/acceptable options are aired; truly unwanted options are left unmentioned by design.
- **Detection**: `option_coverage` in metrics — an option with zero mentions that was in someone's acceptable set.

### O7: Debate-thesis format
- **Symptom**: Every turn is "X is important because Y" — structured like a debate statement, not casual chat.
- **Status**: Addressed by prompt rewrite (conversational register, fragment encouragement). Model-dependent.
- **Detection**: Manual review; turns that all follow "[noun] is [adjective] because [reason]" pattern.

### O8: No conversational texture
- **Symptom**: Every turn is a grammatically complete formal sentence; no fragments, fillers, hedges, or casual reactions.
- **Status**: Addressed by prompt rewrite (explicit fragment/filler encouragement, example reactions). Model-dependent.
- **Detection**: Manual review; check for presence of short reactions ("yeah", "fair", "hmm") and sentence fragments.

### O9: Instant capitulation at moderator nudge
- **Symptom**: Holdouts fold with zero pushback ("Bistro Bliss' atmosphere could work for me too") when moderator asks.
- **Status**: Partially addressed by persuasion gating (min speaker turns before stance change). Prompt now says "don't fold in your first couple of turns".
- **Detection**: In transcript, holdout accepting immediately after first moderator nudge without any intervening argument.

### O10: Mechanical phrasing echoes across turns
- **Symptom**: Three consecutive turns about "generous portions" — different speakers chaining on the same noun phrase.
- **Status**: Echo guard catches verbatim lifts (6+ word runs). Shorter shared phrases (2-3 words) are harder to catch without over-firing.
- **Detection**: Manual review; `ECHOED_PHRASE` in validation issues for longer shared runs.
