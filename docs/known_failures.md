# Known Failures

Tracked failures and quality issues in the dialogue simulator. This is the single file for tracking what needs fixing.

Last updated: 2026-06-27 (R11-R28 all fixed; R29 open — low priority).

---

## How to implement fixes

Every change follows this process — no exceptions:

1. **Pick one item** from the priority list below.
2. **Implement the fix.** Run `pytest tests/` to verify nothing breaks.
3. **Validate with example runs.** One n=3 run is mandatory. Then pick 5-6 more from n=2–7, using a randomly selected topic from `evals/topics.txt`. Always use the `uni` provider.
4. **Read the transcripts.** Evaluate whether the change actually improved things. Check for regressions — did something else get worse?
5. **If new issues surface**, add them directly to this file before moving on.
6. **Update** CLAUDE.md, known_failures.md, memories and/or skills to be always up to date with the latest changes.
7. **Only when all runs are complete and reviewed**, move to the next item.

Do not batch multiple fixes before validating. One fix, validate, evaluate, then next.

---

## Current state

The simulator produces coherent, responsive discussions with natural turn lengths, persona-specific voice, and concrete concessions. Decision mechanics work. Remaining issues are model-dependent (llama3.3 compliance with prompt instructions).

- **decision mechanics:** working — consensus, fallback, unresolved outcomes distinct,
- **grounding:** working — responding-to anchoring, room-questions routed to option champions,
- **repair / malformed output:** low on most topics (5–15%); some topic/persona combos reach 20–38% (model-dependent),
- **moderator:** diagnostic before directive, stall window scales with group size, topic-appropriate closures,
- **persona voice:** two-signal speaking habits (12 primary branches, all traits covered), face-work modifiers for REACT/SUPPORT/COMPARE/OBJECT/PUSH_BACK,
- **concessions:** option-specific with persona's reservation referenced,
- **option generation:** specific realistic names with concrete comparable attributes.

---

## Fixed

- **F1–F10:** Structural issues (mid-discussion accept counting, hard-blocker folding, self-narration, duplicate moderator lines, setup instability, excessive repair, interaction grounding, outcome categories, compromise mechanisms) — all resolved.
- **O1:** Argument-card turns → situated reactions via prompt rewrite and discourse-frame tracking.
- **O2:** Option name overuse → alias instruction after opening (~70% model compliance).
- **O3:** Persona differences → behavioral speaking habits ("blunt — cuts to the chase", "worrier — flags what could go wrong").
- **O4:** Social face-work → trait-based modifiers for objection/push-back acts.
- **O5:** Preference shifts → concrete concession bridges referencing option name and persona's reservation.
- **O6:** Consensus too coarse → hedged accept detection ("still not sure", "not fully sold") clamped to neutral.
- **O7:** Moderator imposed → diagnostic questions, participant names in prompts, stall window scales with n.
- **O8:** Practical constraints late → stall-to-concrete routing (ASK doubled after 2+ no-progress turns).
- **O9:** Endings thin → topic-appropriate closure (no generic "book it"), specific blocker naming.
- **O10:** Surface artifacts → deterministic cleanup of space-before-punctuation, repeated punctuation, stray quotes.
- **O11:** Turn-taking too balanced → extraversion/initiative weight, reduced catch-up boost for n>=5.
- **O12:** Large groups → skip-if-nothing-new for n>=4 when option already well-covered.
- **Room questions ignored** → genuine questions to the room now registered and routed to option champion.
- **Moderator too early for large groups** → stall window = base + max(0, n-3), minimum participant turns = n*2.
- **R6: No shared decision-situation context** → `shared_context` field added to Scenario dataclass, setup prompt, moderator opening, and per-turn prompt. 2-3 stable situational facts generated alongside options.
- **R7: Participant names lack variety** → 48-name diverse pool in `builders.py`, pre-sampled and enforced in parser.
- **R1: Stock phrases persist** → deterministic `fix_stock_phrases` rewrites "is a must for me" → "matters to me", "major draw" → "appeals to me" etc. in `clean_generated`. No LLM call.
- **R5: Repetitive agreement loops** → when concentration_score == 1.0 and no_progress_count >= 2, force narrowing transition.
- **R2: Farewell lines stiff** → explicit ban of formal closers in farewell prompt ("looking forward to", "confirmed and set", "satisfied with", "have a great day").
- **R3: Full option names after opening** → `_short_alias` generates concrete 2-word aliases from option names, shown in the alias instruction. Model shortens consistently.
- **R8: Bad alias truncation** → `_short_alias` used to blindly take first 2 words, producing "Wine and", "Settlers of", "Ticket to". Fixed: `OptionCard` now has a `short_name` field generated by the setup LLM (e.g. "Ticket to Ride" → "Ride", "Settlers of Catan" → "Catan", "Carcassonne" → "Carc"). `_clean_short_name` in `builders.py` rejects any LLM output that ends on a stopword. Deterministic fallback upgraded: 3-word names use the full name; 4+ word names skip word[1] if it is a dangling conjunction/preposition.
- **R4: Named addressee rate low** → address rule now encourages using addressee's name once in the message.
- **R9: Linguistic template leakage** → seven turn-patterns ("X seems like a good fit", "X still beats Y", "What if we...", "Wait, what about...", "works for me", "Giving up X, gaining Y", "I'm voting for X because") repeated across unrelated topics, making all speakers sound like one hidden agent. Root cause: guidance strings in `_concession_bridge`, `_face_work`, and `_move_guidance` in `prompts.py` literally contained these phrases as examples, seeding them verbatim. Fixed by: (1) rewriting `_concession_bridge` to use persona-specific conditional bridges referencing `opt` and `worry` without template verbiage; (2) replacing `"Frame it as 'what if we...'"` with description-only guidance; (3) removing "Try: '...'" examples from PROPOSE_COMPROMISE; (4) removing seeded phrases from ACCEPT/VOTE/COMPARE/REACT guidance strings; (5) adding `seems like a good fit` and `still beats/wins` to `_ROBOTIC_TEMPLATES` in `validation.py`; (6) adding `what_if_opener` and `wait_what_about` patterns to `_FRAME_PATTERNS` for variety-hint injection. Validated: all seven patterns eliminated across 3 runs.
- **R10: Invented context** → speakers asserted real-world facts not in the option cards ("big room in back", "group deals", "happy hour", etc.), especially for topics with named real-world places (cafes, restaurants). Root cause: model draws on training knowledge of real places rather than staying within the cards. Fixed by replacing the per-turn "hedge invented facts" rule with a stronger epistemic constraint: "You know only what's in the option cards — anything else is unknown: say 'I'm not sure' or 'we'd need to check', never a confident claim." Result: speakers now use "we'd need to check their seating capacity" and "if they have one" instead of asserting invented attributes. Also updated `_REPAIR_HINTS` for `INVENTED_OPTION_ATTRIBUTE` and `UNGROUNDED_NUMERIC_FACT` to suggest hedging. Validated: confident invented claims eliminated in re-run of cafe topic (hardest case for this issue).

- **R11: "edges ahead" template** → phrase appeared 3× per run across different speakers (board game and restaurant topics). Root cause: `_move_guidance()` COMPARE guidance said "where yours edges ahead" — exact phrase the model copied verbatim (same class as R9). Fixed by: (1) rewriting COMPARE guidance to "Acknowledge one genuine plus of theirs, then say why you'd still pick yours. No attribute lists."; (2) adding `re.compile(r"\bedges?\s+ahead\b", re.I)` to `_ROBOTIC_TEMPLATES` in `validation.py`; (3) adding 'edges ahead' to the banned stock-phrases list in rule 3 of `sim_utterance`. Test added. Validated: phrase absent across 2 new runs (board game + restaurant).
- **R12: Question echo in ANSWER turns** → when routed to answer an unanswerable question (not covered by option cards), the model copied the question back verbatim instead of hedging — producing 3-question echo chains (Tala→Diego→Tala all asking the same question about train car count). Root cause: `ActType.ANSWER` had no explicit guidance in `_move_guidance()`, falling through to the generic "Respond to the last point directly." Fixed by adding an explicit ANSWER case: "Answer if the option cards cover it. If they don't, say you're not sure and move on — don't repeat the question back." Validated: echo chain eliminated in board game re-run; residual mild echo (2 turns, warn-level only) in restaurant run when the echoing turn is not ANSWER-routed.
- **R13: Question echo deterministic backstop** → prompt guidance alone (R12) didn't prevent 2-turn question echoes when the echoing turn is non-ANSWER-routed (GROUP_REPETITION fired as warn-only, so no repair). Fixed in `validation.py` `_check_repetition`: when GROUP_REPETITION fires and both the current and the matched turn contain "?", issue `QUESTION_ECHO` at repair level instead of GROUP_REPETITION at warn. Repair hint: "don't re-ask what was just asked — if the cards don't say, hedge and move on." Two tests added. Validated: QUESTION_ECHO repairs triggered=0 in restaurant re-run (echo never occurred; prompt+backstop together prevent it).
- **R14: "still pick" template** → phrase "I'd still pick X because..." appeared 3–4× per run across all speakers in 4/6 new runs (farewell gift, sci-fi, framework, TV series). Root cause: R11's COMPARE guidance fix introduced the replacement phrase "why you'd still pick yours" — same R9/R11 class of template seeding. Fixed by: (1) rewriting COMPARE guidance to "One genuine strength of theirs; one concrete reason yours fits you better. No attribute lists, no templates." — no verb phrase that can be lifted as a sentence; (2) adding `re.compile(r"\bstill\s+pick\b", re.I)` to `_ROBOTIC_TEMPLATES`; (3) adding 'still pick' to banned stock-phrases list in rule 3. Validated: phrase absent in re-runs of farewell gift and sci-fi (both eval-clean, echoed dropped from 4→1 and 2→0 respectively).
- **R15: Back-to-back routing at opening→answer boundary** → a speaker could get two consecutive turns when (a) they were next in the opening queue and (b) the router immediately routed them to ANSWER a question raised by someone else's opening statement. Root cause: `next_intent()` checks `state.open_questions` after the opening phase ends, with no guard for the last participant speaker. Fixed in `router.py`: before emitting a question-answer `MoveIntent`, check if `target == last_participant.speaker_id`; if so, skip (question stays queued and is picked up after someone else speaks). Validated: back-to-back eliminated in farewell gift re-run.

---

- **R16: Moderator cuts off open question at closure** → when a participant generated a question on an ACCEPT-routed turn in the confirmation phase, two paths led to the moderator closing immediately after — cutting off the unanswered question. Root cause 1: `_INTENT_FALLBACK_STANCE[ActType.ACCEPT] = "accept"` in `parsing.py` — when the model produces no stance in its trailer, the fallback infers an accept; a question text with no trailer triggered this, silently crediting the speaker as having accepted the candidate option, triggering premature consensus. Root cause 2: the CONFIRMATION→CLOSURE timeout (`max_confirmation_turns`) in `DialogueController.update_phase()` had no guard against open questions, so even after fix 1 (question no longer credited as accept), the timeout could still fire and close mid-question. Fixed by: (1) in `_resolve_move()`, after applying `_INTENT_FALLBACK_STANCE`, override stance back to "neutral" if `question_target` is set (the model asked something, so it did not commit); (2) in `update_phase()`, guard the confirmation timeout with `if not state.open_questions:` — `hard_max_turns` still provides the unconditional backstop. Two tests added. Validated: question-before-closure pattern absent in sci-fi re-runs; `unresolved` outcome correctly issued when true disagreement exists.

---

- **R17: Fallback outcome uses drifted leading option** → `finalize()` called `leading_candidate(state)` = `leading_option(state)`, which is the live `option_support` score at closure time. After a long confirmation phase with many PROPOSE_COMPROMISE turns, `current_preference` values can drift as speakers propose different options, causing `leading_option` to return an option that nobody explicitly voted for — one with <0.66 support — producing `unresolved` even when 2/3 explicitly voted for the true candidate. Root cause: `leading_option` is a real-time score that changes as preferences drift; the correct fallback target is the option the group narrowed to in the vote round (`state.candidate_option`). Fixed by replacing `self.leading_candidate(state)` with `state.candidate_option or self.leading_candidate(state)` — the confirmed candidate takes priority; `leading_option` is only the fallback when no candidate was set (e.g. if the group never reached narrowing). Validated: 3 runs with clear majority outcomes.

---

- **R18: Back-to-back routing in narrowing** → `_vote_intent` moved last speaker to the END of the unvoted list in the `else` branch, but didn't hard-exclude them when others were available. Tightened to: `others = [p for p in unvoted if p.id != last_pid]`; `ordered = others if others else unvoted` — last speaker only votes when they're the sole remaining voter. Sub-fix: when `others` is empty and the only unvoted persona just had a VOTE-routed turn (UNCLEAR_VOTE — their trailer had no valid option, so `explicit_vote` was never set), routing them again immediately caused a consecutive VOTE pair. Guard added: if `not others` and `last_turn.intent.act == ActType.VOTE`, advance straight to CONFIRMATION rather than re-picking the same speaker.
- **R19: POSSESSIVE_SUBJECT opener** → (a) Escalated from warn to repair level so the model is forced to rewrite possessive openers. (b) Fixed pattern-building: option names with parenthetical annotations (e.g. "Inception (2010)") were stripped of the parenthetical so "Inception's" matches. (c) Added `short_name` patterns so abbreviated forms ("Budapest's" for "The Grand Budapest Hotel") are also caught. Prompt rule moved to Rule 1 (higher weight). Named rate improved to 26–31%.
- **R20: Repeated openers (REPEATED_START)** → (a) Opener feedback added to `runtime_speaker_card`: last 2 turn openers shown, with "start differently this time" instruction — gives the model concrete context about its own recent openers. (b) REPEATED_START escalated from warn to repair, triggering an LLM rewrite. (c) Repair hint made dynamic: instead of generic "open with different words", the hint now names the exact repeated phrase ("don't start with 'Do they offer' — use a completely different first word or phrase"). Result: REPEATED_START dropped from 11–25 per 10-run batch (warn-only) to 0–3 with repair triggered on each hit.
- **R21: Named-addressee rate low** → Address rule now fires for ANSWER/REACT acts even without an explicit `addressee_id`: the responding-to name is extracted from the `_responding_to_line` string and injected as a compact "use their name once" instruction. Named rate improved from ~13% to 26–31%.
- **R23: Back-to-back questions from different speakers** → Hard-zero the ASK act weight in `_select_act` when the immediately preceding participant turn contained "?". The prior `ask_after_question_damping=0.40` was too soft — reduced probability but didn't prevent routing another ASK turn immediately after a question. Hard veto (`probs[ActType.ASK.value] = 0.0`) closes the router-level case. Incidental questions appended to non-ASK turns (REACT/COMPARE) are caught by the QUESTION_ECHO repair backstop (R13). Validated: no back-to-back question pairs in holiday party or programming language runs.
- **R22: Trait differences not perceptible** → Added compact `Traits: extra=N agree=N neuro=N` line to `runtime_speaker_card` so llama3.3 has numeric calibration signal alongside the speaking-habit description. Validated: terse vs verbose personas now read more distinctly across 5 runs (Isla/Nico terse, Jasper/Yara more elaborate).
- **R24: Asker routed to answer own question** → In `_best_answerer`, built `other_askers` set from all open questions (excluding the current one), then added two-tier eligibility filter: first prefer non-asker non-other-asker candidates, then fall back to just non-asked_by. Tested with three cases: own question, option champion, cross-question exclusion.
- **R25: Duplicate substance across speakers** → `covered_slots_hint` was broken — always received `text_slots=[]` so condition `not repeated` was always True and hint never fired. Removed `text_slots` parameter; now fires when `len(covered) >= 3`. Rewrote message to "The group already argued…". Call site updated.
- **R26: ANSWER-turn echo loop (R13 gap)** → When an ANSWER-routed turn echoed its question (QUESTION_ECHO repair failed, message kept), `_update_questions` re-registered the kept "?" as a new OpenQuestion, cycling indefinitely. Fixed in `StateTracker._update_questions`: if the turn was ANSWER-routed AND `QUESTION_ECHO` is in `validation_issues`, suppress propagation. Validated: holiday party 3-question echo loop eliminated.
- **R27: SELF_REPETITION skipped for ACCEPT/REJECT intents** → Validation returned early before consulting `already_said` for ACCEPT/REJECT acts ("Confirmations are naturally similar"). This let the exact same sentence repeat 3× on ACCEPT-routed turns without firing (seen: Leo's "Taco Loco offers a unique twist." in restaurant run). Fixed: removed the early-return exemption so SELF_REPETITION checks `already_said` regardless of act type. Added test.
- **R28: "we'd need to check" epistemic phrase chorus in n=6 run** → Epistemic grounding guidance listed exactly two alternatives ("I'm not sure" or "we'd need to check"), making "we'd need to check" the model's default. 7 consecutive turns in the fictional world run all ended with it. Fixed: expanded to 5 alternatives ("I'm not sure", "can't say", "we'd have to check", "no idea", "unknown to me") with an explicit "vary the phrasing" instruction. Validated: variety improved across all 5 follow-up runs (e.g. "unknown to me", "can't say for sure", "no idea how that would play out").

---

## Open items (priority order)

### R29: "Considering..." opener persisting (warn-only, model non-compliance)

**Symptom:** Speakers open turns with "Considering the kids…", "Considering our group's size…", "Considering the vibrant arts scene…" despite it being in the prompt's no-stock-phrases list. The ROBOTIC_TEMPLATE check detects it as warn-level but does not trigger repair. Observed 4+ hits per run in n=4 vacation run.

**Root cause:** ROBOTIC_TEMPLATE is warn-only. The prompt forbids "Considering..." but llama3.3 complies inconsistently — some speakers get it, some don't. Without a repair enforcement layer, the phrase leaks into 1-7 turns per run depending on topic.

**Fix candidate:** Escalate the `^\s*considering\b` pattern from warn to repair-level (same escalation used for POSSESSIVE_SUBJECT in R19, REPEATED_START in R20). This adds one LLM repair call per hit but closes the gap. Alternatively, add "Considering" to the REPEATED_START window so it's caught by the dynamic "don't start with X" hint if it appears twice.
