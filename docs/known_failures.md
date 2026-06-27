# Known Failures

Last updated: 2026-06-27. F1–F55 fixed (F14, F18, F50 removed — contradicted by R37, R33, R36). All R29–R38 resolved.

---

## Implementation process

1. Pick one item from the priority list.
2. Implement. Run `pytest tests/`.
3. Validate: one n=3 run mandatory, then 5-6 more from n=2–7 with random topics from `evals/topics.txt`. Provider: `uni` always.
4. Read transcripts. Did it improve? Any regressions?
5. New issues → add here before continuing.
6. Update CLAUDE.md, memories. Only then move to next item.

---

## Fixed (F1–F55)

**F1** Mid-discussion accepts counted as binding → commitment gating; ACCEPT/VOTE binding only in narrowing/confirmation.
**F2** Hard-blocker personas folded to non-preferred → only backs preferred; votes elsewhere ignored.
**F3** Self-narration in turns → banned from prompt.
**F4** Duplicate moderator lines → deduplication guard.
**F5** Setup instability (invalid worlds silently defaulted) → raise on unusable setup output.
**F6** Excessive repair → repair hints targeted; structural errors only trigger LLM repair.
**F7** No grounding to prior turns → `_responding_to_line` anchors each turn to most relevant prior.
**F8** Outcome categories too coarse → consensus / fallback / unresolved distinct.
**F9** No compromise mechanism → PROPOSE_COMPROMISE act with bridge guidance.
**F10** Argument-card recitation → discourse-frame/claim-slot tracking; situated reactions via prompt rewrite.
**F11** Option name overuse → alias instruction after opening; ~70% compliance.
**F12** Flat persona voice → behavioral speaking habits (12 branches covering all traits).
**F13** No face-work on objections → trait-based modifiers for OBJECT/PUSH_BACK.
**F15** Hedged accepts counted as binding → "still not sure", "not fully sold" clamped to neutral.
**F16** Moderator always directive → diagnostic questions; stall window scales with n.
**F17** Practical constraints never surface → stall-to-concrete routing; ASK doubled after 2+ no-progress turns.
**F19** Surface artifacts (space-before-punctuation etc.) → deterministic cleanup in `clean_generated`.
**F20** Turn-taking too balanced → extraversion/initiative weight; reduced catch-up boost for n>=5.
**F21** Large groups repeat known positions → skip-if-nothing-new for n>=4.
**F22** Room questions unrouted → registered as OpenQuestion; routed to option champion.
**F23** Moderator interrupts too early for large groups → stall window = base + max(0, n-3).
**F24** Stock phrases ("is a must for me", "major draw") → deterministic `fix_stock_phrases` in `clean_generated`.
**F25** Farewell lines stiff → formal closers banned in farewell prompt.
**F26** Option alias bad truncation ("Wine and", "Settlers of") → `short_name` LLM-generated (e.g. "Ticket to Ride"→"Ride"); `_clean_short_name` rejects stopword-ending; deterministic fallback upgraded.
**F27** Named-addressee rate low → address rule encourages name once per turn.
**F28** Repetitive agreement loops → when concentration==1.0 and no_progress>=2, force narrowing transition.
**F29** No shared situational context → `shared_context` generated at setup; injected into moderator opening and per-turn prompt.
**F30** Participant names repetitive → 48-name diverse pool pre-sampled in `builders.py`.
**F31** Seven turn-patterns seeded by guidance strings ("X seems like a good fit", "still beats", "What if we...", etc.) → guidance strings rewritten to describe behavior without quoting phrases; patterns added to `_ROBOTIC_TEMPLATES`.
**F32** Speakers assert real-world facts not in cards → epistemic constraint: "You know only what's in the option cards." Repair hints updated.
**F33** "edges ahead" phrase seeded by COMPARE guidance → guidance rewritten; phrase added to `_ROBOTIC_TEMPLATES` + stock-phrases ban.
**F34** Question echo in ANSWER turns → explicit ANSWER guidance: "don't repeat the question, hedge and move on."
**F35** 2-turn question echo when non-ANSWER-routed → GROUP_REPETITION on question pairs escalated to QUESTION_ECHO repair.
**F36** "still pick" phrase seeded by F33's COMPARE rewrite → guidance rewritten again (no verb phrases); "still pick" added to `_ROBOTIC_TEMPLATES`.
**F37** Back-to-back at opening→answer boundary → guard in `next_intent()` skips ANSWER routing if target just spoke.
**F38** Moderator closes mid-question in confirmation → (a) question stance override in `_resolve_move`; (b) confirmation timeout guards `if not state.open_questions`.
**F39** Fallback targets drifted `leading_option` → `finalize()` prefers `state.candidate_option` over live lean score.
**F40** Back-to-back votes in narrowing → last-speaker excluded from unvoted list; consecutive UNCLEAR_VOTE advances to CONFIRMATION.
**F41** POSSESSIVE_SUBJECT opener warn→repair → escalated; pattern covers `short_name` forms and names with parentheticals. Named rate 26–31%.
**F42** Repeated openers → opener feedback in speaker card (last 2 shown); REPEATED_START escalated to repair with dynamic hint naming exact repeated phrase.
**F43** Named-addressee rate low in ANSWER/REACT without explicit `addressee_id` → responding-to name injected as "use their name once" instruction.
**F44** Trait differences not perceptible → `Traits: extra=N agree=N neuro=N len=N` added to `runtime_speaker_card`.
**F45** Back-to-back questions from different speakers → hard-zero ASK weight when preceding participant turn contained "?".
**F46** Asker routed to answer own question → `_best_answerer` two-tier filter: prefer non-asker non-other-asker candidates first.
**F47** `covered_slots_hint` never fired (dead code) → broken `text_slots` param removed; fires when `len(covered) >= 3`.
**F48** ANSWER echo loop (F35 gap) → if ANSWER-routed turn has QUESTION_ECHO in issues, `_update_questions` suppresses propagation.
**F49** SELF_REPETITION skipped for ACCEPT/REJECT → early-return exemption removed; `already_said` checked for all act types.
**F51** ANSWER turns invent non-card facts → ANSWER guidance states card attributes are exhaustive; "never invent facts."
**F52** response_length absent from trait card → `len=N` added to Traits line.
**F53** "Considering..." opener warn-only despite repair escalation pattern → split from `_ROBOTIC_TEMPLATES` into `_CONSIDERING_OPENER` constant; fires repair on `^\s*considering\b` match only.
**F54** ANSWER-routed turn can contain "?" without triggering repair → `ActType.ANSWER` added to `statement_only` set in `_check_unwanted_question`; echo loop broken.
**F55** Epistemic hedge phrases ("do they offer", "why that matters") became chorus → both added to `_ROBOTIC_TEMPLATES`; ANSWER guidance rewritten to describe hedging behavior without listing example phrases; R9-class mistake corrected.
**F56** Fallback closure phrased as unanimous win → `moderator_closure_prompt` fallback case now opens with "agreement wasn't unanimous"; names remaining concerns before naming the pick.
**F57** Acceptance of non-preferred option was one thin sentence → `_concession_bridge` now enforces two-part structure: (1) own that it wasn't your pick, (2) name the specific trade-off you're accepting as a cost.
**F58** Open questions silently cleared after one hedge → `hedge_count` field on `OpenQuestion`; first hedge keeps question open for one more routing cycle; cleared on real answer or second hedge.
**F59** Persona role/concern contradicts preferred option (setup) → `setup_personas` prompt adds consistency constraint: "a quiet reader prefers a calm option; never assign a preferred option whose core attributes directly contradict the persona's role and concern."
**F60** Token cost regression (~840–875 t/turn for n=3) → resolved as side-effect of F55: removing epistemic phrase list from prompt shortened per-turn cost to ~675 t/turn (~20% reduction).

---

**F61** UNCLEAR_VOTE guidance confusing — "State your pick without 'I'm voting for X because'" read as "be vague" → rewritten: "Say the option name out loud and commit to it." Repair hint sharpened to match. (2026-06-27)
**F62** "Considering X, Y" opener survives through repair (model regenerates it) → deterministic strip `_strip_considering_opener()` added to `clean_generated()`; removes the dependent clause before validation runs. (2026-06-27)
**F63** Hard-blocker voted for non-preferred option: UNCLEAR_VOTE didn't fire (stance was "vote", only option letter was wrong) → `HARD_BLOCKER_WRONG_VOTE` validation check added; fires repair with named option hint when hard-blocker trailer names a different option. (2026-06-27)
**F64** R39: OBJECT guidance now says "from the option card — don't invent flaws not mentioned there". Closes gap where epistemic rule covered positive claims but not negative invented attributes. (2026-06-27)
**F65** R40: "that's a great question / that is a good question" added to `_ROBOTIC_TEMPLATES`; "valid point" swapped for "great question" in sim_utterance rule 3 banned list (still 9 phrases; "valid point" already caught by broader pattern). (2026-06-27)

**F66** Vague ACCEPT text ("handling worries me less with X") counted as consensus — ACCEPT/VOTE guidance rewritten: explicit name + commit required in text; face_work R9-class seeding phrase removed. (2026-06-27)

**F67** Option-name-led openers ("Go offers great performance...") — `OPTION_NAME_OPENER` warn-level check added to `MessageValidator`; Rule 2 updated: "Don't open with just an option name." VOTE/ACCEPT exempt. (2026-06-27)

---

## Open items

**F68** Occasional short backchannel turns missing — real group chats include brief whole-turn reactions ("yeah", "fair point", "good call") but only rarely (1–2 per run max, never back-to-back). Currently all turns are substantive. The fix must be fully natural: the trait system already produces short turns via low `response_length` and `detail` traits — lean on that rather than adding mechanical injection or forced word-count checks. A low-`response_length` persona in a REACT or SUPPORT act after a long turn should naturally produce a short reaction without any extra routing logic. Do NOT add a fixed probability injection, forced branching, or validation repair code — these produce stilted artificial backchannels. Instead, investigate whether the verbosity note for low-detail personas (currently "~N words max. You keep it short") produces short-enough outputs in practice on REACT acts, and whether REACT-routed turns after a long substantive turn naturally land at 3–6 words without intervention. *Priority: low — observe first, implement only if observation confirms a gap.*

**F69** High repair rate on large-group runs (n=5–6, ~27–48%) — `repair_trigger_codes` now logged in run.json. Diagnosis from n=5 runs: INVENTED_OPTION_ATTRIBUTE=~5 per run, REPEATED_START=4 (vote phase), POSSESSIVE_SUBJECT=3. Two sub-issues: (a) model invents prices/capacity on ANSWER/PROPOSE_COMPROMISE; repair forces a hedge but cost extra tokens — fix: add "never invent numbers" to PROPOSE_COMPROMISE guidance; (b) repair-generated question turns ("do they have space limits...") can loop across 2-3 speakers when multiple people ask the same invented-attribute question — QUESTION_ECHO doesn't catch these because max_repairs_per_turn=1 and the second speaker's first-attempt also triggers INVENTED_OPTION_ATTRIBUTE before the echo is detectable. *Priority: medium.*

**F70** New sycophantic templates not caught — "[Name] brings up a good point", "[Name] makes a strong point" escaped existing `_ROBOTIC_TEMPLATES`. Pattern added: `\b(?:brings?\s+up|makes?|raises?)\s+(?:a|an)\s+(?:great|good|fair|valid|strong|excellent|interesting|important)\s+(?:point|concern|question|issue)\b`. Also added to Rule 3 banned list. (2026-06-27)
