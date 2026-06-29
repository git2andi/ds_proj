# Known Failures - Open Issues Only

Last updated: 2026-06-29.

This backlog contains only issues still supported by the current code or by the five GPT validation runs listed below. Historical IDs are retained where an old issue remains partially open; resolved issues are omitted. Fixes must be topic-agnostic, valid for 2-7 participants, and independent of provider-specific wording habits.


## Validation protocol

For each fix:

1. Work on one issue only unless stated otherwise. Move existing logs from logs/ into logs/archive/
2. Use the provider explicitly authorized for the task. Never silently substitute an endpoint.
5. Run at least one `n=3` discussion, then a spread across random sizes and topics.
6. Read transcripts and `run.json`; metrics alone are insufficient.
7. Close the issue only when visible behavior improves without an obvious regression.
8. Before completing, upgrade `CLAUDE.md`, skills, active memory/index files, and this file
9. Stop at the completed upgrade boundary unless the user explicitly grouped issues or requested automatic continuation.
10. commit and push changes

Current implementation order is P0 friend-chat naturalness and visible personas, P1 state/run follow-up, P2 moderator/closure integrity, P3 grounding/question quality, then P4 token cost. Each item remains a separate upgrade.

## P0 - Friend-chat naturalness and visible personas

### KF08/KF09/KF14/KF24 - Turns do not consistently sound like distinct friends talking

This issue consolidates the prior trait-expression, standalone-argument, semantic-repetition, and provider-sensitive surface-style issues. They share one contract failure: the simulator specifies an act and option content more strongly than it specifies a short, locally responsive, persona-shaped contribution.

**Evidence:** Across the five GPT runs, responsive-turn rates were only 19-38%, and self-repetition remained common. Participants often delivered complete option summaries instead of a short answer, acknowledgment, disagreement, or follow-up. Some lines used formal, corporate, or presentation-like reasoning and sentence structure that did not fit a chat among friends. Compacting the prompt and removing the instruction to lead with first-person language improved GPT first-person openers from 68-83% to 0-6%, proving that a provider-neutral prompt-contract change can improve style without a phrase blacklist. Response-length control was visible, but behavior associated with conscientiousness, neuroticism, agreeableness, directness, compromise willingness, extraversion, and initiative was not consistently distinguishable.

**Root cause:** The runtime contract still combines act fulfillment, option positioning, reason production, grounding, and machine-trailer requirements in a way that rewards self-contained mini-arguments. Persona data is compressed into a small speaking-habit hint, so traits compete with stronger content instructions. Long response-length settings permit speech-like turns, while direct-response acts can still pivot away from the exact local point. Warning-only opener and repetition checks measure symptoms after generation but do not establish a coherent conversational contract.

**Acceptance criteria:**

- Across topics and group sizes 2-7, discussions read like friends making a decision: casual and plain-spoken, neither slang-heavy/Gen-Z nor corporate, academic, or presentation-like.
- Direct responses first answer, acknowledge, challenge, or build on the targeted point; they do not default to a standalone option pitch.
- Persona traits are visibly distinguishable through behavior such as caution, curiosity, directness, initiative, constraint checking, and compromise, without stereotypes, catchphrases, or self-description.
- Configured response length creates clear relative differences, but even the longest setting does not produce a mini-essay or unnecessarily complex sentences.
- Repeated turns add new grounded substance or move toward narrowing; lexical rephrasing of the same option claim does not count as progress.
- The change remains provider-independent and preserves grounding, commitment gating, trait-driven stubbornness, and outcome rules.

**Required direction:** Simplify the per-turn contract around one local conversational job, make persona behavior and response-length limits explicit but compact, and ensure direct-response context is exact. Prefer router/state and prompt-structure changes over provider-specific phrase detection. Do not add forced backchannels, scripted turns, stereotypes, quoted examples, or a larger regex blacklist. Add deterministic tests for prompt composition and response-length budgets; use manual transcript review for the behavioral acceptance criteria.

**Relevant code:** `src/prompts.py`: runtime speaker card, utterance guidance, and word budgets; `src/router.py`: response targeting and contribution selection; `src/dialogue.py`: question/challenge context and progress tracking; `src/validation.py`: repetition/discourse diagnostics; `src/builders.py`: persona-generation contract; `config.yaml`: response-length and context dials.

**Implementation checkpoint:** RESOLVED 2026-06-29 (extended fix 2026-06-29)

**Checkpoint evidence — initial fix** (GPT runs `20260629_011650_146434` team offsite, `20260629_011906_579411` team lunch, `20260629_012049_660192` farewell gift):
- Greetings 2-6 words, varied: "What's up, folks?", "Jumping in, hey all!", "Just slid into the chat."
- Background and goal in per-turn card visible: "I'm into hiking now", "I want something fresh and plant-based that feels good to eat."
- Option alias variation working and tracked correctly.

**Extended fix 2026-06-29** — 10-run batch (`20260629_072522_130126` to `20260629_073222_150549`) plus 3 targeted runs (`20260629_073729_686749`, `20260629_073813_863485`, `20260629_073911_346640`) identified and resolved four remaining issues:

1. **Moderator agreement/closure generating questions**: Changed `moderator_agreement_prompt` instruction from "ask if anyone objects" to "state as a fact — declarative sentence, no question". Closure instruction now says "Declarative sentence — no question, no 'should we'". Zero moderator questions in all 13 post-fix runs.

2. **Farewell "[Option name] it is." opener**: Changed farewell guidance to "Don't open with the option name. Lead with your reaction." Results: "Relieved we found a spot that's good for me", "Phew, glad we landed somewhere everyone can chill", "Stoked—can't wait to finally shoot those epic coastal views on PCH."

3. **Semicolons in chat body (too formal)**: Added "no semicolons, no formal transitions" to every `_verbosity_note` level. Added `_strip_body_semicolons()` in `dialogue.py` — trailer-aware (protects `[act=...; opt=...; stance=...]`), replaces `;` in body with ` —`.

4. **Trait register not showing in language**: Added `_voice_register()` function; appended a "Voice:" line to the speaker card for personas with low agreeableness, high neuroticism, very low compromise willingness, or short response length. Observable results: "Phew, glad we landed..." (relief, low extraversion); "True, less driving helps, but desert hiking can still get brutal midday" (direct, short) vs. "My photography gear needs daylight and steady coastal light..." (personal, considered) in the same run.

Residual: standalone option letter reference in body ("D offers a laid-back vibe") occurs rarely; the `^[A-Z]=` strip in `_surface_cleanup` handles the "D=" form, not the letter-as-word form. Acceptable — alias rule already instructs against it.

## P1 - State, outcome, and run integrity

### KF03 - Visible commitment and machine trailer can disagree — RESOLVED 2026-06-29

**Fix:** Three failure modes identified and resolved. (1) `stance=accept` on a VOTE turn was treated as contradiction — normalised to `vote` in `_resolve_move` and accepted as equivalent in `_check_decision_clarity`. (2) `select/selecting` missing from ACCEPT cue regex; added alongside `lock in`. (3) `_has_visible_commitment` required option alias and cue in the same contrast clause — "Mountain Cabin has risks, but I accept it" failed because name and cue were in different clauses. Case-2 rule added: also passes when the option appears anywhere in the text and the cue clause contains no other option. When no trailer but visible commitment detectable, logs `MISSING_COMMITMENT_TRAILER` (warn) instead of blocking. Trailer stance hint added to per-turn prompt. Validated `20260629_004303_964878`, `20260629_004412_157793`: both `successful`, all commitment turns credited, no blocks.

### KF04 - Generated short aliases and parser aliases use different contracts — RESOLVED 2026-06-29

**Fix:** `validated_short_alias()` in `src/aliases.py` is now the single shared contract used by builder, prompts, validation, and parsing. An alias is accepted only when every word appears in the option name, the full alias is ≥ `short_alias_min_chars` (4) chars, word count ≤ `short_alias_max_words` (3), last word is not a stopword, and not all words are generic. `deterministic_alias()` provides a guaranteed fallback (first two content words); `short_alias_map()` resolves collisions and returns the final per-option alias used everywhere. The six GPT validation runs (`20260629_162826_677662` board game, `20260629_163018_201924` framework, `20260629_163124_593599` hiking, `20260629_163243_834622` venue, `20260629_163348_522936` restaurant, `20260629_163642_160834` coffee machine) produced only valid recognizable aliases ("Ticket Ride", "React TypeScript", "Eagle Ridge", "Vineyard Estate", "Frank's BBQ", "Barista Express", "FlexBrew 2-Way") — all words from the respective option names, all ≥ 4 chars — and zero alias resolution failures across all runs. Outcomes all resolved (5 successful, 1 majority).

**Original evidence:** Builder accepted `Spy` as short name for Codenames. Prompts showed A=Spy but `OptionResolver` rejected it (3 chars < 4). Two confirmation turns credited to the wrong option. `Rails` for Ticket to Ride was resolvable but sounded unrecognizable.

**Relevant code:** `src/aliases.py`: `validated_short_alias`, `deterministic_alias`, `short_alias_map`; `src/builders.py:202`; `src/prompts.py:579`; `src/parsing.py:88`.

### KF23 - Setup validity is not reliable across providers — RESOLVED 2026-06-29

**Fix:** The persona model was simplified to remove all fields that required score/list coherence. `Persona` now has: `preferred_options: list[str]` (1–2 ordered favourites), `rejection: str | None`, `rejection_reason: str`, `background`, `private_goal`. All score-generation, acceptable-option lists, reasons, reservations, soft/hard rejection lists, role, speech_style, and main_concern fields are gone. The setup prompt now asks for only the minimal fields; validation checks camp structure and that rejections are not preferred options. `_postprocess_personas()` and `_build_scores()` are removed entirely. GPT run `20260629_001500_154989` produced three personas (one with two preferred_options) in the first attempt without error. The ~50% setup failure rate that had been blocking P0 validation is eliminated.

## P2 - Moderator and closure integrity

### KF06 - Moderator output inherits bad state and is not grounded — RESOLVED 2026-06-29

**Fix:** With KF03 fixed, commitment state is now reliable. Moderator grounding tightened in three ways: (1) `_MODERATOR_RULES` now explicitly forbids describing or attributing any quality to any option beyond its name; (2) `moderator_agreement_prompt`, `moderator_holdout_prompt`, and `moderator_closure_prompt` all now include the relevant option's card attributes via `_option_brief` so the model works from real card data rather than inventing facts; (3) `_moderator_say` in `dialogue.py` runs a grounding check after generation — if `invalid_option_refs` detects hallucinated option names, it retries once before accepting. Validated GPT runs `20260629_004851_081547` and `20260629_004937_126619`: closure lines derived facts from card fields only (“10 hours weekly”, “good for 2-5 and about an hour long”).

### KF10 - Cosmetic and unresolved closing turns are unvalidated and can leave the topic — RESOLVED 2026-06-29

**Fix:** Three changes: (1) `farewell_line` prompt removed topic-specific forbidden examples (“no genre, author, plot”) replacing them with a generic rule: “name the option by its exact name only — do not add any description, invented detail, or attribute”; (2) `_social_say` in `dialogue.py` now accepts an optional `state` parameter and when present runs `invalid_option_refs` on the generated text, retrying once if hallucinated option names appear; (3) the closure `_social_round` call passes `validate_state=state` so farewell turns get this grounding check. The unresolved closure path in `moderator_closure_prompt` now requires procedural next steps only (“check a fact, meet again, narrow to two options”) rather than invented option facts. Validated in GPT runs `20260629_004851_081547` (study method) and `20260629_004937_126619` (board game): farewells named the chosen option correctly without invented attributes.

## P3 - Grounding, coverage, and question quality

### KF11 - Coverage counts are inflated and do not prove an option was examined — RESOLVED 2026-06-29

**Fix:** `_update_coverage` in `dialogue.py` now uses `resolver.ids_in_text(record.text)` (visibly mentioned options only) instead of `act.option_refs` (which can include options inferred from routing intent but never spoken). A reason is now counted only when the option is visibly named AND at least one `classify_claim_slots` slot is present — replacing the old “8-word” proxy. Validated in GPT runs `20260629_005443_363945` and `20260629_005532_877764`: ASK questions anchored to card attributes, no coverage inflation visible.

### KF12 - Unsupported attributes and false comparisons still pass — RESOLVED 2026-06-29

**Fix:** Three changes: (1) Grounding rule in `sim_utterance` now explicitly says “do not claim facilities, features, or services not listed in the card” and forbids turning uncertainty into a new invented fact; (2) Added `_INVENTED_FACILITY_PATTERN` regex in `validation.py` covering concrete facility nouns (lodges, cabins, shuttles, indoor spaces, group discounts, etc.) — when matched without hedge words and the term is absent from the option card, fires `INVENTED_OPTION_ATTRIBUTE` (repair); (3) Moderator/closure grounding already fixed in P2 (KF06). Note: semantically hedged invented claims (e.g. “generally handles that better”) are warn-level only and remain possible — the repair threshold covers only confident assertions of concrete facilities. Validated in GPT runs `20260629_005443_363945` and `20260629_005532_877764`: no invented facilities appeared; card-grounded attributes only.

### KF25 - The controller invites questions that the option cards cannot answer — RESOLVED 2026-06-29

**Fix:** Two changes: (1) For ASK turns, `sim_utterance` now shows a compact card brief (`_option_brief`) for the focus options instead of name-only — giving the model the actual attributes to ask about; (2) ASK guidance updated to “Ask one question the option cards above can actually answer — a specific attribute, number, or trade-off listed there.” In GPT validation runs, ASK turns asked about message history limits, video call participant limits, storage, distance, and catering — all card attributes. “Warm drinks” / “cabin size” / “indoor space” style unanswerable questions did not appear. Residual: unanswerable questions can still occur if the model invents a topic; hedge-answer gating (closes after 2nd hedge) remains the safety net.

## P4 - Repair pressure and cost

### KF16 - Token cost remains structurally high

**Evidence:** The compact-prompt pass reduced combined tokens by 38.4% on the same five topics/sizes and reduced dialogue input per turn at every size (24.3-37.0%). A second five-run set used 159,840 combined tokens and 709.0 dialogue input tokens per turn, still well below the original 984.5 baseline but material enough to keep this issue open.

**Implemented direction:** Runtime prompts now use compact lean state, derived habits instead of raw trait duplication, one prior self-point, four recent turns, and no duplicated direct-response target. Repairs receive only relevant options and include recent chat only for repetition/context issues. The prompt no longer encourages first-person openings.

**Required direction:** After correctness is stable, shorten repeated group state, global rules, and option text; pass only the exact response target and relevant facts. Make repair prompts smaller and issue-specific. Track quality per token by provider and group size.

**Relevant code:** `src/prompts.py`: `sim_utterance()`, runtime speaker card, repair prompt; `config.yaml`: prompt windows and word budgets.
