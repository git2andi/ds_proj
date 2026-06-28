# Known Failures - Open Issues Only

Last updated: 2026-06-28.

This backlog contains only issues still supported by the current code or by the five GPT validation runs listed below. Historical IDs are retained where an old issue remains partially open; resolved issues are omitted. Fixes must be topic-agnostic, valid for 2-7 participants, and independent of provider-specific wording habits.

## Evidence baseline

Provider: `gpt` / `gpt-4.1-mini`.

| Run | n | Topic | Outcome | Repair rate | Flagged turns | First-person openers | Responsive turns |
|---|---:|---|---|---:|---:|---:|---:|
| `20260628_174224_588230` | 2 | Hiking trail | successful | 7.7% | 7/13 | 0% | 38% |
| `20260628_174308_723317` | 3 | Biology presentation topic | majority | 0.0% | 13/26 | 0% | 19% |
| `20260628_174440_130656` | 4 | Next-sprint feature | majority | 17.8% | 15/45 | 2% | 31% |
| `20260628_174629_106905` | 5 | Shared-kitchen coffee machine | unresolved | 23.6% | 25/55 | 2% | 36% |
| `20260628_174836_463880` | 7 | Charity gala theme | successful | 21.1% | 15/38 | 3% | 26% |

Across the five runs: 177 decision turns, 139,572 input tokens, 20,268 output tokens, and 159,840 combined tokens. Dialogue input averaged 709.0 tokens per decision turn, still substantially below the original 984.5 baseline. First-person openers remained at 0-3%. Ten turns in three runs were logged with `state_mutation_blocked`: nine had residual invented attributes and one had an unclear vote. None altered semantic state.

## Validation protocol

For each fix:

1. Work on one issue only.
2. Fix the controller/parser/state contract, not a phrase pattern emitted by one provider.
3. Use the provider explicitly authorized for the task. Never silently substitute an endpoint.
4. Run at least one `n=3` discussion, then a spread across relevant sizes and topics. Portability claims require the same behavioral checks on more than one available provider.
5. Read transcripts and `run.json`; metrics alone are insufficient.
6. Close the issue only when visible behavior improves without an obvious regression.
7. Before completing the upgrade, audit and synchronize every applicable active information source: `AGENTS.md`, `CLAUDE.md`, both repository skill copies, active memory/index files, this backlog, `README.md`, and other affected workflow docs. Historical per-fix records remain historical.
8. Stop at the completed upgrade boundary unless the user explicitly grouped issues or requested automatic continuation.

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

**Required direction:** Simplify the per-turn contract around one local conversational job, make persona behavior and response-length limits explicit but compact, and ensure direct-response context is exact. Prefer router/state and prompt-structure changes over provider-specific phrase detection. Do not add forced backchannels, scripted turns, stereotypes, quoted examples, or a larger regex blacklist. Use manual transcript review for the behavioral acceptance criteria.

**Relevant code:** `src/prompts.py`: runtime speaker card, utterance guidance, and word budgets; `src/router.py`: response targeting and contribution selection; `src/dialogue.py`: question/challenge context and progress tracking; `src/validation.py`: repetition/discourse diagnostics; `src/builders.py`: persona-generation contract; `config.yaml`: response-length and context dials.

**Implementation checkpoint (not yet closed):** Runtime prompts now lead with one local job, include exact response context only when routing identifies a real target, and omit generated role/style labels that could induce stereotypes or formal register. Persona cues describe observable behaviors from traits. Focus-matching recent turns can be selected without changing the routed speaker or act. Response budgets are monotonic by trait and capped at 48 words; long turns are limited to two short sentences. Decision prompts and repairs distinguish selecting an option now from merely praising it. The visible-commitment parser now recognizes the provider-neutral `select` verb family after live evidence showed that the prompt/parser contracts disagreed.

**Checkpoint evidence:** Completed GPT runs `20260628_214821_616924`, `20260628_215131_265947`, and `20260628_215601_400208` were read in full. They confirmed clear relative length differences and more exact local targeting, but also exposed remaining formal turns and decision-loop churn during iteration. The last run produced many plainly visible selections that were rejected only because `select` was absent from the commitment cue set; the parser was corrected, but no post-fix dialogue completed because KF23 repeatedly exhausted setup retries. P0 therefore remains open, and the required final `n=3` plus `n=2-7` spread is still pending.

## P1 - State, outcome, and run integrity

### KF03 - Visible commitment and machine trailer can disagree

**Implemented:** Binding VOTE/ACCEPT validation now requires a clause-local visible commitment to the same option named by the trailer. Wrong-target, pronoun-only, and descriptive non-commitments trigger repair; residual failures are logged with semantic state blocked. The shared cue contract includes `select` after `20260628_215601_400208` showed repeated visible selections being rejected. Targeted GPT run `20260628_175843_081308` produced three matching visible votes and a correct unanimous outcome. Keep open until the deferred broad evaluation.

**Newest evidence:** In `20260628_173039_600135`, Yara's confirmation line named Eagle and Bear but runtime state credited an acceptance of Willow. In `20260628_173250_522106`, Sami's vote was credited to the Advanced Reporting Dashboard even though the visible sentence did not name it.

**New evidence:** In `20260628_171329_538934`, Marco was routed to ACCEPT four times. He visibly said he would stick with Garden and later that he was okay with Garden, but none of those turns populated `act.accepts`; the run therefore reported a 6/7 majority instead of visible unanimity.

**Evidence:** In `20260628_164120_438049`, Wren said “Codenames gets my vote” but the parsed vote was Option C (Terraforming Mars), taken from the trailer. In `20260628_163822_610518`, Yuki accepted Option A with “I’m cool letting go of picnic’s casual feel for this,” which does not visibly name the accepted option. Several other binding lines mention both the rejected and selected options without deterministic reconciliation.

**Root cause:** `_resolve_move()` prioritizes `move.option`. `_check_decision_clarity()` checks trailer stance against routed intent but does not require the visible sentence to resolve to the same option or contain an unhedged commitment. After the repair budget is exhausted, the turn is still applied to runtime state.

**Required direction:** Treat the trailer as metadata, not sole evidence. For VOTE/ACCEPT, require visible, unhedged commitment to one resolved option; reject or repair text/trailer mismatches. If the final attempt is still invalid, log the utterance but do not mutate binding state.

**Relevant code:** `src/parsing.py`: `_resolve_move()`; `src/validation.py`: `_check_decision_clarity()`; `src/dialogue.py`: `_generate_turn()`, `StateTracker._update_runtime()`.

### KF04 - Generated short aliases and parser aliases use different contracts

**Implemented:** Setup, prompt rendering, validation, cleanup, and parsing now share one alias contract. Aliases must be recognizable words from the option name, meet configured size limits, and be unique across the option set; collisions use deterministic unique fallbacks. Targeted GPT run `20260628_180743_359852` exposed four aliases and the parser resolved all four exactly. Keep open until deferred broad evaluation.

**Evidence:** The board-game setup generated `Spy` as the short name for Codenames. Prompts repeatedly encouraged participants to say `Spy`, but `OptionResolver` rejects short names shorter than four characters. Two confirmation turns therefore visibly used an alias the state parser could not resolve. `Rails` for Ticket to Ride was resolvable but sounded less recognizable than the real title.

**Root cause:** The builder accepts one-to-three-word short names without a minimum length or relationship to the option name; prompt generation uses every accepted short name; parsing applies a separate length filter.

**Required direction:** Define one shared alias-validation function used by builder, prompts, validation, and parser. Prefer recognizable substrings or deterministic reductions of the option name, reject generic/colliding aliases, and never prompt an alias the resolver will ignore.

**Relevant code:** `src/builders.py`: `_clean_short_name()`; `src/prompts.py`: `_short_alias()`, `_alias_rule()`; `src/parsing.py`: `OptionResolver._build_aliases()`.

### KF23 - Setup validity is not reliable across providers

**Implemented:** Scenario setup now receives the exact participant count and rejects conflicting decision-group references. Persona scores/lists are validated rather than clamped or rewritten. A controller-selected rotating option is prompted and validated as common ground for all non-stubborn participants. The persona schema's contradictory threshold example was corrected. Targeted GPT run `20260628_181856_135923` produced three count-consistent personas, consistent scores/lists, shared non-stubborn compromise options, and trait-driven stubbornness. Keep open until deferred broad setup-rate evaluation.

**Newest evidence / validation blocker:** During the P0 checkpoint, six GPT CLI attempts exhausted both setup retries with a generated score/list contradiction (`book swap`, `Sunday dessert`, `biology presentation`, `hiking trail`, `coffee machine`, and `client lunch`). Three other attempts completed setup. This failure rate now blocks mandatory behavioral spreads, including post-fix verification of P0. KF23 should be the next implementation boundary before resuming broad naturalness validation.

**New evidence:** In `20260628_170634_413335`, the n=2 hiking scenario described a group of five friends. In `20260628_170750_982638`, the n=3 biology scenario described four students.

**Evidence:** GPT exhausted both setup attempts more than once because same-camp participants chose conflicting preferred options. The completed weekend-trip scenario also described a “Group of 5 friends” while the simulation contained four personas. Generated persona data can be internally questionable, such as a vegetarian-focused persona accepting pulled-pork sliders.

**Root cause:** A large prose-to-JSON setup call must satisfy scenario facts, participant count, traits, coalition structure, acceptability, scores, and semantic persona consistency simultaneously. Validation covers structural fields and coalition equality but cannot verify most semantic contradictions.

**Required direction:** Keep preference camps prompt-driven and validation-based, but make the setup contract provider-neutral and easier to satisfy. Validate participant-count references and deterministic score/list consistency. Add semantic checks only where grounded in explicit option attributes; otherwise retry rather than rewrite persona preferences. Measure setup failure rate by provider.

**Relevant code:** `src/builders.py`: `build()`, `_validate_preference_plan()`, `_postprocess_personas()`, `_validate_world()`; `src/prompts.py`: scenario/persona setup prompts.

## P2 - Moderator and closure integrity

### KF06 - Moderator output inherits bad state and is not grounded

**Evidence:** In the board-game run, the moderator treated Wren as a Codenames holdout because Wren’s visible Codenames vote had been recorded as Terraforming Mars. In the weekend-trip run, the moderator asserted that Indiana Dunes had “cozy lodges,” a fact absent from the option card.

**Required direction:** Fix commitment state first, then build moderator prompts from validated visible support. Validate moderator option claims against cards, and use deterministic state text for supporter/holdout counts rather than asking the model to infer them.

**Relevant code:** `src/dialogue.py`: `_moderator_intervention()`; `src/prompts.py`: `moderator_holdout_prompt()`, `moderator_agreement_prompt()`.

### KF10 - Cosmetic and unresolved closing turns are unvalidated and can leave the topic

**Evidence:** The unresolved study-method run ended with farewells about “The Martian,” “the chosen book,” “snacks,” and “locking in the book.” These are unrelated to the exam topic. The farewell prompt itself lists book-specific forbidden examples, which can prime a provider to emit them. Social turns bypass normal validation.

**Required direction:** Remove topic-specific negative examples from generic prompts. Validate social and moderator closure text for topic/option leakage, or use a compact deterministic closure skeleton with model-rendered tone only. An unresolved close must name the real blocker and a grounded next action.

**Relevant code:** `src/prompts.py`: `farewell_line()`, `moderator_closure_prompt()`; `src/dialogue.py`: `_social_say()`, `_social_round()`.

## P3 - Grounding, coverage, and question quality

### KF11 - Coverage counts are inflated and do not prove an option was examined

**Evidence:** Coverage often reported nearly every mention as a reason (for example, 41 mentions and 41 reasons for study Option A). `_looks_like_reason()` classifies any turn of at least eight words as a reason, and parser-injected intent focus can enter `option_refs` even when the option is not visibly named. Therefore the narrowing gate and metrics can claim coverage without a grounded reason or objection.

**Required direction:** Count visible references separately from routed focus. Count a reason only when a claim slot or explicit card-grounded trade-off is attached to that option. Require the leading option and serious alternatives to have visible support and challenge before natural narrowing.

**Relevant code:** `src/parsing.py`: `parse_dialogue_act()`; `src/dialogue.py`: `_update_coverage()`, `_looks_like_reason()`, `_can_start_narrowing()`; `src/router.py`: `_coverage_gap_option()`.

### KF12 - Unsupported attributes and false comparisons still pass

**Evidence:** Examples include curry being easy to reheat/travel with, shells costing less than a cheaper curry, caterers using insulated boxes, unlisted indoor spaces, “cozy lodges” at Indiana Dunes, lodging availability, parking/access assumptions, and changing a 90-minute quiz option to 15 minutes. Several survived a repair and were used by the moderator or decision state.

**Required direction:** Validate claims against option-card keys and values, not only numbers or a small soft-attribute denylist. Allow explicit uncertainty, but do not let uncertainty introduce a new positive or negative fact. Apply the same grounding contract to participant, moderator, closure, and repair generations.

**Relevant code:** `src/validation.py`: grounding checks; `src/prompts.py`: generation/repair grounding rules; `src/dialogue.py`: moderator and social generation.

### KF25 - The controller invites questions that the option cards cannot answer

**Evidence:** Runs asked about warm drinks, cabin size, indoor spaces, shuttles, public transit, group discounts, and whether game clues were memorized. Routed answers often guessed or repeated “not in the cards” before adding another unsupported claim, creating artificial Q&A loops.

**Required direction:** Route ASK toward comparisons answerable from known option attributes or participant priorities. Mark fact questions with no card source as unanswerable and close them after one grounded response. Repair must not transform an unsupported assertion into another unanswerable question.

**Relevant code:** `src/router.py`: ASK selection; `src/parsing.py`: question targeting; `src/dialogue.py`: open-question state; `src/prompts.py`: ASK, ANSWER, and repair guidance.

## P4 - Repair pressure and cost

### KF16 - Token cost remains structurally high

**Evidence:** The compact-prompt pass reduced combined tokens by 38.4% on the same five topics/sizes and reduced dialogue input per turn at every size (24.3-37.0%). A second five-run set used 159,840 combined tokens and 709.0 dialogue input tokens per turn, still well below the original 984.5 baseline but material enough to keep this issue open.

**Implemented direction:** Runtime prompts now use compact lean state, derived habits instead of raw trait duplication, one prior self-point, four recent turns, and no duplicated direct-response target. Repairs receive only relevant options and include recent chat only for repetition/context issues. The prompt no longer encourages first-person openings.

**Required direction:** After correctness is stable, shorten repeated group state, global rules, and option text; pass only the exact response target and relevant facts. Make repair prompts smaller and issue-specific. Track quality per token by provider and group size.

**Relevant code:** `src/prompts.py`: `sim_utterance()`, runtime speaker card, repair prompt; `config.yaml`: prompt windows and word budgets.
