# Known Failures - Open Backlog Only

Last updated: 2026-06-30.

Scope: this file tracks only open failures still supported by the latest supplied validation logs and the latest `known_failures(6).md`. Resolved implementation history was removed. The backlog is ordered by priority, not by discovery order.

Audit basis: KI01 was validated on GPT across n=2-7 with zero final unclear commitments. Grouped KI02/KI03 validation used controlled GPT runs at n=2, n=3, n=5, and n=7 (`20260630_182438_047519` through `20260630_184518_521015`): three successful outcomes, one majority, and one unresolved outcome across 160 participant turns. Visible support reproduced every status; actual holdout interventions targeted only participants who had visibly backed another option, and the missing-commitment path has deterministic coverage. Other backlog evidence still comes from the supplied 13-run corpus in `logs - Copy.zip`.

Core rule: outcome support must be based on visible transcript text only. A participant counts as supporting an option only when their generated message contains a clear vote or acceptance for that option. Hidden `acceptable_options`, hidden scores, routed intent, or unverified trailers must not count as support by themselves.

---

## Validation protocol

1. Fix one issue at a time unless issues are explicitly grouped.
2. Validate on mixed topics and group sizes, especially n=2, n=3, n=5, and n=7.
3. Read transcripts manually. Metrics and validator counts are not sufficient.
4. Check outcome correctness from visible dialogue, not only `run.json` state.
5. Keep fixes topic-agnostic and provider-agnostic. Avoid adding phrase patches that only solve one observed transcript.
6. Close an issue only when the visible dialogue improves without regressions in grounding, pacing, moderator behavior, or outcome integrity.

---

## P1 - Moderator and narrowing naturalness

### KI04 - Lone-holdout intervention should be direct and specific

**Problem:** When all or nearly all participants support one option except one person, the moderator still uses generic phrasing such as “what’s holding you back from X, or is there another option everyone could get behind?” This is functional but repetitive and not socially precise.

**Required direction:** If there is one real holdout, address only that person. The moderator should ask:
- what specific concern remains;
- whether a named condition would make the candidate acceptable;
- or whether the holdout has a concrete alternative that could gain support.

Do not ask the whole group when the blocker is one person. Do not use the same narrowing template every time.

**Acceptance criteria:** Lone-holdout interventions feel like a natural facilitator addressing the current social state, not a repeated stock transition.

---

### KI05 - Narrowing is still too moderator-led

**Problem:** Narrowing mostly happens through moderator prompts. Participants rarely initiate natural convergence themselves, even when the dialogue clearly shows a leading option or exhausted alternatives.

**Required direction:** Allow participant-led narrowing acts. A participant should sometimes say things like:
- “It sounds like we’re mostly between A and B now.”
- “If X solves the timing issue, I could move there.”
- “I think Y is becoming the realistic choice.”

This should be routed based on state, not just phrased as another support turn.

**Acceptance criteria:** Some runs narrow through participant initiative before the moderator forces the phase transition.

---

### KI06 - Moderator intervention phrasing is repetitive

**Problem:** Even when the moderator targets roughly the right issue, the phrasing repeats the same structure: “what’s holding you back from X, or is there another option…”. This makes the moderator sound mechanical.

**Required direction:** Add state-conditioned moderator forms rather than more random paraphrases:
- lone holdout: ask one blocker question;
- two camps: ask a bridge/comparison question;
- missing vote: ask for explicit confirmation;
- exhausted discussion: summarize the trade-off and move to a decision;
- unresolved: name the blocker and next action.

**Acceptance criteria:** Moderator turns vary because the underlying social situation differs, not because of cosmetic wording randomization.

---

## P2 - Natural conversation flow

### KI07 - Social beat content still sounds procedural

**Problem:** At-most-one greeting/farewell is an improvement, but the remaining social beat often sounds like a moderator line: “Let’s quickly align…”, “Let’s finalize this quickly…”, “Let’s quickly weigh our options…”. This is not a natural participant greeting.

**Required direction:** Keep optional single-speaker social beats, but make them genuinely social or remove them. A participant opening should sound like a light entry into an ongoing chat, not a process command.

**Acceptance criteria:** Social beats no longer duplicate moderator function and no longer contain repeated “quickly align/finalize/weigh” templates.

---

### KI08 - Local flow still overuses standalone argument-card turns

**Problem:** Many participant turns still read as independent mini-arguments rather than situated replies. The latest logs improved response targeting, but dialogues still often alternate preference statements instead of building on prior turns with short uptake, partial agreement, clarification, or direct answer.

**Required direction:** Improve local turn acts before adding more global prompt rules. Encourage:
- direct answer to the previous speaker;
- short backchannels when appropriate;
- explicit uptake of a named concern;
- brief concessions before new arguments;
- fewer self-contained option pitches after the opening phase.

**Acceptance criteria:** A reader can follow a local thread across several turns, rather than seeing a sequence of isolated claims.

---

### KI09 - Trait effects are improved but need more measurable behavioral dimensions

**Problem:** Response length now appears meaningfully trait-sensitive, but trait expression should not reduce to word count. For evaluation, traits need observable behavioral effects across multiple dimensions.

**Required direction:** Preserve the current response-length effect and make other trait effects measurable:
- extraversion: initiative, directness, willingness to address others;
- agreeableness: concession frequency and face-work;
- conscientiousness: practical constraints and detail checks;
- neuroticism: risk sensitivity and blocker persistence;
- openness: novelty/creativity arguments.

**Acceptance criteria:** Trait effects can be measured from transcripts without reading hidden persona fields.

---

### KI10 - Semantic legitimacy of mind changes is still weak

**Problem:** Participants sometimes concede because the phase requires convergence, not because their earlier concern was visibly addressed. The language may be acceptable, but the social movement can feel unearned.

**Required direction:** Before a participant changes position, the dialogue should contain at least one visible answer, trade-off acceptance, or condition that addresses their earlier blocker. If not, route a question or condition-setting turn instead of immediate concession.

**Acceptance criteria:** Concessions read as motivated by prior dialogue, not by controller pressure.

---

## P3 - Grounding and scenario control

### KI11 - Soft qualitative hallucinations still appear

**Problem:** Numeric and hard logistical hallucinations improved, but soft unsupported details still leak through. Examples include recipes being sent after a cooking class, ingredient sourcing, rain fallback behavior, or implied facilities that are not in the card.

**Required direction:** Keep numeric grounding checks, but add a lightweight guard for common unsupported qualitative logistics: availability, included extras, service behavior, weather fallback, staff behavior, room/space features, and post-event materials.

**Acceptance criteria:** Participants hedge or ask about unsupported qualitative details instead of asserting them as facts.

---

### KI12 - User-specified option lists are not always preserved

**Problem:** When the user topic explicitly names candidate options, scenario generation may still introduce a new option. For example, a topic asking about ramen, sandwich place, or falafel cart should not silently add a fourth unrelated restaurant type unless the system is explicitly allowed to propose alternatives.

**Required direction:** Detect explicit option-list topics and preserve the listed options. If the system needs four options for its internal format, either ask/generate only when allowed or mark the added option as an explicit “other suggestion” rather than pretending it was part of the user’s original choice set.

**Acceptance criteria:** The scenario does not change the user’s decision frame without making that change explicit.

---

### KI13 - Option short-name / alias quality remains inconsistent

**Problem:** Some latest runs contain empty `short_name` values for one or more options. This can weaken natural references and visible commitment parsing.

**Required direction:** Ensure every option has a safe short name or intentionally disable alias use for that option. Avoid unsafe generic one-word aliases.

**Acceptance criteria:** Short names are always present when aliasing is enabled, and visible option references resolve consistently.

---

### KI14 - Setup feasibility still needs semantic checks for option constraints

**Problem:** Earlier setup fixes improved topic-count mismatch and broken option names, but semantic feasibility remains only partially checked. Example class: assigning a two- or four-player game to a six-person family decision, or generating options that violate hard shared constraints.

**Required direction:** Add limited deterministic checks for common structured attributes: number of players, group size, budget, time, duration, and capacity. Do not attempt broad world knowledge validation.

**Acceptance criteria:** Obvious card/shared-context contradictions are rejected or regenerated before dialogue starts.

---

## P4 - Repetition, repair pressure, and metrics

### KI15 - Repetition checks miss semantic loops

**Problem:** Exact string repetition is better controlled, but semantic loops remain: participants repeat the same underlying claim in different wording, especially during narrowing and majority formation.

**Required direction:** Track recent `(option, claim-slot, polarity)` patterns and suppress repeated semantic moves when they do not add a new concern, condition, or concession.

**Acceptance criteria:** Repetition decreases without making participants unnaturally terse or preventing legitimate re-emphasis during voting.

---


### KI16 - Prompt complexity can worsen dialogue naturalness

**Problem:** The prompt stack has accumulated many global rules, bans, guidance snippets, validation hints, grounding instructions, trait instructions, and repair-specific constraints. Some of this is necessary and already protects working behavior, but too much instruction density can make the model produce procedural, over-controlled dialogue instead of natural local conversation. It can also increase generic compliance phrases, moderator-like participant turns, and repair pressure.

**Required direction:** Reduce prompt complexity carefully without deleting working controller logic. Preserve the deterministic state machine, visible commitment rules, grounding constraints, trait behavior, and existing successful validations. Simplify by moving stable checks into deterministic code where possible, passing only act-relevant guidance, and removing duplicated or low-value prompt rules. Avoid large prompt rewrites that reintroduce older failures.

**Acceptance criteria:** Prompts become shorter and more act-specific, while transcripts become more natural or at least do not regress. No regression in visible-support outcome correctness, grounding, trait visibility, setup coherence, or moderator targeting.

---

### KI17 - Repair cost remains high

**Problem:** The latest 13 runs contain 143 repaired turns over 500 participant turns. Repairs are still a major cost driver and often cluster around vote clarity and grounding.

**Required direction:** Fix the dominant causes first (`UNCLEAR_VOTE`, then grounding/qualitative claims), then reduce repair prompt size and make repair prompts issue-specific.

**Acceptance criteria:** Repaired-turn rate falls materially without increasing invalid commitments, hallucinations, or unresolved outcomes.

---

### KI18 - Token cost remains structurally high

**Problem:** Latest dialogue input averages about 781 tokens per participant decision turn. Large and repair-heavy runs remain expensive.

**Required direction:** Do not optimize away context before P0-P3 stabilize. Then shorten repeated global rules, pass only act-relevant option facts, compact group state, and accumulate quality-per-token metrics by group size and provider.

**Acceptance criteria:** Input tokens per decision turn fall materially without regressions in grounding, local responsiveness, persona visibility, commitment integrity, or setup reliability.

---

### KI19 - Metrics should separate decision turns, moderator turns, social beats, and repairs

**Problem:** Some quality metrics become hard to interpret when social beats, repaired generations, final closure, and decision turns are mixed. This matters more now because social beats are optional and repairs are frequent.

**Required direction:** Report separate counts for:
- decision participant turns;
- social participant turns;
- moderator turns;
- generated attempts including repairs;
- final accepted turns only.

**Acceptance criteria:** Metrics explain the transcript rather than hiding where cost and quality failures occur.
