# MASTERPLAN v2 — From a decision pipeline to a real deliberation

**Status:** the structural plumbing works (option aliasing → stances → consensus → honest
outcomes, all scaling 2–5 sims). What does **not** work is the thing the project exists
for: the transcripts do not read like a group of humans actually discussing. This document
diagnoses why, grounds the fix in current research, and lays out a staged plan.

This supersedes the previous MASTERPLAN, whose work (simplification, the prose↔state
bridge, fail-fast, the phase rework) is largely **done**. The remaining problem is
qualitatively different and needs an architectural reframe, not more prompt text.

---

## Part I — The evidence

Two real runs (full-LLM, see `logs/20260528_013648_995747.txt` and `…013322_078489.txt`).

### Exhibit A — broken at the source (biology presentation topic)

Topic: *"find a university presentation topic in the course about biology."*
Generated options: **"The Helix Institute", "EcoCycle Conference", "The Biosphere Summit",
"The Cellarius Symposium"** — i.e. conferences/venues, not presentation topics. The decision
is incoherent before a single sim speaks. The sims then argue about abstract qualities
("breadth vs depth", "feasibility") that aren't attached to any option, and the moderator
force-closes on Option A that nobody argued for.

### Exhibit B — "okay" but hollow (lunch spot)

Reads more naturally, but every sim is a **one-note concern repeater**:
- Ava → "accommodates various dietary preferences" (her `key_concern`, verbatim, repeatedly)
- Liam → "familiar sandwich options" (asks "does X have sandwiches?" three times)
- Julian → "filling and affordable"

Nobody is *moved* by an argument. They converge only because every persona's `acceptable`
list contains almost every option, so agreement is automatic. The discussion is filler.

### The six concrete failure modes

1. **Options don't fit the decision.** `option_generation` forces a proper-noun venue onto
   every topic. Abstract decisions (a topic, a strategy, a policy) get nonsensical options.
2. **Personas hold positions, not arguments.** The belief model
   (`preferred/acceptable/rejected/key_concern/concession`) encodes *where* a sim stands but
   not *why*, what *evidence/experience* backs it, or what would *change their mind*. Rich
   backstories never become argumentative ammunition.
3. **No genuine disagreement.** `acceptable ≈ 3 of 4` (cooperative baseline) means nobody
   really objects, so there is nothing to deliberate and "narrowing comes too soon."
4. **The moderator shuts discussion down.** `detect_info_gap` → "isn't specified, decide on
   what's listed" fires on *rhetorical/deliberative* questions, killing the substance.
5. **Grounding sterilizes.** `fact_check` forbids any detail not in the option text, so sims
   can't reason from general knowledge or their own experience — the raw material of a real
   argument.
6. **Pacing is content-blind.** Phase transitions are pure turn counts; the system never asks
   "did a real disagreement actually get surfaced and worked through?"

### The root cause

**The architecture optimizes for *terminating on an option*, not for *having a discussion*.**
Consensus engine, narrowing, confirmation, force-close — all machinery for ending. The
deliberation is treated as filler to be rushed past. To get real chats, deliberation must
become the *product*; the decision is its *byproduct*.

---

## Part II — What a real discussion needs (research grounding)

- **Generative Agents (Park et al., 2023).** Believability comes from *memory + reflection +
  reactivity*: agents remember what was said, form higher-level takes, and respond to each
  other. Our sims see only the last ~6 raw turns and never reflect, so they loop and repeat.
- **Multi-agent debate (Du et al., 2023; ChatEval; AgentVerse).** Quality emerges from genuine
  *critique and rebuttal* between agents, not from each restating its view. Our act mix is
  dominated by self-assertion; there is almost no challenge→response→update cycle.
- **Argumentation theory (Toulmin).** A real argumentative move is *claim + warrant (reason) +
  optional evidence/backing*. Our turns are claims only ("I prefer C"). No warrants, no backing.
- **Negotiation / persuasion agents (e.g. CICERO, Diplomacy).** Persuasion requires *intents*
  and *theory-of-mind*: modelling what others want and appealing to it. Our sims never reason
  about each other's priorities; they broadcast their own.
- **Deliberative-quality research (justification, reciprocity, reflexivity).** A good
  deliberation has participants *justifying* claims and *referencing each other's* justifications,
  and being willing to *update*. Our sims do none of the three.
- **Role-play LLMs (Character-LLM, RoleLLM).** Persona fidelity needs traits→behaviour→speech
  grounding and *distinct* voices. Our personas are differentiated on paper (Big Five, backstory)
  but speak identically because nothing routes those differences into argumentative behaviour.

### The principled line on "inventing facts"

The user's open question — *should sims invent missing detail?* — is answered cleanly by the
literature: **inventing reasons and general world-knowledge is argumentation (good); inventing
specific option attributes is hallucination that corrupts the decision (bad).**

- A biology student *should* be able to say "CRISPR has tons of accessible material and it's
  what everyone's excited about" — that's a warrant from their knowledge, not in any option text.
- No sim should say "Option A costs $40" when no price was given — that's a fabricated option fact.

Today's grounding bans **both**. It must ban only the second.

---

## Part III — The reframe

Stop building a pipeline that rushes to a decision. Build a **deliberation** in which:

1. **There is real substance to argue about** — options that fit the decision and carry
   discussable trade-offs.
2. **Sims have something to argue *with*** — reasons, evidence-from-experience, and a stake.
3. **There is genuine disagreement among cooperative sims** — they start in different places
   for good reasons, so reconciling them *requires* exchange. Everyone wants a workable
   outcome and is open to moving; the difference is in starting preference and priority, not in
   willingness to agree. (Outright obstruction is only the rare, by-design hard-blocker.)
4. **The moderator facilitates rather than terminates** — it surfaces the disagreement, asks
   sims to engage each other, and only closes when deliberation is genuinely spent.

The decision then *emerges* from the discussion (Fisher's actual model), instead of the
machinery declaring one and back-filling chatter.

---

## Part IV — Staged plan

Each stage is independently shippable and lists the modules it touches, the acceptance
signal, and the trade-off. Stages are ordered by leverage: nothing downstream matters if the
options (Stage 0) are nonsense.

### Stage 0 — Options that fit the decision and carry substance
**Problem:** Exhibit A. Forced proper-noun venues; abstract topics get incoherent options.
**Change (architectural, two-step generation in `prompts.option_generation` +
`orchestrator._generate_options`):**
1. First classify the decision kind: *concrete pick* (restaurant, flight, book, venue) vs
   *abstract pick* (a topic, approach, strategy, policy, theme).
2. Generate options that *are the thing being chosen*: concrete picks get real proper names;
   abstract picks get descriptive options with **no forced proper noun** (e.g. presentation
   topics → "CRISPR gene-editing ethics", "Coral-reef symbiosis", not "The Biosphere Summit").
3. Each option must carry **2–3 genuinely contestable dimensions** (a real upside, a real
   trade-off, who it suits) so there is something to weigh — but still **no fabricated hard
   numbers**.
**Acceptance:** for 10 varied topics (concrete + abstract), every option is a plausible answer
to the literal decision, and at least one trade-off per option is debatable.
**Trade-off:** one extra reasoning step in setup (still one LLM call if folded into the prompt).

### Stage 1 — Personas with an "argument kit", not just a position
**Problem:** failure mode 2; Exhibit B's one-note repeaters.
**Change (`persona.AgentBeliefs`, `prompts.agent_beliefs_group`, `prompt_context.build_speaker_card`):**
Extend each persona's private model from a bare stance to an **argument kit**:
- `preferred`, and **1–2 concrete reasons** for it drawn from their goal/expertise/backstory
  (Toulmin warrants), phrased as *their* knowledge/experience.
- **1 genuine reservation** about a rival option — framed as a concern they'd want addressed
  ("I'd worry about the longer walk"), not a veto. This gives them something substantive to
  raise and resolve, while staying cooperative.
- `would_reconsider_if` — the concrete thing that would move them (enables genuine update and
  keeps every disagreement resolvable).
This is the same number of LLM calls; it enriches the existing grouped belief call.
**Acceptance:** transcripts show sims giving *reasons* and *experience*, not just "I prefer X";
backstory details surface as arguments.
**Trade-off:** longer belief prompt/output; must stay disciplined to avoid caricature.

### Stage 2 — Divergent starting positions among cooperative sims
**Problem:** failure mode 3. Everyone accepts almost everything, so the group converges
instantly with nothing to discuss.

**Principle (important):** every sim is fundamentally **cooperative** — they want the group
to reach a workable outcome and they are open to being persuaded and to compromise. The fuel
for a real discussion is **not** an objector blocking progress; it is that sensible people
*start in different places for good reasons* and have to talk it through. The discussion
exists to reconcile honest differences in **preference and priority**, not to overcome anyone's
unwillingness to move. Obstruction is **only** ever the rare, by-design hard-blocker
(`hard_blocker_dialogue_probability`, ~5%), and that is the one case where ending in
`force_close` is a legitimate, realistic outcome.

**Change (`persona.PersonaBuilder.assign_beliefs` + belief prompt + a diversity check):**
- **Spread the starting preferences.** The group should not all `prefer` the same option.
  Enforce this deterministically post-generation, exactly the way trait-diversity is already
  enforced: if every persona's `preferred` is identical, nudge one persona toward the option
  that best fits their stated `key_concern`. This creates a reason to deliberate without making
  anyone difficult.
- **Keep `acceptable` overlapping enough that consensus is reachable.** Aim for ~2 acceptable
  options per persona (their `preferred` plus at least one other), with the group's acceptable
  sets overlapping on at least one common option. Divergent *preferences*, shared *fallback* —
  this is what lets a cooperative group start apart and still converge.
- **Frame each persona's reservation as a concern to be addressed, not a veto.** A persona may
  be lukewarm about an option, but the belief model should phrase that as "I'd want X handled
  first" (resolvable via Stage 1's `would_reconsider_if`), never as a refusal. Hard refusal is
  reserved for the sampled hard-blocker alone.

**Acceptance:** dialogues open with genuinely different preferred options and reasons, then
*converge through discussion* in the large majority of runs; `force_close` stays rare and
correlates with the hard-blocker flag, not with ordinary disagreement.
**Trade-off:** if `acceptable` is tightened too far the group can deadlock even though everyone
is willing — so the overlap guarantee above is load-bearing. Resolvability is protected by the
shared fallback option, the `would_reconsider_if` update path (Stage 1), and the facilitator
(Stage 3). Disagreement should make the discussion *necessary*, never make it *unwinnable*.

### Stage 3 — Moderator as facilitator, not terminator
**Problem:** failure modes 4 and 6. The blunt "isn't specified" shutdown; rushed narrowing.
**Change (`moderation.ModerationEngine`, its prompts, `orchestrator` loop):**
- **Remove the blunt info-gap shutdown.** Replace with two distinct behaviours:
  - *Genuine* missing option-attribute → reframe toward judgment, don't end it:
    "we don't have exact figures — which matters more to you here, speed or cost?"
  - *Deliberative* question (the common case) → leave it alone; it's the discussion.
  Distinguish the two by intent, not by the presence of a "?" (the current detector conflates
  them). A question that targets another sim's reason is deliberation, not an info gap.
- **Add facilitation moves** the moderator can choose: surface the live disagreement
  ("Ava wants depth, Liam wants breadth — can one of you address the other's point?"), ask a
  sim to respond to a specific argument, ask "what would change your mind?" (which Stage 1's
  `would_reconsider_if` makes answerable).
- The moderator should be **mostly quiet** while real exchange is happening, and only step in
  to unstick or to move toward a decision once the disagreement is genuinely explored.
**Acceptance:** moderator no longer ends lines of inquiry; its interventions visibly advance
deliberation; transcripts show challenge→response→(update or principled hold).
**Trade-off:** intent classification is harder than keyword matching; start with tight
heuristics over the structured trace (addressee graph + act types) before any LLM classifier.

### Stage 4 — Grounding that allows argument, forbids fabricated option facts
**Problem:** failure mode 5. Sterile turns.
**Change (`reasoning.fact_check`, `simulator._ground_check`, voice rules):**
- Permit reasoning from **general knowledge and the persona's stated experience**.
- Continue to block **invented option attributes**: numbers/prices/dates/features asserted
  *about an option* that aren't in its text. The existing number-detector already does most of
  this; the refinement is to scope it to claims-about-options, not all world-knowledge.
- Voice rule shifts from "only use facts in the option text" to "don't invent specifics about
  the options; you may bring your own knowledge and experience to bear."
**Acceptance:** sims make experience-based arguments; no sim asserts a false option attribute.
**Trade-off:** the line is fuzzier; accept occasional benign world-knowledge to gain real
argument. Keep the deterministic option-fact check strict.

### Stage 5 — Content-aware pacing (deliberation-gated transitions)
**Problem:** failure mode 6. Narrowing on a turn counter.
**Change (`orchestrator._update_phase`/loop, `moderation.should_narrow`):**
Gate narrowing on *deliberation actually having happened*, read from the structured trace we
already build:
- each sim has stated a position **with a reason**, AND
- at least one **challenge→response** exchange has occurred on a contested option, AND
- repetition pressure indicates the live arguments are exhausted.
Turn counts become a **ceiling/floor**, not the trigger. Naturally, easy decisions narrow fast
and contested ones run longer — which answers "different topics → different lengths."
**Acceptance:** dialogue length correlates with how contested the topic is, not a constant.
**Trade-off:** needs reliable act tagging (CHALLENGE/ANSWER/CONCEDE). Today's keyword
estimator is weak here; may need a small, batched LLM act-tagger (one call per few turns) —
the one place an LLM-in-the-loop classifier is worth its cost.

### Stage 6 — Lightweight per-sim memory (anti-repetition, build-on)
**Problem:** Liam repeating his Option-B line ~4× verbatim; general sameness.
**Change (`state.ParticipantState`, `prompt_context`):**
Maintain per sim a compact running memory fed into their prompt instead of raw last-6-turns:
- *points I've already made* (so they don't repeat),
- *open challenges directed at me* (so they must respond),
- *others' key arguments* (so they can build on/rebut).
This is the scaled-down Generative-Agents idea: relevance-filtered memory, not a transcript dump.
**Acceptance:** near-zero verbatim self-repetition; turns reference and build on prior points.
**Trade-off:** more state to maintain; keep it deterministic and small.

---

## Part V — Guardrails (what NOT to do)

- **No more prompt-instruction piling.** Every stage above is a structural/state change; prompt
  edits only follow from them. If a fix is "add another sentence to the sim prompt," it's wrong. Existing prompts might be adapted, unuseful lines removed and some new created to improve the changes done. 
- **Don't force convergence.** Real disagreement should resolve through argument or fail
  honestly (force-close is acceptable when deliberation is genuinely spent — it just shouldn't
  be the *default* because nothing was discussed).
- **Keep the five papers load-bearing.** Fisher (emergence from ratio shifts), SSJ (turn-taking),
  Ouchi/Tsuboi (addressee graph — now also the basis for detecting challenge→response),
  McCrae/John (Big Five → now routed into argument *style*), MUCA (act cooldowns). Stage 5/6
  finally make Fisher and the addressee graph do real work.
- **One LLM-in-the-loop classifier, at most.** Only Stage 5's act-tagger justifies a per-few-turns
  call; everything else stays deterministic.

---

## Part VI — Sequencing and risk

Recommended order: **0 → 1 → 2 → 3 → 4 → 5 → 6.**

- **0–2 are the foundation:** fit options + argument kits + real disagreement. After these, even
  with the current moderator, transcripts should already show genuine exchange. Do these first
  and re-evaluate before touching the moderator.
- **3–4 unblock the discussion** the foundation creates (stop shutting it down; let arguments
  use world-knowledge).
- **5–6 are refinement:** pacing and anti-repetition. Highest implementation risk (act tagging),
  so last, and only if 0–4 haven't already produced good chats.

**Re-evaluation gate:** after Stage 2, run the same handful of topics (one concrete, one
abstract, one contentious) at n=2/3/5 and judge transcripts qualitatively. If discussions feel
real, Stages 5–6 may be unnecessary. Do not implement speculatively past the point where the
chats are good.

**Biggest risk:** Stage 2 (real disagreement) + a still-blunt moderator = more force-closes.
Mitigation: ship Stage 3 close behind Stage 2, and keep the `would_reconsider_if` update path
so disagreements can actually resolve.
