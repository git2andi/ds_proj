# Dialogue Quality Refactor Plan

This plan updates the previous refactor direction after reviewing the latest transcripts.

The earlier plan is **partly fulfilled**: the simulator is now more coherent, has fewer obvious repair artifacts, responds more directly, and produces usable decision outcomes. The next stage is no longer mainly about making the system coherent. It is about making the interaction socially and linguistically believable.

Core direction: generate **situated human interaction**, not polished preference arguments.

The controller should decide what social action the next turn performs. The LLM should only realize that action naturally.

---

## Current diagnosis

The latest transcripts show a clear improvement over the earlier versions:

- turns are usually on-topic,
- speakers often respond to previous points,
- there are fewer visible self-narration artifacts,
- repair rates are low,
- moderator outcomes are more structured,
- some compromise patterns exist.

But the dialogues still do not fully sound human because:

1. many turns are complete mini-arguments instead of situated reactions,
2. speakers repeat full option names and option-card facts,
3. personas differ more by preferred option than by interaction style,
4. disagreement lacks social face-work,
5. preference shifts are sometimes too smooth or abrupt,
6. consensus can be declared while uncertainty remains,
7. the moderator can sound like an external controller,
8. practical constraints are not used strongly enough to drive the discussion,
9. endings are functional but socially thin,
10. small punctuation and malformed-output artifacts still leak through,
11. turn-taking is often too balanced,
12. larger groups need stricter contribution control.

Token usage is acceptable for this project and is not a target in this plan.

---

## Step 1: Preserve the current working baseline

Before changing generation again, keep regression coverage for what improved.

Preserve:

- no visible self-narration such as `I should consider...`,
- low structural repair rate,
- no invalid hard-blocker folding,
- no final commitment unless an explicit decision act occurs,
- direct response behavior,
- basic concession behavior,
- outcome-specific closure categories,
- moderator ability to produce consensus / fallback / no-decision.

Use a small golden set:

- n=2 simple choice,
- n=3 restaurant or board game,
- n=4 offsite or travel plan,
- n=5 birthday or project-planning topic,
- n=6/n=7 larger group topic.

Do not overfit to one transcript. The goal is robust interaction behavior.

---

## Step 2: Replace preference-argument turns with interaction-act turns

### Problem

Current turns often sound like:

> X is important, but Y matters more, so I prefer Z.

That is coherent, but too synthetic. Real speakers usually perform smaller interactional moves.

### Implementation direction

Introduce an explicit `DialogueAct` layer before language generation.

Example acts:

```python
DialogueAct = Literal[
    "open",
    "support",
    "object",
    "ask_constraint",
    "answer_constraint",
    "challenge_claim",
    "repair_misunderstanding",
    "soften_disagreement",
    "escalate_disagreement",
    "offer_condition",
    "conditional_accept",
    "weak_accept",
    "strong_accept",
    "block",
    "summarize_blocker",
    "push_closure",
    "vote",
]
```

Each turn prompt should receive:

- one act,
- one target speaker or target claim if applicable,
- one option focus if applicable,
- one unresolved issue if applicable,
- one style habit for the speaker.

The prompt should not ask the model to generally `argue for the persona preference`. It should ask the model to realize the selected act.

### Success signal

A transcript should contain a mix of short reactions, questions, objections, conditional moves, and closures. It should not read like every speaker is writing a debate paragraph.

---

## Step 3: Add shared reference and option aliases

### Problem

Speakers repeat full option names too often. This exposes the option table and weakens realism.

### Implementation direction

Create aliases after first mention.

Example:

```python
OptionReference = {
    "B": {
        "canonical": "Mountain Lodge Getaway",
        "aliases": ["the lodge", "the mountain one", "the trip", "that getaway"],
        "used_full_name": True,
    }
}
```

Rules:

- First mention may use the full name.
- Later normal turns should prefer aliases.
- Full names are allowed for explicit voting, moderator summaries, or when ambiguity exists.
- Speakers may use value-based shorthand: `the cheap one`, `the fancy place`, `the longer game`, `the safe option`.

### Success signal

After the opening phase, the transcript should mostly use natural shorthand. Repeated full names should become rare.

---

## Step 4: Track commitment as a ladder, not a boolean

### Problem

The system can treat weak agreement as final consensus. But human group decisions distinguish support, tolerance, uncertainty, condition, and blocking.

### Implementation direction

Replace binary support with a commitment ladder:

```python
Commitment = Literal[
    "blocks",
    "objects",
    "undecided",
    "can_live_with_if",
    "can_live_with",
    "supports",
    "strongly_supports",
]
```

Maintain for each participant:

```python
stance = {
    "preferred": "A",
    "current_option": "B",
    "commitment": "can_live_with_if",
    "condition": "if we cap cost at $25 per person",
    "unresolved_concern": "budget could get out of hand",
    "last_updated_by_turn": 17,
}
```

Consensus requires:

- every required participant is at least `can_live_with`, or
- every `can_live_with_if` condition is attached to the final decision and explicitly accepted as a next step.

The following should block clean consensus:

- `still not sure`,
- `not worth it`,
- `I am worried about...` without resolution,
- `only if...` without condition tracking,
- explicit non-support.

### Success signal

No transcript closes as consensus while a participant's last stance is unresolved or conditional without being named as conditional.

---

## Step 5: Make preference shifts causally visible

### Problem

Some participants change stance too smoothly. The reader sees the vote but not enough reason for the change.

### Implementation direction

Whenever a participant moves away from their preferred option, require a visible bridge.

Bridge types:

```python
ConcessionBridge = Literal[
    "condition",
    "tradeoff",
    "social_alignment",
    "residual_concern",
    "practical_next_step",
]
```

Examples:

- condition: `I can live with the food trucks if we set a rough budget first.`
- tradeoff: `I still prefer the bistro, but the trucks solve the group-variety issue better.`
- social alignment: `If everyone else is excited, I will not block it.`
- residual concern: `I am still unsure about the noise, but it is workable.`
- practical next step: `Let us check whether they can seat ten people before we lock it.`

### Success signal

A reader should be able to answer: `Why did this person change their mind?` from the transcript alone.

---

## Step 6: Add social face-work acts

### Problem

The dialogues are polite but socially flat. They lack the interpersonal work people do when disagreeing.

### Implementation direction

Add face-work as explicit act modifiers.

```python
Facework = Literal[
    "none",
    "hedge",
    "self_deprecate",
    "soften_objection",
    "affiliate_before_disagreeing",
    "signal_strong_boundary",
    "repair_tone",
    "invite_group_check",
]
```

Examples:

- hedge: `Maybe I am overthinking this, but...`
- self-positioning: `I know I keep bringing up budget, but...`
- affiliation: `I get why the lodge sounds more fun.`
- strong boundary: `I really would not want to do the expensive one.`
- tone repair: `Okay, I said that too strongly.`
- group check: `Would everyone actually enjoy that, or just us?`

Use sparingly. Do not make every turn emotionally marked.

### Success signal

Disagreement should feel socially situated, not only rationally contrasted.

---

## Step 7: Make personas behaviorally distinct

### Problem

Personas currently differ mainly by goals and option scores. Their actual speech often remains similar.

### Implementation direction

Convert each persona into a compact `voice_policy`.

Example:

```python
voice_policy = {
    "length": "short",
    "directness": "high",
    "conflict_style": "blunt_but_not_rude",
    "question_rate": "low",
    "hedging": "low",
    "closure_pressure": "high",
    "example_style": "concrete_constraints",
}
```

Possible behavior profiles:

- **direct organizer:** short, decisive, pushes closure,
- **cautious planner:** hedges, asks about risks and logistics,
- **peacemaker:** acknowledges both sides, asks bridging questions,
- **enthusiast:** reacts emotionally, uses vivid but short positive language,
- **analyst:** compares one concrete trade-off, avoids hype,
- **quiet participant:** speaks less, but final stance matters,
- **social driver:** checks group enjoyment and mood.

Do not make profiles caricatures. One or two stable habits per speaker are enough.

### Success signal

If speaker names are removed, at least some participants should still be identifiable by how they interact.

---

## Step 8: Route by contribution value, not round-robin fairness

### Problem

Balanced turn counts are tidy but not always natural. Real groups contain dominant speakers, quiet supporters, and people who skip a phase when they have nothing new to add.

### Implementation direction

Before selecting a speaker, compute possible contribution value.

A speaker is worth selecting if they can provide one of:

- unanswered objection,
- answer to a direct question,
- missing constraint,
- new concrete example,
- concession or condition,
- bridge between two positions,
- explicit vote or block,
- closure pressure.

Use fairness as a soft constraint across the whole dialogue, not as a reason to force low-information turns.

### Success signal

No speaker should talk merely to restate a known preference. Some unevenness in turn counts is acceptable and often more realistic.

---

## Step 9: Add practical-constraint routing

### Problem

The dialogues often stay at the level of abstract values: comfort, cost, novelty, complexity, feasibility. They need more concrete decision constraints.

### Implementation direction

Maintain a topic-sensitive list of missing constraints.

Examples:

```python
constraint_schema = {
    "restaurant": ["group_size", "budget_ceiling", "reservation", "dietary_needs", "noise", "parking"],
    "board_game": ["player_count", "available_time", "experience_level", "desired_mood", "teach_time"],
    "offsite": ["travel_time", "accessibility", "weather_backup", "budget_ceiling", "overnight_allowed"],
}
```

If the dialogue repeats an abstract value without progress, route the next turn to `ask_constraint` or `propose_condition`.

Examples:

- `How many people are actually coming?`
- `Would we cap it at 30 per person?`
- `Do we need a reservation?`
- `Can everyone handle a 200-mile drive?`
- `Do we want one long game or several short ones?`

### Success signal

Concrete constraints should visibly change the decision path.

---

## Step 10: Rework moderator policy

### Problem

The moderator sometimes sounds like the controller exposing internal state. It identifies broad value conflicts and pushes a compromise too early.

### Implementation direction

Moderator policy should be staged:

1. **diagnose:** name the exact blocker, not only broad values,
2. **ask holdout:** ask the resisting speaker what condition would work,
3. **surface missing constraint:** request budget, size, time, accessibility, etc.,
4. **propose modification:** attach condition to one existing option,
5. **can-live-with check:** ask all participants for explicit tolerance,
6. **close:** only after commitments are clear.

Moderator should prefer concrete questions over abstract summaries.

Bad:

> The issue is novelty versus comfort, can we settle on C?

Better:

> Liam, is your concern mainly the physical intensity, or the lack of downtime? If we keep one relaxed block in the schedule, could you live with the city option?

### Success signal

The moderator should not close or propose a compromise until the active blocker is known.

---

## Step 11: Strengthen option modification as compromise

### Problem

Real groups often do not simply choose A/B/C/D. They choose an option plus a condition or implementation detail.

### Implementation direction

Keep modifications lightweight. Do not create entirely new options.

```python
ModifiedProposal = {
    "base_option": "B",
    "condition": "if we can keep it under $30 each",
    "implementation_detail": "share plates and skip the expensive add-ons",
    "addresses_concern": "budget",
    "owner": "Ava checks menu prices",
}
```

Common modification types:

- budget cap,
- reservation check,
- weather backup,
- transport/carpool,
- shorter version,
- add-on activity,
- accessibility check,
- backup option.

### Success signal

Consensus can be `B with condition X`, not only plain `B`.

---

## Step 12: Improve closure realism

### Problem

Closures are clearer than before but still too generic.

### Implementation direction

Use outcome-specific closure templates, but keep language natural.

Consensus:

```text
Okay, then we will do the food trucks. It solves the variety issue, and Lena is okay with it if we keep an eye on cost. Ava, can you check whether they take group reservations?
```

Conditional consensus:

```text
Sounds like Ticket to Ride works if we keep Sushi Go as a short backup. Let us confirm who is coming before we call it final.
```

Fallback:

```text
Most of us are leaning toward the lodge, but Kai still has accessibility concerns. Let us treat it as the working option and check transport before booking.
```

No decision:

```text
We are stuck between budget and atmosphere. Before deciding, we need the actual group size and whether the bistro can seat us.
```

Participant closing lines should be optional. One or two practical acknowledgments are enough.

### Success signal

The ending should explain what was decided, under what condition, and what happens next.

---

## Step 13: Add narrow surface cleanup

### Problem

Small artifacts still make transcripts look generated.

### Implementation direction

Use deterministic cleanup only for obvious surface issues:

- remove spaces before punctuation,
- reject or regenerate unfinished sentences,
- fix obvious typo whitelist if safe,
- prevent repeated identical final acknowledgments,
- prevent malformed quote/spacing artifacts.

Do not reintroduce broad style repair. This is cleanup, not rewriting.

### Success signal

No visible ` .`, unfinished line, or obvious malformed ending in final transcripts.

---

## Step 14: Add larger-group contribution roles

### Problem

n=5-7 discussions can become repetitive because too many participants need something to say.

### Implementation direction

Assign temporary contribution roles by phase.

Opening phase:

- first supporter,
- first objector,
- missing-constraint asker.

Middle phase:

- answer provider,
- bridge-builder,
- concrete-detail provider,
- holdout.

Closing phase:

- conditional accepter,
- blocker,
- summarizer,
- final voter.

Speakers without a useful role can remain silent until they have a meaningful contribution.

### Success signal

Large-group transcripts should have fewer low-information agreement turns and more role-specific contributions.

---

## Step 15: Update evaluation metrics

The current metrics are useful but not enough for human-likeness.

Add metrics for:

- full option-name repetition after first mention,
- alias usage rate,
- unresolved concern at closure,
- conditional acceptance count,
- unsupported preference shift count,
- repeated dialogue-act pattern,
- practical constraint coverage,
- face-work presence,
- speaker distinguishability estimate,
- low-information support turns,
- malformed surface artifacts.

Do not over-automate subjective realism. Use metrics as warning signals, then inspect transcripts manually.

---

## Recommended implementation order

1. Preserve current regression tests and golden examples.
2. Add `DialogueAct` routing and stop generating generic preference arguments.
3. Add shared option aliases and reduce repeated full option names.
4. Replace binary support with the commitment ladder.
5. Require visible concession bridges for non-preferred acceptance.
6. Add face-work act modifiers for disagreement and social repair.
7. Convert persona traits into compact behavioral `voice_policy` objects.
8. Route speakers by contribution value instead of balanced participation.
9. Add topic-sensitive practical-constraint detection.
10. Rework moderator policy around blockers, holdouts, and can-live-with checks.
11. Represent modified proposals as option + condition + owner.
12. Improve consensus, conditional consensus, fallback, and no-decision closures.
13. Add narrow deterministic surface cleanup.
14. Add n=5-7 contribution roles.
15. Extend evaluation metrics and manually review the golden set.

---

## What not to do

- Do not solve realism by only increasing turn count.
- Do not force every participant to speak in every phase.
- Do not treat weak agreement as consensus.
- Do not let the moderator close while a blocker is still active.
- Do not make every turn a full argument.
- Do not make personas into exaggerated stereotypes.
- Do not add broad LLM repair loops for style.
- Do not introduce entirely new options during compromise; modify existing options instead.
- Do not optimize away context that is needed for social continuity.
- Do not overfit to one successful transcript.

The target is not a cleaner debate. The target is a believable group decision process: partial, socially managed, constraint-driven, and clear about what each person can actually accept.
