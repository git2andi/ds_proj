# UPDATE.md — Current Dialogue Quality Problems and Next Fixes

## 0. Current status

The latest code update improved **validity**, but it did not yet solve **naturalness**.

The verifier, full option visibility, and repair infrastructure are useful progress. The system is now better at avoiding hard invalid outputs, such as denying existing options or drifting outside the option set. However, the generated chats still sound too mechanical.

The current main failure is no longer: “the simulator is completely broken.”

The current main failure is:

> The simulator produces valid but over-safe option-commentary instead of natural group conversation.

The chats still often follow this repetitive pattern:

```text
X is a valid concern.
Y has this trade-off.
I still prefer Option Z.
```

This makes the dialogue feel like a sequence of tiny evaluations instead of humans casually deciding something together.

---

## 1. What got better

### 1.1 Full option visibility improved validity

The earlier problem where sims claimed that a valid option was not available appears to be improved.

The change that made `build_relevant_options()` return all options was correct. Do not revert this.

All participants should always see all options.

### 1.2 The verifier layer is a useful addition

The new `verifier.py` module is conceptually correct.

The generation flow now roughly does this:

```text
generate raw message
strip name prefix
verify message
repair once if needed
store verification result
```

This is the right direction.

The verifier should remain deterministic and fast. It should not become another LLM-based reasoning layer.

### 1.3 Logging is better

Verification issues and repair attempts are now available for analysis.

This is important because future improvements should be measured, not guessed.

### 1.4 Token cost improved somewhat

The new logs appear cheaper than the older heavy runs.

This is positive, but token cost is still not the main issue. The main issue is conversational quality.

---

## 2. What is still bad

### 2.1 The chats still sound like option evaluation, not conversation

The most common message shape is still:

```text
[Name]'s point is valid, but Option X has Y, so I prefer Option Z.
```

This is too formal and too repetitive.

The sims do not often produce natural in-between moves such as:

```text
Yeah.
Not sold.
That might be enough.
Can we rule that one out then?
Wait, why D?
Okay, fair.
I can live with that.
No, not for me.
```

The result is a dialogue that is valid but fake.

### 2.2 Too many turns explicitly acknowledge the previous point

The chats overuse phrases like:

```text
That's a valid point.
X is right.
That is a concern.
I agree that...
Good point.
```

Acknowledgement is natural sometimes. It becomes robotic when almost every turn starts this way.

The system needs to distinguish between:

```text
useful acknowledgement
```

and

```text
acknowledgement loop
```

### 2.3 The sims repeat the same abstract concern words

The logs still repeat terms like:

```text
cost
reliability
legroom
challenge
role clarity
complex mechanics
convenience
```

The problem is not only lexical repetition. It is **semantic repetition**: the same point is rephrased several times.

Example pattern:

```text
A: Higher cost is a concern.
B: Yes, the cost concern is valid.
C: The cost issue matters, but reliability is important.
A: Reliability offsets the cost.
```

This is not real progression.

### 2.4 There is not enough conversational move variety

Current turns mostly do one of these:

```text
support option
oppose option
acknowledge concern
state preference
```

The system needs more surface-level conversational moves:

```text
short yes
short no
uncertainty
rule-out
compromise
decision pressure
small question
light acknowledgement
actual new reason
```

This does **not** mean restoring a heavy dialogue-act planner. It means adding a small surface-move mechanism or prompt instruction that permits non-analytical turns.

### 2.5 Options are too thin

Some generated option sets give only one upside and one trade-off per option.

That means the LLM has very little material to discuss. It naturally repeats the same few attributes.

Example:

```text
Option A: convenient but expensive
Option B: cheap but less comfortable
Option C: entertaining but long layover
Option D: scenic but less frequent
```

This structure is valid but too shallow.

The sims need bounded but richer option cards.

### 2.6 Confirmation and compromise feel too sudden

Sometimes a participant votes for one option and then immediately accepts a different candidate without the transition feeling earned.

This can be logically valid if the option is in their acceptable set, but it sounds unnatural unless the sim explicitly frames it as compromise.

Bad:

```text
I vote D.
Yeah, A works for me.
```

Better:

```text
I still prefer D, but I can live with A if reliability matters more to everyone.
```

### 2.7 The moderator still needs more targeted holdout handling

Generic confirmation questions produce weak agreement.

Instead of:

```text
Everyone good with Option A?
```

use:

```text
Ava, you picked D. Could you live with A, or is that a no?
```

This makes compromise visible and human-readable.

### 2.8 Closure is still too scripted

Requiring every participant to close often creates repeated lines like:

```text
Sounds good.
Works for me.
Good plan.
```

Real chats often end after the decision, with maybe one or two short sign-offs.

The system should not force full-group closure every time.

---

## 3. Main causes

### 3.1 The prompt still over-prioritizes option-specific commentary

The negotiation/discussion prompt still strongly pushes the model to:

```text
talk about options by name
engage with the last claim
agree with a detail
push back on a claim
name a trade-off
```

This sounds reasonable, but it causes the exact repetitive pattern now visible in the logs.

The prompt is still training the LLM to produce:

```text
X is good/bad, so I prefer Y.
```

### 3.2 “Engage with the last point” is too strong

The instruction to always engage with the previous claim causes repeated acknowledgement.

In natural chat, people do not explicitly validate the previous point every turn.

They often simply:

```text
answer
reject
accept
move on
ask a small question
narrow the decision
```

The system should stop forcing explicit evaluation of the last message.

### 3.3 The verifier checks validity more than naturalness

The verifier currently catches hard failures such as:

```text
invalid option reference
option denial
invented facts
missing vote
unclear confirmation
self-repetition
```

But it does not yet catch:

```text
group-level acknowledgement loops
semantic repetition of the same concern
overuse of "valid point" patterns
too many option-review turns in a row
```

So the system is valid, but still monotonous.

### 3.4 Repetition is mostly checked per speaker

The current repetition logic focuses on whether one speaker repeats themselves.

But the current bad pattern is often distributed across speakers:

```text
Speaker 1: That's a concern.
Speaker 2: Yes, that concern is valid.
Speaker 3: I agree it's a concern.
```

This is a group-level repetition problem, not only a self-repetition problem.

### 3.5 The old debate/state architecture still influences behaviour

The code still contains concepts such as:

```text
DialogueAct
StanceTable
ChallengeRecord
conditional_support
concession
challenge
```

Even if parts of the old act planner were removed, the prompt and state logic still think in debate fragments.

This keeps pushing the dialogue toward micro-debate instead of casual decision-making.

### 3.6 Option cards do not provide enough bounded content

The LLM cannot produce richer reasoning if each option only contains one or two attributes.

It either invents details, which the verifier should block, or repeats the few available details.

The fix is not to allow hallucination. The fix is to generate better structured options.

---

## 4. What needs to be done next

## 4.1 Replace “always engage with the last claim”

This is the highest-impact prompt change.

Current behaviour pushes:

```text
Agree with detail / push back / name trade-off.
```

Replace it with:

```text
Do not automatically evaluate the last message.
Reply only if you have something new.
Otherwise make a different natural move:
- ask a short question,
- propose narrowing,
- state uncertainty,
- give a brief yes/no reaction,
- compromise,
- or move the decision forward.
```

The goal is to stop the constant “valid point” loop.

---

## 4.2 Add a lightweight surface-move mechanism

Do **not** restore the heavy old act planner.

Instead, add a small set of surface moves that shape the next turn’s form.

Suggested moves:

```text
ACK_ONLY
SHORT_NO
ANSWER
NEW_REASON
PUSHBACK
COMPROMISE
DECISION_MOVE
QUESTION
```

Rules:

```text
ACK_ONLY:
  max 6 words
  no option analysis

SHORT_NO:
  max 8 words
  no full explanation unless asked

ANSWER:
  directly answer the addressed question

NEW_REASON:
  add a genuinely new point
  do not reuse the same option attribute

PUSHBACK:
  disagree directly
  do not start with "X's point is valid"

COMPROMISE:
  explicitly mark compromise
  e.g. "I still prefer B, but I can live with A."

DECISION_MOVE:
  move toward narrowing or voting
  e.g. "Then we're basically between A and D."

QUESTION:
  ask one short useful question
```

This should be small and local, not a large theoretical dialogue-act planner.

---

## 4.3 Add `ACK_LOOP` to the verifier

Add a new verifier issue:

```text
ACK_LOOP
```

Trigger it when:

1. the generated message is mostly acknowledgement, and
2. the recent 2–3 participant turns already contain acknowledgement language.

Patterns to detect:

```text
valid point
fair point
good point
X is right
I agree with X
that's a concern
that concern is valid
I see your point
```

Repair instruction:

```text
This message only acknowledges the previous point again.
Rewrite it as a different natural move:
- ask a short question,
- make a concrete decision proposal,
- say a brief yes/no,
- compromise,
- or add one genuinely new reason.
```

Do not ban acknowledgements globally. Only repair acknowledgement loops.

---

## 4.4 Make repeated old points a repair, not only a warning

If a speaker repeats the same option attribute again, repair it.

Example repetition to repair:

```text
First turn:
SAS reliability is worth the cost.

Later turn:
Option A's reliability offsets the higher cost.
```

This is the same point.

Detection can be approximate:

```text
same speaker
same option
same attribute keyword
high token overlap or same point signature
```

Store point signatures like:

```text
A: reliability offsets cost
B: cheaper but less legroom
D: scenic but infrequent
```

Then prevent the same speaker from reusing the same point signature unless they are explicitly voting or confirming.

---

## 4.5 Stop requiring option mention every turn

Not every natural turn needs an option name.

Allow turns like:

```text
Yeah, that would annoy me too.
Not enough for me.
Can we rule that one out then?
I’d rather not make this harder than needed.
Okay, then we’re basically between A and D.
```

The current system pushes too hard toward option labels. That makes every line sound like an evaluation rubric.

Use option names when adding a real reason, voting, or clarifying a candidate. Do not require them for short reactions.

---

## 4.6 Improve option generation

Options need more bounded material.

Current thin structure:

```text
upside
trade-off
best_for
```

Recommended richer structure:

```text
title
upside
tradeoff
practical_concern
social_or_group_fit
uncertainty
best_for
```

Example:

```text
Option A - SAS Flight 101
upside: reliable and convenient departure
tradeoff: higher cost
practical_concern: less flexibility if plans change
social_or_group_fit: easiest for people who want low disruption
uncertainty: unclear baggage/extra-fee situation
best_for: reliability-first travellers
```

This gives sims more grounded content without inventing fake numbers.

---

## 4.7 Make compromise explicit in votes and confirmations

If a sim accepts a non-preferred option, it should often say so.

Bad:

```text
Option A works best for me.
```

when the sim privately prefers B.

Better:

```text
I still prefer B, but I can live with A if everyone cares more about reliability.
```

Rule:

```text
If chosen option != private preferred:
  require compromise framing unless phase is simple confirmation.
```

This makes preference movement visible.

---

## 4.8 Target holdouts directly

When votes are split, the moderator should ask holdouts by name.

Bad:

```text
Everyone okay with Option A?
```

Better:

```text
Ava, you picked D. Could you live with A, or is that a no?
```

This prevents fake consensus.

---

## 4.9 Reduce forced closure

Do not require every sim to produce a closing line.

Possible closure strategies:

```text
moderator-only final line
```

or:

```text
moderator final line + one optional participant reaction
```

This avoids repeated “sounds good” endings.

---

## 4.10 Lower randomness after prompt fixes

Suggested config:

```yaml
temperature: 0.65
top_p: 0.9
top_k: 40
```

Do not expect temperature alone to fix naturalness.

The prompt shape and verifier rules matter more.

---

## 5. Concrete code areas to inspect

### 5.1 `prompts.py`

Inspect and change:

```text
phase_instruction_text()
interaction_instruction_block()
position_discipline_block()
```

These functions currently create much of the robotic style.

Most important:

- weaken forced last-claim engagement;
- allow short non-analytical replies;
- stop requiring every turn to name an option;
- reduce “say one reason” pressure.

### 5.2 `prompt_context.py`

Inspect:

```text
build_memory_block()
build_move_instruction()
build_output_contract()
```

The memory block may be overfeeding abstract repeated terms.

Simplify memory to:

```text
last own turn
recent own point signatures
recent chat
current candidate/votes
```

Avoid too many summaries of other people’s arguments.

### 5.3 `verifier.py`

Add:

```text
ACK_LOOP
SEMANTIC_POINT_REPEAT
```

Strengthen:

```text
SELF_REPETITION
```

Older point repetition should trigger repair, not only warnings.

### 5.4 `simulator.py`

Make sure repaired messages are re-verified.

Also ensure the repair prompt targets the exact issue.

Do not use generic repair prompts for everything.

### 5.5 `moderation.py`

Make holdout prompts targeted.

Prevent generic confirmation after split votes.

Add deterministic moderator lines for:

```text
ask_holdout(candidate, holdout_name)
ask_compromise(candidate, holdout_name)
announce_success(candidate)
announce_force_close(candidate, reason)
```

### 5.6 `orchestrator.py`

Inspect vote and confirmation flow.

Check whether:

- split votes are handled before generic confirmation;
- a non-preferred acceptance is marked as compromise;
- force-close is honest;
- full-group closure is always forced.

### 5.7 `reasoning.py`

Consensus still uses public stance logic.

Long-term, use:

```text
explicit votes first
private acceptability second
stance table only as weak context/evaluation
```

Do not let vague statements drive final candidate selection.

---

## 6. Updated priority order

Do these in this order.

### Priority 1 — Prompt shape

Change the discussion prompt so it no longer forces every turn to:

```text
acknowledge previous point + option pro/con + preference
```

This is the biggest naturalness fix.

### Priority 2 — ACK_LOOP verifier

Add group-level acknowledgement-loop detection and repair.

This directly attacks the current visible failure.

### Priority 3 — Surface move variety

Add a small, non-theoretical surface-move mechanism.

This gives the LLM permission to produce short yes/no/uncertain/decision-moving turns.

### Priority 4 — Richer option cards

Generate options with more bounded qualitative fields.

This gives the sims more real material.

### Priority 5 — Compromise visibility

If a sim accepts a non-preferred option, it should often explicitly frame it as compromise.

### Priority 6 — Targeted holdout handling

Moderator should ask specific holdouts, not the whole group generically.

### Priority 7 — Reduce forced closure

Stop making every participant produce a closing line.

---

## 7. What not to do

Do not add more literature right now.

Do not add a large emotional model.

Do not reintroduce a heavy dialogue-act planner.

Do not add another LLM judge into the runtime loop.

Do not solve acknowledgement repetition with a huge phrase blacklist.

Do not force conflict into every dialogue.

Do not force every turn to include an option name.

Do not force every turn to include a reason.

The next fix is not more theoretical complexity.

The next fix is:

```text
less forced analysis
more local conversational move variety
stronger repair for repeated acknowledgement and repeated points
```

---

## 8. Success criteria for the next update

After the next code update, inspect 10 generated dialogues.

The update is successful if:

1. Not every turn evaluates an option.
2. Some turns are very short and natural.
3. Acknowledgement phrases do not appear in chains.
4. Repeated concern words are reduced.
5. Sims sometimes say simple yes/no/uncertain replies.
6. Compromise is explicitly framed.
7. Split votes trigger targeted holdout questions.
8. Final decisions are still valid A-D.
9. The verifier logs `ACK_LOOP` or repeated-point repairs when appropriate.
10. The chats feel less like a debate rubric and more like people deciding something.

---

## 9. Summary

The simulator is now more valid, but still not natural.

The main issue is that the prompt and state logic still reward safe option commentary:

```text
acknowledge -> mention option -> state pro/con -> restate preference
```

The fix is to change the system’s local turn behaviour:

```text
allow short human reactions
avoid automatic acknowledgement
repair acknowledgement loops
repair repeated semantic points
make compromise explicit
target holdouts directly
provide richer option material
```

This should be the next focused implementation pass.
