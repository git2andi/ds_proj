# TODO: Option-Grounded Multi-User Simulator

This file lists only open issues. Completed items are intentionally not tracked here. Keep this file current after every implementation pass.

The current priority is to fix prominent failures visible in real generated transcripts before adding new features. Fixes must be general across arbitrary option-grounded topics. Do not solve a problem by adding large prompt blocks unless a smaller controller, parser, validator, state, or repair-policy change is not sufficient.

## Implementation protocol for each update

1. **Archive old logs first.** Before changing behavior, create `logs/archive/` if it does not exist and move all existing log files/directories from `logs/` into `logs/archive/`. Do not delete logs.
2. **Work on one issue at a time.** Pick exactly one open issue from this file unless the issue explicitly says that several small changes belong together.
3. **Apply the minimal fix.** Change the smallest amount of code/config needed to solve the selected issue. Prefer controller, parser, validator, state, or repair-policy fixes over simply making prompts longer.
4. **Validate with example runs.** After the fix, run at least:
   - one mandatory `n=3` run with a random topic,
   - at least one additional run with a different group size in the `n=2..7` range,
   - more runs only if the changed behavior is unstable or group-size-dependent.
5. **Inspect the transcript and metrics.** Do not rely only on successful execution. Check whether the transcript actually shows the intended behavior.
6. **Append newly observed issues.** If validation exposes a new problem, add it under `Newly observed issues` with log path/date, topic, group size, and the smallest description of the failure.
7. **End only after verification.** Finish the update only when the selected issue is implemented, the code compiles, and validation runs show the intended behavior or a clearly documented remaining limitation.

## Open issues, ordered by priority

### 1. Enforce response obligations for direct questions

Current problem: direct questions do not reliably receive answers from the addressed participant. This affects both moderator-to-user questions and user-to-user questions. In the Stockholm run, the moderator directly asks Anton whether the red-eye baggage policy is a deal-breaker, but Kenji answers next. Kenji also asks Anton a direct question about checked bags, and Anton does not answer that adjacency pair before the conversation moves on.

Required behavior:
- If the moderator addresses a named participant, the next participant turn should normally be assigned to that participant.
- If one participant asks another participant a visible direct question, the next participant turn should normally be assigned to the addressed participant.
- Other participants may interrupt only if the policy explicitly chooses an interruption/side-comment act.
- The response must answer the pending question rather than starting a new argument or voting unless the question explicitly asked for a vote.
- Response obligations should expire only after the target answers, the moderator cancels/redirects, or a hard closure condition is reached.

Implementation notes:
- Add a small `response_obligation` state object: `target_speaker`, `source_speaker`, `question_text`, `expected_act`, `created_turn`, `expires_after`.
- Detect direct questions with existing visible text, not hidden metadata.
- Make the router consume `response_obligation` before normal speaker selection.
- Add a validation guard: if a targeted answer is required and the wrong speaker is selected/generated, repair by selecting the required target instead of only changing the prompt.
- Log unanswered obligations in metrics.

Validation target:
- In an `n=3` run, every direct named question should be answered by the named participant within the next one or two participant turns.

### 2. Reduce excessive name-prefixing and artificial address markers

Current problem: many utterances start with the addressee name followed by a generic agreement/disagreement marker, e.g. `Kenji, true, but...`, `Lila, I get...`, `Anton, that is fair...`. Some addressing is good, but in the current output it becomes formulaic and makes the dialogue feel synthetic.

Required behavior:
- Keep direct names when they serve a clear interactional function: answering a direct question, challenging someone, clarifying a misunderstanding, or handing over to a specific user.
- Avoid name-prefixing for ordinary continuation turns.
- Avoid repeated openings such as `Name, I get...`, `Name, true, but...`, `Name, fair, but...` across consecutive turns.
- Allow implicit response without a name when the local context is obvious.

Implementation notes:
- Add a lightweight local style tracker over the last 3-5 participant turns:
  - count name-prefixed openings,
  - detect repeated openings with regex,
  - mark `avoid_name_prefix` for the next realization when name-prefix density is too high.
- Prefer a compact prompt flag such as `avoid_name_prefix=True` over adding many naturalness instructions.
- Add a repair only when the generated utterance repeats a recent opening pattern or starts with an unnecessary name prefix.

Validation target:
- In a normal `n=3` run, not more than roughly one third of participant turns should start with another participant's name unless the topic genuinely involves many direct questions.

### 3. Break repetitive sentence templates

Current problem: many turns use the same rhetorical structure: `X is good, but Y is a concern`, `I get that, but...`, `true, but...`, or `saving money is good, but comfort matters`. This creates fluent but samey assistant-like dialogue.

Required behavior:
- Participants should vary between concise claims, questions, comparisons, concessions, objections, summaries, and final commitments.
- The same local template should not repeat several times in a row.
- Different simulator parameters should produce visible differences in style:
  - direct users: shorter, less hedged claims,
  - cooperative users: bridge statements and summaries,
  - stubborn users: stronger resistance and fewer concessions,
  - low-engagement users: shorter turns unless directly asked.

Implementation notes:
- Add a local `surface_pattern` classifier for generated turns, e.g. `concede_but`, `cost_tradeoff`, `comfort_tradeoff`, `direct_vote`, `question`, `summary`.
- Penalize or repair repeated `surface_pattern` values in consecutive turns.
- Prefer changing move/act selection and repair checks before expanding the base prompt.
- Keep voice guidance compact and parameter-driven.

Validation target:
- In an `n=3` run, the last 8 participant turns should not all follow the same concession-plus-objection template.

### 4. Prevent unsupported factual additions beyond the option board/context

Current problem: sims sometimes add plausible but unsupported facts. In the Stockholm run, a participant mentions quieter airports and customs, and another suggests the direct SAS flight includes checked bags, even though these facts are not part of the shared option/context board. The generated option facts are allowed to be artificial, but once generated they are the hard world facts of the simulation.

Required behavior:
- Sims may reason from provided facts, but they must not introduce new concrete logistical facts, included services, policies, locations, hidden fees, timing consequences, or operational assumptions unless those are present in the option board/context.
- Sims may express uncertainty as uncertainty: `we do not know whether checked bags are included` is allowed if the fact is absent.
- Option positives/negatives and attributes are the authoritative fact base.

Implementation notes:
- Add a compact `known_fact_terms` / `unsupported_fact_risk_terms` check for common concrete additions: included baggage, customs, visa, airport security, hotel, refund policy, seat availability, weather, exact arrival time, etc.
- Do not try to solve this with a huge prompt. Use a small validation warning/repair when unsupported concrete facts appear.
- Allow domain-generic reasoning only if it follows from listed attributes, e.g. red-eye + no checked baggage → discomfort / packing light.
- Consider adding `unknowns` to the scenario board later, but do not invent them during dialogue.

Validation target:
- In a travel run, participants should not invent new services/policies such as checked baggage being included unless listed.

### 5. Stabilize final-vote behavior and avoid redundant re-voting

Current problem: finalization can become noisy. A participant may vote, then be asked again unnecessarily. Later votes can overwrite earlier votes, producing unstable visible votes. In the Stockholm run, Anton votes for Option A and later votes for Option C after another moderator prompt, while Lila repeats her vote for B.

Required behavior:
- Once a participant gives a clear final vote, do not ask them for another final vote unless their later visible text explicitly says they are changing their vote.
- Separate observed vote states:
  - no vote yet,
  - weak/conditional support,
  - clear final vote,
  - explicit vote change.
- If a participant gives a deal-breaker plus a clear vote, count the vote only if the commitment is syntactically clear and not conditional.
- If a vote is unclear, ask the same participant one targeted clarification question.
- Final-vote prompts must not restart the debate.

Implementation notes:
- Add `vote_status_by_persona` or equivalent observed-state tracking.
- Do not overwrite clear votes unless text contains an explicit change marker such as `I changed my mind`, `then I switch to`, `actually I vote for`.
- Moderator should ask only non-voters or unclear voters during finalization.
- Add small deterministic tests for vote overwrite and clarification behavior.

Validation target:
- In a final vote round, each participant should normally produce at most one clear vote.

### 6. Improve unresolved-handling before closure

Current problem: unresolved status can be correct, but the path to unresolved should feel socially and procedurally justified. In the Stockholm run, the final unresolved state is technically valid because votes split D/B/C, but the conversation contains missed answers and repeated vote prompts before closure.

Required behavior:
- Close as unresolved only after:
  - required response obligations are resolved or explicitly abandoned,
  - each participant has had a chance to clarify one final stance,
  - no unique majority is visible,
  - no obvious compromise option has pending discussion.
- If votes are split, the moderator should summarize the split once and either ask for one compromise attempt or close if no movement occurs.
- Avoid repeated final-vote prompts after a clear split.

Implementation notes:
- Add a small `closure_attempts` or `compromise_attempted` flag.
- If all participants voted for different options, trigger one `split_vote_compromise_prompt` before unresolved closure, unless hard max turns is reached.
- Keep this bounded so unresolved runs do not drag on.

Validation target:
- Split votes should produce either one bounded compromise attempt or a clean unresolved close, not repeated vote loops.

### 7. Refine option coverage without forcing artificial discussion

Current problem: option coverage is better than before, but coverage should mean meaningful processing, not only raw mentions. Also, clearly unattractive options should not be over-discussed.

Required behavior:
- Before voting, each option should be at least mentioned, compared, rejected, or explicitly skipped.
- Coverage prompts should be short and natural.
- A participant does not need to like the covered option.
- Do not over-discuss clearly unattractive options.
- Coverage should happen before finalization, not after votes already started.

Implementation notes:
- Track meaningful processing, not only mention count: `reason`, `objection`, `comparison`, `explicit_skip`.
- Prefer comparison prompts for compromise options.
- Avoid adding more than one coverage nudge unless the run is still clearly in discussion phase.

Validation target:
- In a four-option run, no option should remain completely untouched before voting unless the moderator or participants explicitly skip it.

### 8. Strengthen agenda-based simulator behavior

Current problem: agenda items are minimal. They help structure behavior, but they are not yet strong enough to make the sims look like persistent user simulators with goals and pending communicative tasks.

Required behavior:
- Each sim should have a small private agenda based on goal, initial preferences, blockers, and simulator parameters.
- Agenda items should include pending communicative acts such as:
  - state preference,
  - ask practical constraint,
  - object to option,
  - answer challenge,
  - propose compromise,
  - give final vote.
- Agenda items should have status: pending, completed, blocked, or obsolete.
- The router should prefer agenda-compatible moves without scripting exact text.

Implementation notes:
- Keep agenda simple; do not rebuild a full ConvLab-style policy yet.
- Use agenda for behavior selection, not hidden outcome evidence.
- Log agenda status for later debugging and evaluation.

Validation target:
- In a transcript, each sim should show continuity between their goal, earlier statements, later objections, and final vote.

### 9. Prepare evaluation layer, but keep it lightweight

Current problem: evaluation exists as a scaffold, but it should be organized so later work can expand it without touching generation logic.

Required behavior:
- Keep evaluation separate from logging.
- Include only stable basic metrics for now:
  - participant turn counts,
  - top speaker share,
  - moderator ratio,
  - visible vote count,
  - outcome status,
  - option coverage,
  - unanswered direct-question count,
  - name-prefix rate,
  - repeated-opening-pattern count.
- Prepare placeholders for later metrics without implementing complex scoring yet.

Implementation notes:
- Do not spend a full implementation pass on advanced metrics yet.
- Add TODO stubs for future metrics such as participation Gini, direct response rate, question-answer completion, repetition score, and engagement realization error.

Validation target:
- Metrics should expose the failures that are currently being manually spotted in transcripts.

### 10. Add automated tests for fragile logic

Current problem: important behavior is currently validated mostly through manual runs. The fragile parts need small deterministic tests.

Required tests:
- visible option reference resolution,
- clear vote detection,
- conditional support rejection,
- vote overwrite behavior,
- majority/successful/unresolved outcome logic,
- moderator-to-user response obligation,
- user-to-user direct-question obligation,
- option-coverage trigger,
- repeated-speaker allowance only when justified,
- name-prefix/repeated-template detector.

Implementation notes:
- Start with pure functions from `parsing.py`, outcome logic, style tracking, and routing policy.
- Avoid tests that require live LLM calls.

Validation target:
- A local test command should catch the known Stockholm-style failures without needing an LLM call.

### 11. Keep token usage bounded, but do not optimize prematurely

Current position: token usage around 5k-20k input tokens per typical `n=3` run is acceptable for now. Do not aggressively compress prompts if it worsens transcript quality.

Required behavior:
- Prevent token use from growing unbounded as group size increases.
- Keep per-turn prompt context intentional.
- Do not reintroduce extremely large transcripts such as 100k+ tokens per run.
- Revisit token optimization only after simulator behavior stabilizes.

Implementation notes:
- Log total setup/dialogue input and output tokens as already done.
- Add a warning threshold if a normal `n=3` run exceeds the configured upper range.
- Do not make token optimization a priority unless it starts harming iteration speed or cost.

Validation target:
- Normal `n=3` runs should stay below the configured warning threshold unless the transcript is intentionally long.

### 12. Add optional corpus-inspired presets later

Current problem: corpus statistics such as Delidata-style turn length, group size, and speaker dominance are known but not yet represented as selectable presets.

Required behavior:
- Add optional presets later, not hard constraints.
- Example preset fields:
  - typical discussion length,
  - preferred group size,
  - expected top-speaker share,
  - dominance range,
  - participation imbalance tolerance.
- The simulator should still work without a corpus preset.

Implementation notes:
- Keep this lower priority until routing, votes, moderator targeting, and local surface naturalness work reliably.

Validation target:
- Presets should change runtime parameters measurably without requiring topic-specific hacks.

## Newly observed issues

Add new validation findings here after each implementation pass. Include log path/date, topic, group size, and the smallest description of the failure.

- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: moderator directly addressed Anton, but Kenji answered next.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: Kenji asked Anton a direct question, but Anton did not answer it before the flow moved to finalization.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: excessive name-prefixed openings made the dialogue formulaic.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: repeated concession-plus-objection sentence templates made speakers sound too similar.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: participants added unsupported factual assumptions beyond the option/context board.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: Anton produced two different final votes after redundant prompting.
- `20260701_002531_085964`, `Book a flight to Stockholm`, `n=3`: unresolved outcome was technically valid, but closure happened after missed response obligations and repeated vote prompts.
