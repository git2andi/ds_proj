# TODO — Move Discussion Authority from the Global Controller to the User Simulators

## Goal

Refactor the dialogue architecture so that the simulated users are actual behavioral decision makers rather than LLM voices realizing controller-authored moves.

The final authority split must be:

```text
Environment / discussion framework
- defines the scenario, options, phases, legal actions, grounding rules, and termination
- detects visible questions, concerns, comparisons, votes, and thread state
- enforces direct-response and formal protocol obligations
- arbitrates access to the floor
- validates generated utterances and updates state only from accepted visible text

Simulator policy
- decides whether this simulator wants to speak when the floor is open
- calculates its own willingness to speak
- chooses its own communicative act
- chooses the public turn, participant, issue, or option it wants to react to
- chooses its own intended direction and reason
- decides whether to hold, soften, compromise, switch, or vote for an option

LLM utterance realization
- receives one already-selected simulator intention
- expresses that intention naturally in the simulator's voice
- does not decide the speaker, stance direction, act, target, vote, or compromise outcome
```

This is a Python-based symbolic/probabilistic simulator policy. Do not add one LLM policy call per simulator per turn. The only participant LLM call in an ordinary turn remains the utterance-generation call for the winning simulator.

---

## Fixed design decisions

These are implementation requirements, not open design questions.

1. **The existing centralized open-floor router must be removed.**
   - Do not keep the old controller routing as a second configurable mode.
   - The previous state is stored in Git and can be used externally as a baseline.
   - Do not leave parallel controller and simulator policies that can silently disagree.

2. **Opening turns remain protocol-required.**
   - Every simulator speaks once during the opening round.
   - The framework schedules the opening order.
   - The simulator policy chooses the opening option focus and participant-specific reason from its own state.
   - The LLM only realizes that opening intention.

3. **Direct questions remain mandatory adjacency-pair obligations.**
   - A valid direct question to a named simulator gives that simulator the next turn.
   - The broad act is constrained to `ANSWER`.
   - The simulator still decides the answer's stance, direction, option focus, reason, concession, condition, or pushback.
   - The controller must not prescribe “accept,” “reject,” “defend,” or “concede.”

4. **Group questions are not assigned to a respondent by the controller.**
   - A group question opens a public question thread with no `required_respondent`.
   - It becomes a strong stimulus for all eligible simulators.
   - Relevant simulators may self-select an `ANSWER` bid.
   - The floor manager chooses among those bids.

5. **Engagement affects willingness inside the simulator policy.**
   - Engagement provides the simulator's baseline tendency to claim the floor.
   - The floor manager must not apply engagement a second time.
   - Relevance and personal stake must be able to outweigh engagement: a low-engagement simulator whose preferred option was challenged can beat a highly engaged simulator with nothing new to add.

6. **The floor manager only arbitrates access to the floor.**
   - It may enforce eligibility, mandatory obligations, phase legality, anti-monopoly constraints, recent-speaker penalties, and bounded failure recovery.
   - It may select the next-best existing bid when the best bid is invalid, ineligible, or repeatedly fails realization.
   - It must not alter a bid's act, focus, target, behavioral direction, vote, or reason.

7. **Formal voting remains protocol-required, but vote choice belongs to each simulator.**
   - The framework starts a vote and schedules voters.
   - Each simulator policy chooses its vote target from its own ranks, visible discussion state, concerns, concessions, hard constraints, and switch resistance.
   - The resulting vote target is passed to the LLM as a required realization target.
   - `required_vote` is simulator-selected, not controller-selected.

8. **All stochastic behavior must remain reproducible under `simulation.random_seed`.**
   - Use the existing seeded Python random path or introduce one explicit run RNG.
   - Do not scatter untracked random decisions across controller and simulator modules.

9. **Private-information boundaries must be explicit.**
   - A simulator may read its own persona, private goal, option ranks, and private reasons.
   - It may read the shared scenario and accepted public conversation state.
   - It must not use another simulator's private goal, hidden ranks, or hidden reasons when deciding its bid.

10. **Visible accepted text remains the only authority for public state changes.**
    - A bid or intended move never counts as support, concern, concession, answer, comparison, switch, or vote.
    - The observer updates public state only after generation, interpretation, validation, and acceptance.

---

# Part A — Introduce explicit simulator decisions

## [ ] 1. Add a simulator-bid model

### Main files

- `src/models.py`
- `src/simulator.py`

### Add an explicit bid/decision representation

Create a compact model such as:

```python
@dataclass(slots=True)
class SimulatorBid:
    participant_id: str
    wants_to_speak: bool
    willingness: float
    intent: MoveIntent | None
    trigger: str
    action_scores: dict[str, float] = field(default_factory=dict)
```

The exact name may differ, but the model must represent:

- which simulator produced the decision;
- whether it claims the floor;
- a normalized willingness score;
- exactly one complete intended move when it wants to speak;
- a concise trace explanation of the main trigger;
- optional per-act scores for evaluation/debugging.

### Required invariants

- `wants_to_speak=False` implies `intent=None`.
- `wants_to_speak=True` implies a complete `MoveIntent` owned by the same participant.
- `0.0 <= willingness <= 1.0`.
- A bid contains one act, not a menu for the LLM.
- The bid already contains its chosen target, option focus, and participant-specific objective.
- The LLM must not fill missing behavioral choices later.

### Update misleading model comments

Update `MoveIntent` comments so they no longer describe fields such as `required_vote`, `old_preference`, and `allowed_reason` as controller-selected. They are selected by the simulator policy or imposed only by a protocol obligation.

---

## [ ] 2. Build a simulator-specific decision view

### Main files

- `src/simulator.py`
- `src/models.py` only if a small context dataclass is useful

The centralized Python implementation must not accidentally use every participant's hidden state when deciding one participant's behavior.

Create a simulator decision context/view that contains:

- this simulator's `Persona`;
- this simulator's `ParticipantRuntime`;
- the shared `Scenario`;
- recent accepted public turns;
- public option support, public leans, public votes, and current top pair/candidate;
- active public thread information;
- public option coverage;
- the simulator's own turn history and already-used reasons;
- public turn counts and recent-speaker information;
- the current phase and any explicit obligation or protocol request.

Do not include:

- other simulators' private goals;
- other simulators' hidden option ranks;
- other simulators' hidden reasons;
- controller-only predictions about who will switch.

Add a deterministic test showing that changing another participant's hidden goal/ranks while keeping the public transcript fixed does not change this simulator's bid.

---

# Part B — Implement the Python simulator policy

## [ ] 3. Implement one participant-owned decision entry point

### Main file

- `src/simulator.py`

Add a clear public function such as:

```python
def decide_simulator_bid(
    state: DialogueState,
    participant_id: str,
    *,
    obligation: TurnObligation | None = None,
    stimulus: DiscussionStimulus | None = None,
) -> SimulatorBid:
    ...
```

Avoid creating many new files. `src/simulator.py` should become the owner of simulator behavior, not only OCEAN-to-parameter derivation.

The function must support three classes of decisions:

1. open-floor self-selection;
2. constrained responses to a direct obligation;
3. protocol decisions such as opening, narrowing reaction, and voting.

Use small internal helpers, but keep one obvious top-level behavioral entry point.

---

## [ ] 4. Compute participant-local willingness to speak

Willingness is not the same as engagement.

Start with a transparent score composed of a small number of documented factors. Keep constants centralized and avoid scattered ad hoc probability checks.

### Positive factors

- engagement baseline;
- relevance of the last accepted turn to this simulator's current option or concerns;
- this simulator's preferred/current option was challenged;
- a disliked or rejected option gained visible support;
- this simulator's own concern was answered or directly engaged;
- an active question is relevant and this simulator can answer it;
- the simulator has an unused grounded reason or comparison;
- the simulator has been silent relative to its engagement-derived expected share;
- a relevant under-discussed option matches this simulator's own stance;
- the discussion is narrowing around an option this simulator strongly supports or opposes.

### Negative factors

- the simulator spoke on the immediately preceding turn;
- recent over-participation relative to its expected share;
- the intended point would repeat its own recent contribution;
- no new grounded reason, reaction, question, or comparison is available;
- the active issue is irrelevant to this simulator;
- the simulator's contribution would violate the current phase or hard constraints.

### Trait ownership

- `engagement` controls baseline floor-claim frequency.
- `stubbornness` raises reaction pressure when the simulator's stance is challenged and lowers concession/compromise scores.
- `switch_resistance` affects narrowing, compromise, and vote movement, not ordinary speaking frequency.
- `directness` should primarily affect action/response direction and utterance wording, not general turn share.
- `verbosity` must not affect willingness or act selection.

### Claim decision

Use seeded probabilistic self-selection rather than always making every positive score claim the floor. For example:

- calculate willingness in `[0, 1]`;
- sample `wants_to_speak` from that willingness, with an explicit minimum threshold for meaningless bids;
- preserve the numeric willingness for floor arbitration and metrics.

The seeded policy must be reproducible.

---

## [ ] 5. Let each simulator score and choose its own act

### Open-floor action set

For ordinary discussion, support:

- `SUPPORT`
- `CONCERN`
- `ASK`
- `COMPARE`
- `COMMENT`
- `COMPROMISE` when context makes a genuine conditional movement plausible
- silence through `wants_to_speak=False`

`ANSWER`, `OPENING`, and `VOTE` are obligation/protocol acts, not ordinary sampled acts.

`PROCESS` may be available only under a clear stall/no-moderator stimulus. It must not become a generic normal act.

### Action scoring requirements

#### `SUPPORT`

Increase when:

- the simulator's current/preferred option was challenged;
- it has an unused grounded supporting reason;
- another participant raised a concern the simulator can answer from listed facts;
- it wants to reinforce visible movement toward an acceptable option.

The simulator decides whether it defends firmly, acknowledges a limitation, or supports conditionally. The controller must not decide this direction.

#### `CONCERN`

Increase when:

- a disliked/rejected option is gaining visible support;
- a relevant concern remains unanswered;
- a visible claim conflicts with this simulator's stance;
- high stubbornness makes pushback more likely after an incomplete response.

Hard blockers must use their explicit blocker reason and may never soften into acceptance of a rejected option.

#### `ASK`

Increase when:

- a relevant uncertainty remains;
- another participant's visible statement needs clarification;
- a concern could be resolved by a concrete question;
- the simulator lacks enough public information to support or reject a claim.

The simulator chooses whether the question is directed or group-addressed.

A direct addressee may be selected only from visible conversational relevance, such as a participant's prior statement or public vote. Do not target someone based on hidden ranks.

#### `COMPARE`

Increase when:

- two publicly active options are being weighed;
- the simulator has meaningfully different ranks/reasons for those options;
- an active comparison thread exists;
- an under-discussed option is personally relevant to this simulator.

The simulator chooses the pair and trade-off. Coverage must not force a random simulator to compare an option it does not care about.

#### `COMMENT`

Use only for a genuine brief acknowledgement, interpretation, or public-state observation when no stronger action is appropriate.

Do not use `COMMENT` as a controller fallback that overwrites a failed simulator intention.

#### `COMPROMISE`

Increase only when:

- the discussion has a visible candidate or top pair;
- the candidate is not hard rejected;
- the simulator's relevant concern has been addressed or the trade-off is acceptable;
- switch resistance and stubbornness make movement plausible;
- the movement can be explained from visible discussion evidence.

The simulator chooses the condition and direction of compromise. No hidden stance movement occurs from the bid alone.

### Act selection

Use seeded weighted selection among meaningful positive act scores, or a clearly documented max-plus-small-noise strategy. Do not leave the act selection to the utterance LLM.

---

## [ ] 6. Let the simulator choose target, focus, and behavioral objective

After choosing an act, the simulator policy must choose:

- the public turn or active thread it reacts to;
- an addressee when functionally useful;
- one or two valid option IDs;
- the participant-specific direction of the move;
- a concise grounded objective/reason for the LLM.

The objective must be specific enough that the LLM performs realization only.

Good:

```text
Defend Option B against the visible cost concern using B's listed included transfer.
```

Bad:

```text
Respond naturally and move the discussion forward.
```

Do not expose a menu such as “defend or concede.” The simulator policy must decide one direction before generation.

Use this simulator's own private reasons only as behavioral motivation and prompt guidance. They become public evidence only if the accepted utterance expresses them.

---

# Part C — Replace controller routing with floor arbitration

## [ ] 7. Remove participant-behavior authority from `controller/policy.py`

### Main files

- `src/controller/policy.py`
- `src/controller/__init__.py`
- `src/dialogue.py`
- `src/simulator.py`

Prefer renaming `controller/policy.py` to `controller/floor.py` and replacing `PolicyMixin` with a narrowly scoped `FloorMixin`. Do not keep both files.

Remove or move the participant-behavior logic currently owned by these functions:

- `_normal_intent`
- `_choose_speaker`
- `_choose_discussion_act`
- `_choose_target_turn`
- `_focus_options`
- `_reason_for_act`
- `_thread_intent`
- `_maybe_cooling_continuation`
- `_thread_speaker`
- `_maybe_continuation_intent`
- `_speaker_for_option_coverage`
- `_vote_intent`
- `_stance_consistent_vote_target`
- discussion/narrowing/repair helpers that decide for a participant whether to defend, concede, compromise, hold, or switch

Delete dead constants such as the centralized normal-act mapping and old global move weights once no caller uses them.

Keep only genuinely framework-level helpers in the controller/floor layer:

- mandatory-obligation lookup;
- eligibility checks;
- bid validation;
- floor-score adjustment;
- winner selection;
- next-best selection after invalidity or repeated generation failure;
- public candidate/top-pair calculation;
- phase and protocol scheduling helpers that do not decide participant behavior.

---

## [ ] 8. Implement open-floor bidding

Replace the old final routing step with this sequence:

```text
1. Check for a mandatory direct-response or formal protocol obligation.
2. If none exists, build the public stimulus/context.
3. Ask every eligible simulator policy for one bid.
4. Validate all bids structurally.
5. Apply limited floor-access adjustments.
6. Select the highest-scoring valid claiming bid.
7. Generate one utterance for the winning simulator and its unchanged intent.
8. If no valid simulator claims the floor, run stall handling.
```

### Eligible simulators

Normally all participants are eligible, including the previous speaker.

Do not hard-ban the last speaker. Instead:

- apply a strong recent-speaker penalty;
- allow a genuine continuation only when no better bid exists and the simulator has a new additive point;
- cap repeated same-speaker chains.

This more closely models self-selection than the current hard exclusion.

### Floor score

The floor manager may calculate a score from:

- the simulator's submitted willingness;
- recent-speaker penalty;
- anti-monopoly damping;
- minimum-visibility correction for a participant silent far beyond its expected share;
- hard eligibility/phase validity.

Do not add engagement again. It is already inside willingness.

### No content changes

When a bid wins, preserve exactly:

- participant ID;
- act;
- target turn;
- addressee;
- option focus;
- reason/direction;
- vote target or compromise direction.

The floor manager may accept or reject a complete bid. It must not rewrite it.

---

## [ ] 9. Add structural bid validation before LLM generation

Reject a bid before generation when:

- the participant is not eligible;
- the intent speaker does not match the bid owner;
- the act is illegal in the current phase;
- a direct-answer obligation belongs to another simulator;
- the target turn or thread does not exist;
- option focus contains invalid options;
- the addressee is invalid or self-targeted without a valid reason;
- a hard blocker proposes accepting/voting for a rejected option;
- a comparison has fewer than two distinct options;
- an answer does not point to the active question;
- the intent is a clear repetition with no new grounded contribution.

Record the rejection reason. Then consider the next valid submitted bid.

Do not repair an invalid behavioral bid by changing its act or focus in the controller.

---

## [ ] 10. Replace route-failure mutation with bid-preserving recovery

### Current problem

`_adapt_failed_route` may replace the selected speaker, change the act to `COMMENT`, remove the thread, and invent a generic recovery reason. This violates simulator authority.

### Required behavior

For a winning bid whose utterance generation/validation fails:

1. retry or repair the same simulator's same intended move;
2. do not change speaker, act, target, focus, or behavioral direction during that retry;
3. after the bounded retry fails, mark that bid as failed for this turn;
4. choose the next-best previously submitted valid bid;
5. generate that simulator's own unchanged intention;
6. if no valid bids remain, use stall handling.

A generation failure must never silently convert a concern, answer, or comparison into a controller-authored generic comment.

---

# Part D — Preserve conversational obligations without scripting behavior

## [ ] 11. Keep direct questions mandatory but move answer content to the simulator

### Main files

- `src/observer.py`
- `src/simulator.py`
- controller/floor module

Keep `_required_answer_thread` or an equivalent deterministic lookup for direct questions.

Replace `_answer_intent_for_thread` with a simulator-policy decision:

```text
Framework constraint:
- speaker = required respondent
- act = ANSWER
- target = source question turn

Simulator decision:
- answer direction
- relevant option focus
- acceptance, rejection, partial concession, condition, uncertainty, or pushback
- grounded reason
```

The generated answer must still be validated against the target question. Only accepted visible answer evidence cools/resolves the thread.

---

## [ ] 12. Stop assigning group questions to a hidden required respondent

### Main file

- `src/observer.py`

Remove `_pick_group_respondent` and the group-question assignment path.

For a group question:

- create a question thread with `question_scope="group"`;
- keep `required_respondent=None`;
- expose the question as a high-priority public stimulus;
- boost `ANSWER` willingness for simulators with relevant public/own knowledge;
- allow the floor manager to select among their bids.

Update answer observation so a relevant accepted answer from any eligible simulator may cool the group-question thread.

Direct questions still require the named respondent.

---

## [ ] 13. Convert threads from scripts into public stimuli

### Main files

- `src/controller/threads.py`
- `src/observer.py`
- `src/simulator.py`

Keep:

- question/concern/blocker/comparison thread creation;
- issue normalization;
- lifecycle (`hot`, `cooling`, `resolved`, `stale`);
- contribution limits;
- one primary active thread at a time if that remains the project simplification;
- visible-evidence-based resolution.

Remove thread-driven speaker and act prescriptions.

A hot thread should influence participant-local scores:

- concern raiser may push back, acknowledge, ask, or remain silent;
- option advocate may defend, concede part of the concern, ask, or remain silent;
- neutral participant may clarify or compare;
- blocker may restate only when there is new relevant pressure;
- another participant may support either side based on its own stance;
- comparison participants may add a trade-off or decline to speak.

The thread engine must not decide which of those reactions occurs.

---

# Part E — Coverage, stalls, and moderator behavior

## [ ] 14. Stop forcing participants to cover ignored options

### Main files

- controller/floor module
- `src/controller/flow.py`
- `src/simulator.py`

Remove the old coverage route that selects a participant and forces a `COMPARE` move.

Coverage gaps should work as follows:

1. an under-discussed option gives a relevance bonus only to simulators that have a meaningful stance/reason for it;
2. those simulators may self-select a support, concern, ask, or compare bid;
3. if nobody naturally raises it and coverage remains required, the moderator asks the group about that option;
4. the moderator's group question creates a public stimulus; it does not assign a respondent;
5. in no-moderator mode, use the stall-recovery stimulus described below rather than forcing a random participant to compare it.

Redefine `coverage_attempts` clearly as framework prompts/invitations to cover an option, not controller-forced participant turns. Rename it only if the rename materially improves clarity and all logging/tests are updated consistently.

---

## [ ] 15. Implement no-bid and stall handling without reclaiming simulator authority

When no valid simulator claims the floor:

1. check whether the discussion is ready to narrow or vote;
2. if ready, transition normally;
3. if the moderator is enabled and a nudge is allowed, emit a moderator group prompt about the active concern, missing comparison, or coverage gap;
4. do not assign a respondent unless the moderator explicitly names one;
5. if the moderator is disabled, run one second simulator-policy pass with a public `stall` stimulus that raises `ASK`, `PROCESS`, `COMPARE`, or concise `COMMENT` relevance;
6. if there is still no meaningful bid, allow the framework to progress at the configured bounds rather than inventing a participant stance.

A stall is a valid simulated group state. Do not guarantee a participant utterance by overwriting a silence decision.

---

# Part F — Narrowing, voting, compromise, and repair authority

## [ ] 16. Keep narrowing readiness in the framework; move participant reactions to simulators

### Main files

- `src/controller/flow.py`
- `src/simulator.py`

The framework may continue to determine:

- when minimum discussion conditions are met;
- whether hot blocking threads prevent narrowing;
- public candidate/top pair;
- whether the hard turn cap requires progression;
- whether one return to discussion is allowed.

The framework must stop deciding that a selected participant:

- defends the candidate;
- concedes a concern;
- compares a specified pair;
- accepts a trade-off;
- rejects the candidate;
- softens for the group.

Instead, create a narrowing stimulus containing the visible candidate/top pair and ask relevant simulators for bids. A bounded narrowing round may still limit the number of accepted turns, but the reactions themselves come from simulator decisions.

---

## [ ] 17. Move final vote selection into the simulator policy

### Main files

- `src/simulator.py`
- controller/floor module
- `src/controller/flow.py`
- `src/prompts.py`
- `src/models.py`

Implement a simulator-owned vote decision that considers:

- current runtime ranks;
- explicit hard rejections;
- the simulator's current visible lean/acceptance;
- its own unresolved concerns;
- visible candidate support and discussion evidence;
- visible concessions or softenings by that simulator;
- `switch_resistance`;
- hard-blocker constraints;
- the tested candidate or top pair.

The simulator policy returns a `VOTE` intent with:

- exactly one `required_vote`;
- whether this is a visible switch;
- the previous public preference when relevant;
- one grounded allowed reason for the vote/switch.

The LLM must realize that vote clearly. Validation continues to block drift.

Delete controller logic that chooses a participant's final option merely to engineer consensus.

---

## [ ] 18. Refactor majority, split-vote, and deadlock repair into framework prompts plus simulator decisions

### Main files

- `src/controller/flow.py`
- `src/simulator.py`

Keep the framework responsible for:

- detecting unclear votes, majority holdouts, splits, and two-person deadlocks;
- selecting an existing candidate to test from visible votes/evidence;
- bounding the number of repair rounds;
- scheduling a direct question, group prompt, or re-vote protocol;
- computing the final outcome from visible votes.

Move these decisions to the affected simulator policy:

- which reservation it raises;
- whether another participant answers that reservation;
- whether the answer addresses the concern;
- whether the participant holds, softens, conditionally accepts, or switches;
- its final re-vote target.

Remove or relocate controller helpers that currently decide movement, including logic equivalent to:

- `_can_shift_to`
- `_should_switch_after_reservation`
- controller-selected `can_move`
- controller-authored final switch decisions
- controller-selected concession/holdout directions

Hard blockers and rank-1 options remain impossible switches.

Do not fabricate unanimity. Majority and unresolved outcomes remain valid when simulator policies do not move.

---

## [ ] 19. Make peer procedural turns simulator-owned

Current no-moderator helpers select a supporter or “procedural speaker” and prescribe a process move.

Replace that behavior with one of:

- a direct obligation created by a visible participant question; or
- open-floor bids under a `stall`, `narrowing`, or `repair` stimulus.

A participant may choose `PROCESS` to suggest summarizing, narrowing, or asking holdouts, but the framework must not author that participant's social initiative.

The framework still decides whether the proposed process is legally actionable and whether phase gates are satisfied.

---

# Part G — Prompt and realization boundaries

## [ ] 20. Make prompts consume simulator-owned intentions without adding choices

### Main file

- `src/prompts.py`

Keep the compact Voice / Move / Context / Output structure.

The prompt receives:

- the winning simulator persona;
- its own relevant private state;
- the complete simulator-selected intent;
- public target/thread context;
- grounded option cards;
- word budget and surface-style constraints.

Remove wording that still implies controller-selected participant behavior.

The prompt must not ask the LLM to decide among alternatives such as:

- defend or concede;
- accept or reject;
- ask or compare;
- hold or switch;
- choose any final vote.

For direct answers, the prompt must preserve the simulator-selected response direction while requiring relevance to the question.

For votes, the prompt must preserve the simulator-selected `required_vote` and visible switch bridge.

---

# Part H — Tracing, metrics, and evaluation

## [ ] 21. Replace controller-centric traces with authority-aware decision traces

### Main files

- `src/dialogue.py`
- `src/logger.py`
- `src/models.py`

For every participant turn, log:

- authority source: `opening_protocol`, `direct_obligation`, `self_selection`, `narrowing_protocol`, `vote_protocol`, or `repair_protocol`;
- compact bid summary for every eligible simulator:
  - participant ID;
  - `wants_to_speak`;
  - willingness;
  - proposed act;
  - option focus;
  - trigger;
  - structural rejection reason, if any;
- winning bid;
- floor-score adjustments;
- why another bid was rejected or deprioritized;
- whether the winning bid was replaced after generation failure;
- intended act/focus versus realized visible act/focus;
- resulting visible state changes.

Rename `controller_trace` only if all readers, tests, and docs are updated. The important requirement is that the trace no longer presents participant intentions as controller decisions.

---

## [ ] 22. Add autonomy and floor metrics

Extend evaluation/logging with at least:

- self-selected participant turns;
- protocol-forced participant turns;
- direct-answer turns;
- self-selected-turn ratio;
- bid rounds;
- no-bid rounds;
- claim rate per simulator;
- average willingness per simulator;
- floor wins per simulator;
- submitted-act distribution;
- intended-versus-realized act match;
- invalid bid count by reason;
- next-best-bid substitutions;
- direct-question next-turn compliance;
- group-question response rate;
- speaker-chain maximum;
- expected versus realized turn share;
- engagement versus realized claim/win rate;
- repetition and option coverage under self-selection.

Keep the previously required discussion, thread, vote, compromise, switch, trait, grounding, and token metrics.

---

## [ ] 23. Add focused deterministic tests for the authority split

### Required tests

1. Every eligible simulator is asked for an open-floor bid.
2. The winning bid's speaker, act, focus, target, and reason reach generation unchanged.
3. The floor manager cannot replace `CONCERN` with `COMMENT` or alter option focus.
4. The next-best bid is used only after invalidity, ineligibility, or bounded generation failure.
5. Engagement changes long-run claim frequency under fixed repeated conditions.
6. A low-engagement but highly relevant simulator can beat a high-engagement irrelevant simulator.
7. Direct questions force the named respondent on the next participant turn.
8. The forced respondent's answer direction is selected by its simulator policy.
9. Group questions do not receive a controller-selected `required_respondent`.
10. A relevant group-question answer can come from any self-selecting simulator.
11. Concern threads influence bids but do not force an advocate to defend or concede.
12. Coverage gaps do not force a participant `COMPARE` turn.
13. Hard blockers never bid compromise or vote for rejected options.
14. Formal vote target comes from simulator policy.
15. Repair does not directly set a switch or `can_move` result.
16. A failed utterance retry preserves the same intent.
17. Changing another simulator's hidden state does not change this simulator's bid when public state is unchanged.
18. The same random seed reproduces bids, winners, and turn sequence.
19. Different seeds can vary valid bids without violating constraints.
20. No removed centralized routing function remains imported or called.

### Existing suite

Run the full deterministic suite after every major part.

Observed baseline for the supplied project archive with `PYTHONPATH=.:src pytest -q`:

- 273 tests pass;
- one unrelated stale suite-version assertion expects `v7` while the suite declares `v8`.

Do not confuse that stale assertion with a simulator-authority regression. Update it only when the new suite version is intentionally finalized.

---

## [ ] 24. Add/adjust evaluation-suite scenarios

The full evaluation suite must explicitly exercise:

- strongly different engagement levels;
- low-engagement simulator reacting when personally challenged;
- direct question and immediate obligated answer;
- group question with self-selected respondent;
- concern thread where an advocate defends;
- concern thread where a flexible advocate concedes or qualifies;
- comparison thread with a third participant joining voluntarily;
- ignored option becoming relevant naturally;
- ignored option requiring a moderator group prompt;
- all simulators declining the floor once;
- no-moderator stall recovery;
- hard blocker behavior;
- majority holdout;
- split vote;
- two-person deadlock;
- larger group anti-monopoly behavior;
- generation failure followed by next-best bid selection.

Inspect transcripts and traces, not only aggregate outcome counts.

---

# Part I — Cleanup and documentation

## [ ] 25. Remove old routing configuration and dead code

After the new policy is complete:

- remove `routing.move_weights` if no longer used;
- remove old engagement-weighted speaker-selection config that duplicates simulator willingness;
- retain only floor-specific anti-monopoly/recent-speaker/min-visibility settings;
- rename routing config to `self_selection` and/or `floor` where clearer;
- remove obsolete comments describing the controller as owner of speaker/act selection;
- remove dead imports, helpers, route-source values, fallback branches, and tests;
- run pyflakes/static checks;
- do not leave compatibility shims for the old policy.

Keep configuration small. Do not expose dozens of poorly interpretable scoring constants. Prefer a few documented weights or module-level constants with clear behavioral ownership.

---

## [ ] 26. Update all project documentation

### Files

- `README.md`
- `CLAUDE.md`
- `info/00_overview.md`
- `info/02_sim_generation.md`
- `info/03_routing_and_turn_taking.md`
- `info/04_moderator.md`
- `info/05_discussion_and_decision.md`
- `info/06_consensus_and_outcomes.md`
- `info/07_evaluation_and_logging.md`
- `info/08_configuration_and_running.md`

Document the final architecture accurately:

```text
simulator policy -> bid and intended move
floor manager     -> turn-access arbitration
LLM               -> utterance realization
observer          -> visible-evidence state update
flow              -> phases, protocols, and termination
```

Explicitly state:

- ordinary participant behavior is simulator-driven;
- direct questions and formal votes are protocol obligations;
- group questions use self-selection;
- engagement affects willingness;
- floor arbitration does not choose participant content;
- threads are public stimuli, not scripts;
- visible text remains the only public evidence;
- the implementation is a controlled hybrid simulator, not unrestricted autonomous agents.

---

# Final acceptance criteria

The refactor is complete only when all of the following are true.

## Authority

- The global framework no longer chooses ordinary participant acts, targets, option focus, reasons, concessions, compromises, or vote targets.
- Every open-floor participant turn originates from a complete simulator bid.
- Direct questions constrain the respondent and answer act but not the answer's behavioral direction.
- Group questions are answered through self-selection.
- Opening and voting are protocol-required, with participant substance chosen by simulator policy.

## Floor management

- The floor manager chooses among complete bids.
- Engagement is not double-counted.
- Relevance can outweigh engagement.
- Anti-monopoly rules affect floor access only.
- Invalid or failed bids may be skipped, never rewritten.

## Behavioral correctness

- Traits produce measurable effects without becoming deterministic scripts.
- High engagement increases claim/win rate on average.
- Stubbornness increases defense/pushback and lowers concession probability.
- Switch resistance lowers compromise and vote switching.
- Directness affects response direction/wording without changing turn share.
- Verbosity affects utterance length only.
- Hard blockers remain rare and enforced.

## Conversation quality

- Direct questions are answered promptly.
- Local threads remain coherent without dictating reactions.
- Option coverage remains sufficient without forced participant comparisons.
- Discussions can stall naturally and recover structurally.
- Majority and unresolved outcomes remain legitimate; consensus is not engineered by controller-selected switches.
- Generated utterances visibly match the winning simulator intention.

## Engineering quality

- No parallel old router remains.
- No participant policy depends on another simulator's hidden state.
- Seeded runs are reproducible.
- Deterministic tests pass.
- Evaluation traces make the authority split inspectable.
- Documentation matches the implemented behavior.
