# 03 — Turn-taking: self-selection and floor arbitration

Turn-taking is **simulator-driven**. There is no centralized router that picks a
speaker and authors a move. Instead, when the floor is open, every eligible
simulator independently decides whether it wants to speak and, if so, submits one
complete intended move; a floor manager arbitrates access and selects a winner
without rewriting it. Bidding and arbitration are read-only over dialogue state
— effects only count after the winning utterance's validated visible evidence is
observed (the observer consumes exactly the evidence object that passed
validation; it never reparses text).

## Open-floor bidding

Per ordinary turn (`controller/flow.py::_run_open_floor_turn`):

```text
1. build a public DiscussionStimulus (candidate, top pair, coverage gap, open group question, kind)
2. ask every eligible simulator policy for one SimulatorBid (src/simulator.py)
3. validate bids structurally (controller/floor.py::_validate_bid)
4. score floor access and select the highest-scoring valid claiming bid
5. generate one utterance for the winning bid's unchanged intent
6. on bounded generation failure, use the next-best submitted valid bid
7. if no valid simulator claims the floor, run stall handling
```

A simulator bid is `wants_to_speak` + a normalized `willingness` + (when
claiming) one complete `MoveIntent` owning act, target turn/thread, addressee,
option focus, direction, reason, and any vote/compromise. `wants_to_speak=False`
is a legitimate simulated silence and is never overwritten.

## Willingness and act scoring (per simulator)

Each simulator computes its own **willingness** from its persona, private stance,
and public state (`src/simulator.py::_willingness`): engagement is the baseline,
plus stake/relevance factors (its option was challenged, a disliked option gained
visible support, its concern was engaged, an answerable group question, an unused
grounded reason, being under its expected share, a stake in the narrowing
candidate) minus damping (spoke last, over its share, repetition). Relevance and
personal stake can outweigh engagement, so a low-engagement simulator whose
preferred option was challenged can beat a highly engaged one with nothing new to
add. `wants_to_speak` is sampled from willingness (seeded).

Act scores (`_score_acts`) measure content availability, never turn frequency:
`SUPPORT` rises when the sim's option is challenged or it has an unused reason;
`CONCERN` when a disliked/rejected option gains support (scaled by stubbornness
and directness); `ASK` when a relevant uncertainty is open; `COMPARE` when two
active options are ranked differently; `COMMENT` is a weak baseline;
`COMPROMISE` only when a non-rejected candidate has visible backing and movement
is plausible; `PROCESS` only under an explicit stall stimulus. The act is chosen
by seeded weighted selection among the positive scores, and the simulator then
picks its own target/focus/reason/addressee.

## Floor arbitration (access only)

The floor manager (`controller/floor.py`) may reject or reorder complete bids but
never rewrites an act, focus, target, addressee, reason, or vote. Its floor score
starts from the submitted willingness and applies **only** floor mechanics:
recent-speaker penalty, anti-monopoly damping when a sim is past its expected
share, and a minimum-visibility boost when a sim has been silent beyond its
expected share. Engagement is not applied a second time (it is already inside
willingness). The last speaker is not hard-banned — a strong recent-speaker
penalty plus a speaker-chain cap models self-selection instead. Structural
validation rejects a bid before generation when the intent speaker mismatches,
the act is illegal in the phase, a referenced turn/thread is missing, option
focus is invalid, the addressee is invalid/self, a comparison has fewer than two
options, a hard blocker targets a rejected option, or the bid is a clear
repetition. A rejected or failed bid is skipped, never rewritten; the next-best
valid bid is used.

`engagement` is the only participation-share parameter: each sim gets an expected
share (`0.30 + engagement`, normalized), used inside willingness and the
floor's anti-monopoly/min-visibility corrections. Age/speech_style are never a
floor signal.

## Protocol obligations

Some turns are protocol-required. A `TurnObligation` fixes only the speaker and
act; the simulator still chooses the substance:

- **opening** — every sim speaks once; the simulator chooses its opening option
  focus and reason from its own stance.
- **direct answer** — a valid direct question gives its named respondent the next
  turn (act `ANSWER`); the simulator decides the answer's direction, focus, and
  grounded reason (accept, reject, partial concession, condition, uncertainty, or
  pushback). The controller never prescribes "accept/reject/defend/concede".
- **vote** — the framework starts a vote and schedules voters; each simulator
  selects its own `required_vote` from its ranks, visible lean/concessions,
  switch resistance, hard constraints, and the tested candidate, and whether it
  is a visible switch with a grounded reason.
- **narrowing** — the framework creates a public group stimulus; relevant simulators self-select a response or remain silent.
- **repair reactions** — bounded reservation/re-vote obligations fix only the participant and broad act; the simulator chooses the reservation and stay/switch substance.

## Questions

Question *scope* comes from validated visible evidence only: a named or
"you"-directed question is direct; a genuine question without an addressee is a
**group** question with no assigned respondent. A direct question is a mandatory
adjacency pair (its named respondent owes the next turn). A group question opens
a public question thread with `required_respondent=None`, becomes a high-priority
stimulus that raises `ANSWER` willingness for relevant simulators, and may be
answered by any self-selecting simulator other than the asker; a relevant
accepted answer from any of them cools the thread.

## Threads as stimuli

Local interaction is tracked as threads (`question`, `concern`, `blocker`,
`comparison`) with statuses `hot / cooling / resolved / stale`, identity
`(type, focus options, deterministic issue key)`, and per-thread contribution
caps. The engine in `controller/threads.py` owns all lifecycle transitions but no
longer selects a "primary thread" or prescribes a speaker/act. A hot thread is a
**public stimulus**: it raises the relevant participant-local scores in each
simulator's bid (a concern raiser may push back, an advocate may defend or
concede, a bystander may clarify or compare, a blocker may restate under new
pressure — or any of them may stay silent). Each simulator picks which hot thread
it reacts to; a thread past its contribution cap stops being a live stimulus.

## Coverage and stalls

An under-discussed option is a relevance bonus to simulators that have a real
stance on it, never a forced comparison. If it stays uncovered, the moderator
asks the group about it (a group question with no assigned respondent); with the
moderator off, a `stall` stimulus raises ask/compare/comment/process relevance
for one more bid pass. A stall is a valid simulated group state: if no simulator
claims the floor, the framework progresses at its configured bounds rather than
inventing a participant stance.

## Macro acts

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Open-floor self-selection can produce `answer, support, concern, ask, compare, comment, compromise` when a concrete contribution is available; `process` appears only under a stall stimulus. COMMENT has no generic baseline, and silence is valid.
`opening`, `answer`, and `vote` are obligation/protocol acts whose substance the
simulator still owns; softening is an observed stance effect parsed from visible
text, never a chosen act.

## Parameter influence

- engagement -> willingness baseline / expected turn share (applied once, inside the simulator);
- verbosity -> average utterance length (numeric word budgets, soft targets); never affects willingness or act choice;
- directness -> wording bluntness and a higher concern/challenge prior and directed-question tendency; never turn share;
- stubbornness -> discussion-phase defense and concession probability — never final switching;
- switch_resistance -> final movement only: switches, compromise acceptance, holdout concession, vote/repair resistance;
- speech_style -> lexical and register variation, never a turn-taking signal.


## Public social awareness

Each simulator receives a derived public participant ledger: names, latest visible positions, visible support/concerns, recent acts, and active question relationships. It is rebuilt from accepted turns and contains no other simulator's private goal, ranks, reasons, stubbornness, or switch resistance. Social targeting uses this ledger so questions and responses prefer the visible owner of the relevant claim or concern.
