# Discussion and decision

**Code:** `src/dialogue.py` (`run`, `_opening_round`, `_discussion_loop`,
`_decision_loop`, the generation+repair pipeline), `src/validation.py`
(`ValidationMixin`), `src/consensus.py`.

This note follows a run from the first message to the close, and explains the
generation pipeline that turns each `MoveIntent` (from `03`) into one safe printed
line.

## Phases

A run moves through soft control states (they react to the transcript, they are not a
rigid script). They are recorded in `phase_history` (`07`):

```text
opening    -> discussion -> narrowing -> closure
```

- **Opening** — every sim states its initial favorite and one grounded reason, no
  final vote yet (`_opening_round`).
- **Discussion** — the routing loop from `03` runs: comparisons, challenges,
  questions, answers, invitations, compromise probes.
- **Narrowing** — the controller judges the group is ready to decide (see below) and
  moves into vote rounds.
- **Closure** — recorded **only** when the outcome is actually resolved.

## When narrowing is allowed (vote readiness)

`_ready_for_vote` never narrows on a fixed turn count alone. It requires **visible
evidence**: after a minimum discussion length, a visible support cluster (or visible
support plus a visibly proposed compromise) for a candidate, with no open question and
no active blocker on it. Latent (hidden) lean never triggers voting. A hard turn cap
still forces a visible vote so runs terminate.

## The decision loop

`_decision_loop` runs up to `conversation.max_vote_rounds`:

1. Clear any owed direct question first.
2. Pick the vote candidate from visible evidence (`_candidate_for_vote`: votes and
   acceptances weigh double, visible proposals count once; latent lean only breaks
   ties).
3. The moderator calls the vote (if enabled, `04`); each sim casts a visible vote.
4. Compute the provisional outcome (`06`). A `majority` gets one **minority-check**
   beat; a `successful`/`majority` result closes.
5. If everyone has voted but no majority formed, the loop records an **intermediate
   `narrowing`** marker (not a false closure — issue 6) and falls through to one
   bounded **split-vote compromise** pass. Only a resolved outcome marks `closure`.

Round 0 asks everyone; later rounds re-prompt only unclear/non-voters. A sim that has
already cast a clear vote is not asked again, and its vote is not silently overwritten
(`06`).

## Bridged preference switches (issue 5)

A sim may legitimately move its vote — but a **switch away from its current internal
lean must be socially explained in the text**, not just legal at the state level. The
validator enforces a bridge clause: the line must name the old option (or an explicit
concession) **and** give a reason. The new option is the vote itself.

```text
Unbridged (blocked): "My pick is Youth Arts because it supports creative education."
Bridged (allowed):   "I still like Senior Safety, but I'll go with Youth Arts because
                      it's easier for the whole group to support."
```

A missing bridge is the blocking issue `UNBRIDGED_SWITCH`; it is repaired (the repair
prompt names the old pick) and, if repair still fails, the deterministic fallback
**restates the current lean** rather than printing an unexplained flip. Switches are
recorded as `switch_events` with `has_reason` and `has_bridge` (`07`).

## The generation + repair pipeline

Each participant turn (`_generate_and_append`) is not just "call the LLM and print":

```text
1. build the per-turn prompt from the MoveIntent + voice capsule + option facts
2. call the LLM  ->  clean_generated trims to a complete sentence within the budget
3. parse the line into a DialogueAct (observer, see 06)
4. validate + ground-check the line (ValidationMixin)
5. if there are issues: run up to `max_repairs_per_turn` repair passes
6. if a BLOCKING issue still survives -> replace with a deterministic fallback line
7. only then: append, print, and apply semantics to the visible state
```

Steps 4–6 are what guarantee no invalid line reaches the transcript.

## Validation (what can block a turn)

`ValidationMixin._validate_turn_text` flags issues; some are **blocking** (the line
must not be printed as-is):

```text
EMPTY / MULTI_TURN_OUTPUT / LEAKED_METADATA / INVALID_OPTION_REFERENCE
MISSING_REQUIRED_OPTION_FOCUS       a coverage turn didn't name the required option
UNCLEAR_VISIBLE_COMMITMENT          a vote/accept turn produced no clear commitment
HARD_BLOCKER_ACCEPTED_REJECTED_OPTION   a blocker tried to accept its rejected option
BLOCKED_OPTION_ACCEPTED             committing to an actively blocked option
OFF_TARGET_SWITCH                   a sanctioned switch landed on a disallowed option
UNBRIDGED_SWITCH                    a switch with no bridge clause (issue 5)
UNSUPPORTED_FACT                    grounding judge flagged an invented fact (non-blocking)
```

## Grounding (no invented facts)

Grounding keeps sims inside the option board (`01`). To stay cheap it runs in
**tripwire** mode (`validation.grounding_mode`, default): the LLM fact-judge
(`prompts.grounding_check`) is called only when a regex tripwire finds a suspicious
concrete claim — a number or policy/medical/weather-style term absent from the world,
or a **cross-option fact transfer** (a line naming option X while using another card's
distinctive tokens). The judge flags invented facts, wrong-option attribution, and
unlike-unit comparisons. `always` mode judges every turn.

## Deterministic fallbacks

If a blocking issue survives repair, `_safe_fallback_text` substitutes a
parser-clean line for the intent (a hard blocker commits to an allowed alternative;
an unclear vote becomes one clear commitment; a switch restates the current lean; a
coverage turn names the required option). It is **restate-first** — it never
fabricates consent. Metrics track `fallback_turns` and `invalid_printed_turn_count`
(which must stay 0).

## Bounded unresolved

`unresolved` is a valid, honest outcome — but only after a clear process: split
preferences, one compromise attempt where appropriate, and no visible majority. It is
never a silent give-up.
