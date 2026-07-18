# Discussion and decision process

## Discussion loop

The runtime repeatedly performs:

```text
required direct answer, if any
otherwise collect simulator bids
→ select one intact bid
→ realize one utterance
→ deterministic validation
→ commit visible state and point history
→ maintain or close the active thread
```

For `n` participants, the default voluntary-turn budgets are:

```text
minimum = 2n
target  = min(4n, 20)
maximum = min(6n, 28)
```

Discussion may stop after the minimum when public preferences have converged, after the target when no novel candidate remains, or at the hard maximum after any outstanding required answer. After the configured number of empty bidding rounds, one moderator prompt and one forced simulator-owned bid may be used for liveness.

## Public state and repetition control

The structured action determines the intended behavior, but its effects are committed only after the visible text passes validation. Failed turns do not enter the transcript, consume a reason, change a preference, register an acceptance, advance a thread, or create a vote.

Grounded reasons carry point keys based on option ID and public source. The runtime tracks per-participant use, group-level counts, recent points, and closed thread points. This prevents exact reuse where alternatives exist, but it is not a general semantic duplicate detector.

## Language realization

The LLM receives the selected action, grounded source, persona voice, active thread, relevant option facts, and the configured recent-turn window. The prompt also includes recent sentence openings to discourage repetitive syntax. The LLM controls wording only; it does not choose the speaker, action, option, movement, or vote.

Openings, explicit movements, and votes must identify their option. Ordinary reactions and answers may use local contextual references when the active exchange is unambiguous. Comparisons receive one grounded source for each focused option.

## Validation and failure handling

Before realization, structured actions are checked for valid speakers, addressees, options, comparison sources, movement, and vote constraints. After realization, deterministic checks reject unusable output, invalid speaker labels, unsupported numeric claims, malformed direct questions, invisible movement, hard-blocker contradictions, and vote mismatches.

Only an invalid opening receives one focused LLM repair attempt and may then use a deterministic fallback. Invalid ordinary discussion turns, including answers, are dropped and logged. A failed required answer is recorded as a protocol error. Final vote wording is deterministic and uses no dialogue-generation or repair call.
