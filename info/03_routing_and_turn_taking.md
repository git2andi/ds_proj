# Routing and turn taking

## Autonomous bidding

On an open floor, every simulator constructs currently valid candidate actions:

- `SUPPORT`: add a grounded reason for the current option;
- `REACT`: respond to another visible position;
- `OBJECT`: raise a grounded concern;
- `COMPARE`: contrast relevant options;
- `ASK`: ask about another participant’s visible option or trade-off;
- `ACCEPT`: make another option acceptable or switch when allowed.

The simulator selects one candidate using simple contextual weights, then engagement determines whether it submits that action or remains silent. The selected bid is complete: speaker, act, option focus, grounded source, optional addressee, and optional movement. A comparison contains the same named public attribute from both focused options. Comparisons use a lower weight, are not available as a participant’s first voluntary contribution, and are suppressed when another comparison occurred in the preceding two participant turns.

## Floor selection

The floor receives intact bids and applies required-answer priority, thread priority, a consecutive-turn bound, and light anti-monopoly/anti-ping-pong weighting. It never rewrites a bid.

## Threads

A direct question schedules the named participant’s answer. A group question allows an eligible participant to answer. After the first response, simulators may voluntarily agree, disagree, add a new grounded point, compare, or accept. The thread closes after no related bid or its configured turn cap.

Questions are not generated merely to enumerate attributes. A question point must not already be public or belong to a closed thread. Later thread turns may not repeatedly paraphrase the opening point; after the first answer they must add another point, compare, or visibly move.

## Local references

Openings, preference movements, and votes explicitly identify the option. Ordinary discussion turns are encouraged to name an option when clarity needs it, but a missing exact alias is not a hard failure. This permits natural continuation without requiring a general fuzzy-matching subsystem.

`ACCEPT` can either make another option acceptable or switch the current preference. A mid-discussion switch is allowed only after the option has appeared in the recent public exchange, already has more public support than the participant's current choice, and the participant's stubbornness-based movement draw succeeds.
