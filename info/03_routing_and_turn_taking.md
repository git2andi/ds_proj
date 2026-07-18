# Routing and turn taking

## Autonomous bidding

On an open floor, every simulator constructs currently valid candidate actions:

- `SUPPORT`: add a grounded reason for the current option;
- `REACT`: respond to another visible position;
- `OBJECT`: raise a grounded concern;
- `COMPARE`: contrast relevant options;
- `ASK`: ask about another participant’s visible option or trade-off;
- `ACCEPT`: make another option acceptable or switch when allowed.

The simulator selects one candidate using simple contextual weights, then engagement determines whether it submits that action or remains silent. The selected bid is complete: speaker, act, option focus, reason source, optional addressee, and optional movement.

## Floor selection

The floor receives intact bids and applies required-answer priority, thread priority, a consecutive-turn bound, and light anti-monopoly/anti-ping-pong weighting. It never rewrites a bid.

## Threads

A direct question schedules the named participant’s answer. A group question allows an eligible participant to answer. After the first response, simulators may voluntarily agree, disagree, add a new grounded point, compare, or accept. The thread closes after no related bid or its configured turn cap.

Questions are not generated merely to enumerate attributes. A question point must not already be public or belong to a closed thread. Later thread turns may not repeatedly paraphrase the opening point; after the first answer they must add another point, compare, or visibly move.

## Local references

Openings, new questions, comparisons, movements, and votes explicitly identify the option. Reactions and answers may omit a repeated name only when the active thread or immediately preceding accepted turn gives one unique option focus. This permits natural continuation without general fuzzy matching.
