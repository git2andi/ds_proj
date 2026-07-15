# Consensus and outcomes

Every participant states one clear visible choice. Natural forms such as `Lab gets my vote`, `I’m going with Riverside`, `I’ll stick with Online`, and the contextual `Library for me` are valid; the word `formal` is unnecessary. In the voting phase, exactly one visible intended option with no competitor is sufficient.

Outcomes:

- `successful`: every participant selects the same option;
- `majority`: one option reaches the majority threshold;
- `unresolved`: no option reaches a majority and no further visible movement produces a viable second vote.

A valid majority closes immediately. The environment does not pressure holdouts toward unanimity.

A non-blocker may move after a concrete trigger: a concern response made the trade-off acceptable, stagnation exposed an eligible alternative, or narrowing identified viable common ground. Rank 3 may be considered directly; rank 2 requires that participant's own concern to have been resolved or softened first; rank 1 is never eligible. The configured `movement_probability_by_stubbornness` controls whether the simulator takes the opportunity. A hard blocker never moves.

Each accepted movement records a grounded reason and, when relevant, the remaining concern. A first acceptance also preserves the participant's previous priority, for example: `I still prefer the Lab for its equipment, but I can accept the Library for the quieter setting.` A later vote can be short because the reason is already public; if the vote is the first visible movement, it must include the rationale. Selected movement is never silently dropped after realization failure.

A clear choice for a new option is itself visible movement. A short bridge is encouraged for readability but validation does not discard an otherwise unambiguous changed vote.

A second vote occurs only when the first vote has no majority and the intervening re-narrowing produces at least one visible acceptance or preference switch. If nobody moves, the system closes unresolved instead of repeating the same votes.
