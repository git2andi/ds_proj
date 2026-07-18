# Moderator

The moderator is deterministic and does not have a persona or independent agenda. Its role is limited to protocol facilitation.

It may:

1. open the decision and present the option board;
2. provide one neutral liveness prompt when the floor is empty too early;
3. announce a decisive lead or request one bounded compromise round;
4. request explicit final votes;
5. state the computed outcome.

Consecutive moderator messages are coalesced where possible. A moderator question is emitted only when the runtime will allow a participant response. The moderator uses public protocol state and does not invent participant reasons or hidden preferences.

Moderator mode can be disabled. The participant floor and deterministic voting still operate without it.
