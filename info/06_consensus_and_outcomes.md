# Narrowing, voting, and outcomes

Narrowing uses public participant preferences rather than hidden ranks and occurs before the formal vote.

- Unanimity proceeds directly to voting.
- A decisive majority proceeds directly to voting.
- A narrow 2–1 or 3–2 majority receives one bounded opportunity for outliers to accept the leader.
- A split without a majority receives one short compromise window.

During compromise, the moderator asks one grammatically appropriate narrowing question and then the floor runs up to two movement-bid rounds. Simulators independently decide whether a visible alternative is acceptable; unwilling participants remain silent and no response is forced. The moderator does not narrate the absence or success of movement. Hard blockers never move. Smaller grounded preference switches may already occur during discussion when they follow recent public support and can help the group converge.

Every participant then produces one explicit structured final vote. Its short visible wording is deterministic, so voting adds no LLM generation or repair calls. The deterministic outcome calculation returns:

- `successful` for unanimity;
- `majority` when an option exceeds the configured majority fraction;
- `unresolved` otherwise.

There is no routine second voting round. This keeps the protocol concise and prevents fixed vote lines from dominating short discussions.
