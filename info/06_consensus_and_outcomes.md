# Narrowing, voting, and outcomes

Narrowing uses public participant preferences rather than hidden ranks and occurs before the formal vote.

- Unanimity proceeds directly to voting.
- A decisive majority proceeds directly to voting.
- A narrow 2–1 or 3–2 majority receives one bounded opportunity for outliers to accept the leader.
- A split without a majority receives one short compromise window.

During compromise, simulators independently decide whether a visible alternative is acceptable. The controller does not force movement. Hard blockers never move.

Every participant then produces one explicit structured final vote, realized as visible text. The deterministic outcome calculation returns:

- `successful` for unanimity;
- `majority` when an option exceeds the configured majority fraction;
- `unresolved` otherwise.

There is no routine second voting round. This keeps the protocol concise and prevents fixed vote lines from dominating short discussions.
