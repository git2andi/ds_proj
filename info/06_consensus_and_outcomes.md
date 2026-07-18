# Narrowing, voting, and outcomes

Narrowing uses public participant preferences rather than hidden ranks and occurs before the formal vote.

- Unanimity or any strict public-preference majority proceeds directly to voting.
- Without a majority, public preferences and visible acceptances may identify one uniquely strongest option.
- If several strongest options remain tied, the seeded runtime randomly selects one tied option as the single compromise target, including complete preference splits.

When one leader exists, the moderator names the participants who still prefer something else and asks whether the leader fits their requirements. The floor then allows up to two holdout responses. A previously visible acceptance of that leader remains valid and carries into the final vote without a second random decision. Otherwise, rank-4 holdouts with low or medium stubbornness normally accept, highly stubborn rank-4 holdouts remain probabilistic, rank-5 holdouts accept, and ranks 1–3 or hard blockers do not move. Participants who do not accept may reject the leader with a remaining concern or remain silent. A positive acceptance makes that participant eligible to switch to the same leader in the final vote; no movement toward an opposing narrowing target is allowed. The moderator does not narrate the absence or success of movement. Smaller grounded preference switches may already occur during ordinary discussion when they follow recent public support and can help the group converge.

Every participant then produces one explicit structured final vote. Its short visible wording is deterministic, so voting adds no LLM generation or repair calls. The deterministic outcome calculation returns:

- `successful` for unanimity;
- `majority` when an option exceeds the configured majority fraction;
- `unresolved` otherwise.

There is no routine second voting round. This keeps the protocol concise and prevents fixed vote lines from dominating short discussions.
