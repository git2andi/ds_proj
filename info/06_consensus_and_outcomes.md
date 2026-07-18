# Narrowing, voting, and outcomes

Narrowing occurs before the formal vote and uses only public preferences and visible acceptances.

- If an option already has a strict public-preference majority, the runtime proceeds directly to voting.
- Otherwise, support counts include each participant who currently prefers or has visibly accepted an option.
- The highest support count determines the target; current preference counts break a support tie.
- If several options remain tied, the seeded runtime selects one tied option as the single compromise target.

The moderator names the holdouts and asks whether the target fits their requirements. At most two holdout turns are available by default. A participant may accept only when the target has private rank 4 or 5 and is not hard rejected. Rank-4 participants with stubbornness 1--3 accept; stubbornness 4 remains probabilistic. Hard blockers and ranks 1--3 do not move. Participants who do not accept may state one grounded concern or remain silent.

A previously visible acceptance of the target carries into narrowing. A positive narrowing acceptance is committed as a public acceptance and becomes a switch to that target in the participant's final vote.

Every participant then casts one explicit final vote. Vote wording is deterministic. A participant normally votes for the current preferred option, or for the unique narrowing target when it was publicly accepted. The outcome calculation returns:

- `successful` when all participants vote for the same option;
- `majority` when one option reaches the strict majority threshold;
- `unresolved` otherwise.

There is one final voting round and no routine re-vote.
