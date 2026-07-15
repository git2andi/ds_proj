# Routing and turn taking

Every eligible simulator creates either silence or one complete `UserAction`.

There is no numeric urgency. Bids use categorical priority:

1. required direct answer;
2. active-concern or moderator-stimulus response;
3. ordinary voluntary contribution.

Concern owners and third parties use the same voluntary active-issue category. The floor does not force an owner reaction or choose a participant merely to resolve a concern.

The floor selects randomly within the highest non-empty category using the run seed. It prefers a different speaker when possible and enforces the configured maximum consecutive turns.

Engagement affects whether a voluntary bid exists. The floor does not equalize participation or use expected shares.

A direct question must clearly name its addressee in a natural vocative position—at the beginning, middle, or end—and creates a next-turn obligation. The addressee must answer next, but its simulator owns the answer content and any resulting stance effect. After the mandatory answer, the runtime reuses ordinary reaction logic: at most one simulator may voluntarily react when the answer genuinely triggers a novel contribution. The required answerer cannot answer the same question repeatedly, and a global addressee/option/concern key prevents another simulator from reopening that same question. If no natural reaction exists or nobody bids, the answered question closes as resolved.

A rare condition mode asks whether the provided information states how a concrete concern would be handled. The addressed simulator answers that the information is insufficient instead of inventing a solution.

Concern opening and concern responses are voluntary. Up to two distinct non-owner simulators may respond. The owner may then voluntarily accept the trade-off, partially soften, maintain the concern, or stay silent. The same semantic concern is recorded globally and cannot be opened again by another participant during ordinary discussion. One narrowing-time reopening is allowed when it still blocks agreement. If nobody continues, the unresolved concern becomes stale.

Soft coverage is also engagement-gated. The environment offers the uncovered option as a possible response focus, but it does not force any participant to comment. If nobody bids, the option is recorded as receiving no expressed interest and the discussion continues without a fabricated response.


When the ordinary floor stalls after the minimum discussion budget, the environment exposes at most one compromise window. Eligible non-hard-blockers independently decide whether to propose common ground. The floor selects among actual proposals and never creates one. The moderator prompt is appended only after a selected compromise contribution has been successfully realized, so failed generation cannot leave a visible nudge followed by silence.

For a complete public split during narrowing, the moderator states the split once even when nobody moves. Only options that at least one participant publicly prefers or accepts are named and offered as compromise material. An option that was merely uncovered, mentioned, or criticized cannot become a compromise candidate.
