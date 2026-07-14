# Routing and turn taking

Every eligible simulator creates either silence or one complete `UserAction`.

There is no numeric urgency. Bids use categorical priority:

1. required direct answer;
2. concern-owner reaction after a response;
3. active-issue or moderator-stimulus response;
4. ordinary voluntary contribution.

The floor selects randomly within the highest non-empty category using the run seed. It prefers a different speaker when possible and enforces the configured maximum consecutive turns.

Engagement affects whether a voluntary bid exists. The floor does not equalize participation or use expected shares.

A direct question creates a next-turn obligation for the addressee. It uses one configured semantic mode—choice impact, trade-off, or an optional condition—and asks about a concrete concern rather than requesting the opening rationale again. After the addressed participant answers, the question issue closes; other participants do not repeat answers to the same direct question.

When a concern receives a response, the participant who raised it gets the next reaction opportunity. That reaction may accept the trade-off, partially soften, or maintain the concern. Moderator coverage/stall stimuli are handled before unrelated ordinary actions, so visible nudges are not silently ignored.


When the ordinary floor stalls after the minimum discussion budget, the environment exposes at most one compromise window. Eligible non-hard-blockers independently decide whether to propose common ground. The floor selects among actual proposals and never creates one. The moderator prompt is appended only after a selected compromise contribution has been successfully realized, so failed generation cannot leave a visible nudge followed by silence.
