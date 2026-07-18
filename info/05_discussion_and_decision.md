# Discussion and decision process

## Discussion loop

The runtime repeatedly performs:

```text
required direct answer, if any
otherwise collect simulator bids
→ select one intact bid
→ realize one utterance
→ hard validation
→ commit visible state and public point history
→ maintain or close the active thread
```

The loop has minimum, soft-target, and hard-maximum voluntary-turn budgets. It may stop after the target when no novel bid remains, or after minimum participation when public preferences have converged. One neutral liveness intervention is available before the target.

## Public state

The structured action determines intended behavior. Preference changes are committed only when the accepted text visibly identifies the target option. For ordinary discussion acts, omission of an exact alias is treated as a minor wording issue rather than a reason to discard the turn. The runtime does not attempt general semantic parsing.

## Repetition control

Grounded reasons carry a point key based on option ID and attribute name. The runtime records group-level point counts and the two most recent points.

- a participant avoids reusing its completed point;
- a recently used point is suppressed for ordinary contributions;
- an already public point cannot open another question;
- a completed thread is not reopened on the same key;
- after the first thread answer, later contributions must add another point, compare, or move;
- narrowing may reuse a point when it directly explains a final movement decision.

## Language realization

The compact prompt includes the selected action, exact grounded source, active thread, four recent turns, and the speaker’s recent sentence openings. It asks the speaker to continue a live group chat, connect to the previous message, vary syntax with the persona style, and avoid routinely beginning with an option name, participant name, or `I`. It asks for an option reference when clarity needs one, while allowing contextual continuation for ordinary discussion. Comparisons receive the same named public attribute from both options. The values are separately labeled, but the utterance may express the difference naturally rather than through one fixed template.

## Generation failure

Voluntary invalid utterances are dropped and flagged. Only openings receive one repair attempt and a deterministic fallback. Required answers keep the original generated wording and are not subjected to a semantic-relevance score, repair call, or generic fallback. They still pass the ordinary hard checks for usable output, supported numbers, and hard-blocker consistency. Missing an exact option alias is not by itself a hard failure for an answer.

For a structured movement, visibility requires the target option to be explicitly named. The simulator has already made the acceptance or switching decision, so the validator does not require a fixed acceptance phrase.
