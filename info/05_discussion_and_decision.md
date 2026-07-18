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

The structured action determines intended behavior, but preference and acceptance changes are committed only when accepted text visibly realizes them. Explicit aliases are required for context-setting actions. Local reactions and answers may inherit a unique visible focus from the immediately preceding turn or active thread. The runtime does not attempt general semantic parsing.

## Repetition control

Grounded reasons carry a point key based on option ID and attribute name. The runtime records group-level point counts and the two most recent points.

- a participant avoids reusing its completed point;
- a recently used point is suppressed for ordinary contributions;
- an already public point cannot open another question;
- a completed thread is not reopened on the same key;
- after the first thread answer, later contributions must add another point, compare, or move;
- narrowing may reuse a point when it directly explains a final movement decision.

## Language realization

The prompt includes the selected action, grounded source, relevant board facts, active thread, recent dialogue, and the speaker’s recent sentence openings. It asks the speaker to connect to the previous message, vary the sentence opening, avoid copying recent structure, and place option references naturally.

## Generation failure

Voluntary invalid utterances are dropped and flagged. Openings, required answers, and votes receive one repair attempt. A deterministic opening is used only after generation and repair both fail. Required-answer and vote fallbacks use natural text because the protocol cannot safely continue without them.
