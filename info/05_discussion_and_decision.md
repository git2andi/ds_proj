# Discussion and decision process

## Opening

Each participant must visibly state a current preferred option and one reason. The first speaker receives an `INITIAL` opening mode; later speakers receive `ALIGN` or `CONTRAST` according to already-visible preferences. A greeting is required only for the first opening and remains optional afterward.

Mandatory openings retry and then fail clearly; they never disappear silently.

## Discussion

Simulators may support, raise concerns, ask concern-based questions, answer directly, compare options, react to another participant’s latest relevant statement, respond to an active issue, or remain silent.

The ordinary reason hierarchy is:

1. persona reason;
2. option upside/concern fallback;
3. raw attribute only when explicitly relevant.

One active issue may be a question, concern, or comparison. Comparison is a soft language objective: if the endpoint visibly expresses only one useful side, the message remains valid but the hidden pair is not recorded as public comparison evidence.

- A direct question closes after its addressed answer. The structured question carries a small semantic mode instead of a prewritten sentence; prompts describe the meaning without prescribing “deal-breaker” or a fixed drawback-versus-benefit construction. The answer distinguishes trade-off acceptance, maintained concern, known mitigation, or unknown information.
- A concern may receive one or more relevant responses, then its owner visibly resolves or maintains it.
- A neutral rank-3 option may be considered directly. A disliked rank-2 option becomes eligible only after that participant's own concrete concern was visibly resolved or softened. Rank-1 and hard-blocked options remain impossible.
- A visible acceptance must state the concrete reason that made the option workable. That reason is stored with the stance update and reused by later switching or fallback realization.
- Maintained concerns become stale public reservations rather than looping until the turn cap.
- Hard blockers never resolve a concern by accepting another option.
- Ordinary public pro/con reasons are not repeatedly offered by different speakers as new standalone points; direct answers and issue responses remain exempt.

The issue does not prescribe the content of a response; the responding simulator still selects its own structured action.

## Pacing

Voluntary-turn budgets scale with participant count. The current defaults are 2/4/6 voluntary turns per participant for minimum/soft/hard pacing, bounded by absolute soft/hard caps of 22/30 turns. Openings, mandatory answers, narrowing turns, and votes are separate. This gives engagement more room in small and medium groups without letting six- or seven-person chats grow without bound.

## Narrowing

Narrowing is adaptive rather than a mandatory restatement round:

- unanimous public preference or shared public acceptability → skip participant narrowing and request votes;
- one clear leader → only dissenters and unresolved concern owners receive final-position opportunities; supporters may respond voluntarily when a dissenter raises a concern or proposal;
- exact top pair → participants whose public position does not settle the pair may clarify, accept, or remain firm;
- complete tie → expose one bounded simulator-owned compromise opportunity. Participants may propose an acceptable alternative, remain firm, or produce no bid.

An unchanged participant normally states only a short position. Explanations are required for new acceptance, switching, or a maintained blocker. Narrowing options are derived from latest public preferences and shared public acceptability, never weighted support/concern scores.

## Stagnation and compromise

A no-bid window after the minimum discussion budget is treated as stagnation. The moderator asks whether any non-first-choice option is workable only when a simulator has already produced a selectable compromise proposal. Each non-hard-blocker then independently applies the configured movement probability to its own acceptable alternatives. One proposal may create a short reaction exchange. If nobody proposes movement, the discussion proceeds toward a valid unresolved outcome.

## Protocol-critical realization

One normal generation and one focused repair are used for a selected action. Failed attempts and errors are retained compactly in `run.json`. Once a stance-changing action has won the floor, it is authoritative and may not disappear: if both language attempts fail, the runtime commits a minimal grounded movement statement. Formal votes use the same principle. Moderator compromise prompts remain buffered and are appended only together with the successfully realized or fallback participant turn.
