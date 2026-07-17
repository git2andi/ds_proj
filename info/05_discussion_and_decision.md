# Discussion and decision process

## Structured simulator actions

Each eligible simulator produces either silence or one complete `UserAction`. The action contains the act, option focus, optional addressee, simulator-owned grounded reason, issue relation, optional stance update, and vote when applicable.

The controller never fills in or rewrites an ordinary participant action. Only the selected action is sent to the LLM for language realization.

## Content sources

Simulator reasons come from:

1. persona-specific reasons for or against an option;
2. a public option upside or concern;
3. a concrete public attribute when a question, comparison, or active issue needs it;
4. optional relevant backstory/private-goal context.

Objective option claims must remain tied to the focused option. Personal preferences and experiences may be expressed as subjective statements.

## Discussion floor

Bids use categorical priority:

1. required direct answer;
2. active-issue or moderator-stimulus response;
3. ordinary voluntary contribution.

The floor chooses within the highest available category using the run seed, avoids pathological consecutive turns, and never equalizes participation. Engagement affects whether a voluntary bid exists.

## Questions and concerns

The environment stores at most one active issue.

- A direct question creates one required answer from the named addressee.
- One optional third-party reaction may continue the question exchange.
- Concern responses and owner reactions remain voluntary.
- A concern may be resolved, softened, maintained, or become stale when the group moves on.
- Semantic duplicate concerns are suppressed and reopening is bounded.

The active issue provides context; it does not prescribe who must defend, concede, or change stance.

## Pacing

Voluntary discussion budgets scale with participant count and are bounded by absolute caps. Openings, mandatory answers, narrowing turns, and votes are tracked separately.

The soft target is not a fixed required length. Public unanimity can close earlier after some substantive discussion. Novel bids can continue past the target until the hard cap. Small- and large-group bounds prevent very short filler loops and excessive scaling.

## Narrowing

Narrowing uses public preferences, acceptances, unresolved concerns, and visible movement:

- unanimous public preference can proceed directly to voting;
- one leader gives relevant dissenters or unresolved concern owners bounded final-position opportunities;
- a top pair allows clarification or acceptance;
- a complete split exposes one simulator-owned compromise opportunity.

The environment may schedule a protocol opportunity, but the simulator still chooses the actual position and whether to move. A valid majority is an acceptable outcome; the runtime does not attempt to force unanimity.

## Realization and repair

For one selected action, the runtime performs one normal generation and at most one focused repair. It records both calls and aggregates their token usage.

A voluntary movement that remains invalid after repair is dropped and logged rather than converted into scripted participant language. A formal vote, or a mandatory movement statement required by the current protocol, may use one concise grounded fallback so the run can close. Moderator prompts that introduce a participant response are buffered until the response is successfully accepted.

Evaluation-only long cases may temporarily allow semantic reason reuse. Normal runs keep reason reuse disabled.
