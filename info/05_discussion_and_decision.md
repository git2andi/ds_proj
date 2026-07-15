# Discussion and decision process

## Opening

Each participant must visibly state a current preferred option and one reason. The first speaker receives an `INITIAL` opening mode; later speakers receive `ALIGN` or `CONTRAST` according to already-visible preferences. Greetings are optional. In a two-person chat, the prompt explicitly avoids group greetings such as “everyone” or “all”.

Mandatory openings retry and then fail clearly; they never disappear silently.

## Discussion

Simulators may support, raise concerns, ask concern-based questions, answer directly, compare options, react to another participant’s latest relevant statement, respond to an active issue, or remain silent.

The ordinary reason hierarchy is:

1. persona reason;
2. option upside/concern fallback;
3. raw attribute only when explicitly relevant.

One active issue may be a direct question or concern. Comparison remains an ordinary voluntary action: if the endpoint visibly expresses only one useful side, the message remains valid but the hidden pair is not recorded as public comparison evidence.

- A direct question clearly names one participant in a natural position and requires that participant to answer next. A global `(addressee, option, concern)` key prevents the same question from being asked repeatedly. The answer clears the obligation. At most one later turn may continue the exchange, but only through the ordinary reaction logic when a simulator has a novel response to the answer. If no such bid is selected, the answered question closes immediately. The structured question carries a small semantic mode instead of a prewritten sentence; the answer distinguishes trade-off acceptance, maintained concern, or rare unknown information.
- A concern may receive up to two voluntary responses from distinct non-owners. Its owner may voluntarily resolve, partially soften, maintain, or ignore the issue. No owner reaction is mandatory. A semantic concern is opened only once during discussion and may be reopened once during narrowing when it remains blocking.
- A neutral rank-3 option may be considered directly. A disliked rank-2 option becomes eligible only after that participant's own concrete concern was visibly resolved or softened. Rank-1 and hard-blocked options remain impossible.
- A visible acceptance must state the concrete reason that made the option workable. That reason is stored with the stance update and reused by later switching or fallback realization.
- Maintained or abandoned concerns become stale public reservations rather than looping until the turn cap. Answered questions close as resolved even when nobody uses the optional follow-up.
- Hard blockers never resolve a concern by accepting another option.
- Ordinary public pro/con reasons are not repeatedly offered by different speakers as new standalone points; direct answers and issue responses remain exempt.

The issue does not prescribe the content of a response; the responding simulator still selects its own structured action.

## Pacing

Self-selected voluntary-turn budgets scale with participant count. The current defaults are 2/5/7 voluntary turns per participant for minimum/soft/hard pacing, bounded by absolute soft/hard caps of 22/30 turns. Openings, mandatory answers, narrowing turns, and votes are separate. For groups of two through four, shared acceptability alone does not trigger narrowing at the bare minimum; three additional voluntary turns are allowed when novel bids remain. After the first empty floor before the soft target, the group receives one additional ordinary bidding round, but no contribution is forced. Genuine public unanimity may proceed to voting after roughly one substantive post-opening contribution round, even before the normal minimum, so liveness handling does not manufacture filler.

## Narrowing

Narrowing is adaptive rather than a mandatory restatement round:

- unanimous public preference with no unresolved concern → skip participant narrowing and request votes; shared public acceptability may shorten, but does not automatically eliminate, narrowing;
- one clear leader → relevant dissenters and unresolved concern owners receive final-position opportunities. Groups of five through seven schedule at most three such participants, with at most one bounded issue response per scheduled participant. Once at least one additional acceptance or the strict majority threshold has been reached, the runtime stops scheduling generic concessions;
- exact top pair → participants whose public position does not settle the pair may clarify, accept, or remain firm;
- complete tie → visibly state the split once and expose one bounded simulator-owned compromise opportunity. Only options that somebody publicly prefers or accepts are included; untouched options are excluded. Participants may accept another option, remain firm, or produce no bid.

An unchanged participant normally states only a short position. Explanations are required for new acceptance, switching, or a maintained blocker. Narrowing options are derived from latest public preferences and shared public acceptability, never weighted support/concern scores.

## Stagnation and compromise

A no-bid window after the minimum discussion budget is treated as stagnation. The moderator asks whether any non-first-choice option is workable only when a simulator has already produced a selectable compromise proposal. Each non-hard-blocker then independently applies the configured movement probability to its own acceptable alternatives. One proposal may create a short reaction exchange. If nobody proposes movement, the discussion proceeds toward a valid unresolved outcome.

## Protocol-critical realization

One normal generation and one focused repair are used for a selected action. Failed attempts and errors are retained compactly in `run.json`. Once a stance-changing action has won the floor, it is authoritative and may not disappear: if both language attempts fail, the runtime commits a minimal grounded movement statement. Formal votes use the same principle. Moderator compromise prompts remain buffered and are appended only together with the successfully realized or fallback participant turn.


Two evaluation-only stress cases override these budgets without changing `config.yaml` defaults. Normal pacing uses a soft target of five and a hard maximum of seven voluntary turns per participant, bounded by absolute caps. The stress cases intentionally allow semantic reason reuse so extended-dialogue failure modes remain observable.

## Realization quality

Reaction-like actions include a compact interpersonal realization cue: the previous speaker’s visible point and the current participant’s selected reason or priority. This changes wording only; it does not alter the authoritative action. Movement prompts explicitly allow several natural semantic shapes rather than requiring the fixed `I still prefer X, but I can accept Y` construction.
