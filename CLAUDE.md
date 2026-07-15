# Development instructions

Read `README.md`, `ACTION_PLAN.md`, and `info/00_overview.md` before changing runtime behavior.

## Architectural invariants

1. `UserAction` is authoritative.
2. `UserSimulator` chooses participant behavior.
3. The floor selects an intact bid and never rewrites it.
4. The LLM realizes wording only.
5. Public state changes only after accepted visible text. Questions and newly opened concerns must be visibly realized before they change protocol state.
6. State-changing actions—opening preference, issue resolution, stance change, and vote—must be visibly expressed.
6a. Every acceptance or switch carries a concrete simulator-owned movement reason. Do not replace it with generic fairness wording. Once that reason is public, a later changed vote may be brief.
7. There is no validator LLM.
8. Do not reintroduce urgency scores, floor multipliers, expected-turn-share correction, candidate weights, or public-pressure formulas.
9. All behavioral probabilities and language limits belong in `config.yaml`.
10. Raw option attributes do not automatically become conversation topics.
11. Direct questions must clearly name their intended participant in a natural vocative position and create one required answer. The same addressee/option/concern question may not be reopened. Afterward, at most one ordinary voluntary reaction may continue the exchange; answered questions then close as resolved.
12. Narrowing is adaptive: unanimous groups skip participant restatements; with one leader only dissenters or unresolved concern owners need a final-position opportunity; complete ties expose optional compromise.
13. Rank-3 alternatives may be considered directly. A rank-2 disliked alternative becomes compromise-eligible only after that participant's concrete concern was visibly resolved or softened. Rank-1 and hard-blocked alternatives never become acceptable.
14. Stagnation creates a simulator-owned compromise opportunity, never a controller-ordered concession.
15. A second vote is allowed only after visible acceptance or switching in re-narrowing.
16. Ordinary discussion must not systematically open concerns against every alternative. Concern responses and owner reactions are voluntary, responders are bounded, the same semantic concern is opened only once during discussion, and it may be reopened at most once during narrowing.
17. Do not commit a moderator compromise/stagnation prompt unless a selected simulator response has been successfully realized and will immediately follow it.
18. Comparison wording is soft: only visibly mentioned option pairs become public comparison evidence.
19. During voting, one visible intended option with no competitor is a valid natural vote even without a fixed vote phrase. If generation and focused repair both fail, render a minimal deterministic statement for the already-authoritative vote; never lose a vote.
19a. Once any stance-changing action wins the floor, it must commit visibly. After failed generation and focused repair, use a grounded minimal movement fallback and log the realization failure; never silently select and then discard movement.

20. Opening realization uses `INITIAL`, `ALIGN`, or `CONTRAST` mode to avoid restarting every participant turn with the same greeting/preference skeleton.
21. Voluntary pacing is participant-scaled but bounded by absolute caps. Groups of two through four receive one additional ordinary no-bid retry before narrowing, without forcing a contribution. Large-group clear-leader narrowing is capped to three required final-position participants.
22. Realization must preserve the exact meaning and strength of supplied option facts. Do not invent option subtypes, facilities, use cases, guarantees, or unsupported relative claims such as cheapest, shortest, fastest, balanced, or best value. Deterministic grounding is intentionally narrow and does not claim full semantic entailment.
23. Scenario alias repair may use a deterministic last resort, but only by selecting short validated phrases from words already present in the full option name. Never fabricate abbreviations or external codes.

## Scope

This is an option-grounded group-decision user simulator, not a general human social simulation. Do not add coalitions, emotions, deception, status hierarchies, or unrestricted memory unless explicitly required.

## Runtime phases

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

One bounded `VOTING → NARROWING → VOTING` return is allowed when no majority exists.

## Testing

Run after each meaningful change:

```powershell
py -m pytest -q
```

Tests should assert public behavior and ownership boundaries, not exact LLM wording or removed implementation details.

## Evaluation

The LLM-backed suite is diagnostic. Adapt it when the public runtime contract changes, but do not turn it into a second runtime policy.


## Final turn semantics

The `voluntary` metric means self-selected floor entry. Direct answers are required; later unsolicited comments by other simulators are self-selected. The two `long_*` evaluation cases use per-case stress overrides and do not change normal pacing.
