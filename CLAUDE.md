# Development instructions

Read `README.md`, `POST_EVAL_CLOSEOUT_PLAN.md`, and `info/00_overview.md` before changing runtime behavior.

## Architectural invariants

1. `UserAction` is authoritative.
2. `UserSimulator` chooses participant behavior.
3. The floor selects an intact bid and never rewrites it.
4. The LLM realizes wording only.
5. Public state changes only after accepted visible text.
6. State-changing actions—opening preference, issue resolution, stance change, and vote—must be visibly expressed.
7. There is no validator LLM.
8. Do not reintroduce urgency scores, floor multipliers, expected-turn-share correction, candidate weights, or public-pressure formulas.
9. All behavioral probabilities and language limits belong in `config.yaml`.
10. Raw option attributes do not automatically become conversation topics.
11. Direct questions use a compact semantic mode, ask about a concrete concern without prescribing stock wording, and close after the required answer.
12. Narrowing is adaptive: unanimous groups skip participant restatements; with one leader only dissenters or unresolved concern owners need a final-position opportunity; complete ties expose optional compromise.
13. A non-hard blocker may resolve a previously disliked option concern; a hard blocker never accepts another option.
14. Stagnation creates a simulator-owned compromise opportunity, never a controller-ordered concession.
15. A second vote is allowed only after visible acceptance or switching in re-narrowing.
16. Ordinary discussion must not systematically open concerns against every alternative; the configured concern cap is a safety bound.
17. Do not commit a moderator compromise/stagnation prompt unless a selected simulator response has been successfully realized and will immediately follow it.
18. Comparison wording is soft: only visibly mentioned option pairs become public comparison evidence.
19. During voting, one visible intended option with no competitor is a valid natural vote even without a fixed vote phrase. If generation and focused repair both fail, render a minimal deterministic statement for the already-authoritative vote; never lose a vote.

20. Opening realization uses `INITIAL`, `ALIGN`, or `CONTRAST` mode to avoid restarting every participant turn with the same greeting/preference skeleton.
21. Voluntary pacing is participant-scaled but bounded by absolute caps for groups of six and seven.

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
