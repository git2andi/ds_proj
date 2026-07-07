# 06 — Consensus and outcomes

The simulator has exactly three outcome labels:

- `successful`: all visible final stances support the same option;
- `majority`: enough visible support exists for one option, but not everyone supports it;
- `unresolved`: no sufficient agreement remains after bounded narrowing.

## Visible evidence rule

The final outcome is computed from visible transcript evidence. Public votes, acceptances, and parser-recognized commitments count. Hidden current preferences guide routing but do not directly decide the outcome.

## v3 compromise rule

v3 keeps compromise conservative and visible. The latest version also protects valid holdouts so majority outcomes remain possible instead of converting every dissent into consensus:

- no downhill compromise into a smaller visible camp;
- strict plurality gives some pressure;
- ties only get a small compromise bonus for flexible, non-hard-blocked sims;
- very stubborn or high-threshold sims require stronger evidence to switch;
- decision turns have a required target, so generated vote/switch lines cannot silently commit to a different option;
- final-vote routing avoids old latent favorites that the same sim visibly rejected.

This should reduce unresolved endings without making every split magically successful.

## Unresolved endings

Unresolved is valid. It should appear when the transcript shows real remaining blockers or unresolved preference camps. v3 adds one participant acknowledgement before closure so the deadlock is explicit in the conversation.
