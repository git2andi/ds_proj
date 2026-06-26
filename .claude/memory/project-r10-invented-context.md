---
name: project-r10-invented-context
description: R10 fix — speakers invented real-world facts about named options not in the cards; fixed with epistemic grounding rule
metadata: 
  node_type: memory
  type: project
  originSessionId: 61f5108c-d2c8-409d-b209-5586e337a8b2
---

R10 (2026-06-26): Speakers asserted invented facts about real-world named options ("The Daily Grind has a big room in back", "Cupcake has group deals", "Sakura has tight tables"). Problem is specific to topics with named real-world places — the model draws on training knowledge. Abstract options (presentation topics, board games) don't trigger this.

**Why:** When given a real cafe/restaurant name, the model uses its world knowledge to be "helpful" and answer questions, rather than staying within the option cards. Especially bad in the Q&A chain: Person A asks "do they have seating?" → Person B invents a confident answer → downstream speakers treat the invention as established fact.

**How to apply:** When reviewing transcripts for cafe/restaurant/location topics, look for specific attribute claims (seating, pricing, amenities, hours, staff) and check if they're in the option cards. If not, they're invented. The fix is the epistemic framing rule in `sim_utterance` rule 3: "You know only what's in the option cards — anything else is unknown: say 'I'm not sure' or 'we'd need to check', never a confident claim." Never weaken this to "hedge it" (too vague) — the model interprets hedging as optional.

Fixed in `src/prompts.py`: rule 3 of `sim_utterance`, and `_REPAIR_HINTS` for `INVENTED_OPTION_ATTRIBUTE` and `UNGROUNDED_NUMERIC_FACT`.
