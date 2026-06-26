---
name: project-short-name-fix
description: "R8 fix — OptionCard.short_name LLM-generated alias replaces blind 2-word truncation; prevents \"Wine and\", \"Settlers of\", \"Ticket to\" artifacts"
metadata: 
  node_type: memory
  type: project
  originSessionId: 61f5108c-d2c8-409d-b209-5586e337a8b2
---

`OptionCard` now has a `short_name: str = ""` field (models.py). The setup LLM generates it as part of the option schema (e.g. "Ticket to Ride" → "Ride", "Settlers of Catan" → "Catan", "Carcassonne" → "Carc"). `builders._clean_short_name` rejects any value that ends on a stopword or is >3 words. `prompts._short_alias` uses `option.short_name` first; fallback: 3-word names return the full name, 4+ words take first 2 but swap word[1] if it is a dangling stopword.

**Why:** The old deterministic `_short_alias` (first 2 words) produced broken aliases like "Wine and", "Mac and", "Settlers of", "Ticket to", "The Daily".

**How to apply:** The short_name is stored in run.json as part of each OptionCard. Old logs have `short_name=""` (empty string, fallback kicks in). New runs always have it.
