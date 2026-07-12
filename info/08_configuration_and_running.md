# 08 — Configuration and running

Important configuration sections:

- `llm`: the dialogue-generation backend and model mappings. A validator role may remain configured
  for compatibility, but normal runtime does not instantiate or call it.
- `simulation`: participant count, seed, and setup attempts.
- `environment`: automatic or manual option setup.
- `participants`: automatic/manual profiles and hard-blocker sampling.
- `conversation`: soft discussion pacing and vote-round limits.
- `threads`, `narrowing`, and `routing`: interaction and controller behavior.
- `moderator`: opening, nudges, vote call, and whether the deterministic closing line is shown.
- `validation`: `mode: critical`; deterministic correctness checks with at most one critical repair.
- `output`: log paths and optional prompt dumps.

Critical validation covers malformed output, option identity/aliases, required focus/questions,
formal votes and switches, blocked-option acceptance, existing-option compromise, transferred exact values,
unlisted exact quantities, and explicit unlisted feature/location claims. Ordinary opinions, support, concerns, and natural comparisons are not
sent to an LLM validator.

## Running

```powershell
py .\main.py
py .\main.py "Choose a restaurant for a group dinner"
py .\main.py topics.txt
"Choose a restaurant" | py .\main.py
```

An explicit CLI topic or topic file takes precedence over configured automatic topics. Manual
environment mode may be run without a CLI topic. Run `py -m pytest -q` before the live suite.
