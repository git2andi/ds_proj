---
name: project-stubborn-trait-refactor
description: is_hard_blocker flag removed; stubbornness is now purely trait-driven via agreeableness=1 (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**Change:** `is_hard_blocker: bool` removed from `Persona` model and all enforcement code. Stubbornness is now expressed through trait values only.

**How it works:**
- 4% per-run chance one participant gets `agreeableness=1` + `compromise_willingness` in `[0.10, 0.35]` (sampled from `hard_blocker_compromise` config range). All other participants keep `agreeableness` in `[3, 5]` (the existing normal range — no change needed).
- The LLM sees `agreeableness=1` in the trait card and plays the persona as stubbornly committed to their preferred option. No separate flag is passed.
- Router still uses `persona.traits.agreeableness == 1` for routing decisions (vote focus on preferred option; blocked from persuasion/fold path). This is trait-informed routing, not mechanical enforcement.
- No vote auto-correction in dialogue.py — LLM output is accepted as-is.
- `setup_personas` prompt: says agreeableness=1 → strong conviction, acceptable_options should contain only preferred option.
- Logger: `stubborn_participant` bool in run.json; " stubborn" label in transcript header.

**Why:** User's original intent was trait-driven emergence, not system-enforced immovability. "Traits should be meaningful."

**Key config knobs:**
- `personas.hard_blocker_probability: 0.04` — per-run chance of one stubborn participant
- `personas.hard_blocker_compromise: [0.10, 0.35]` — compromise range for stubborn sims
- `personas.trait_ranges.agreeableness: [3, 5]` — normal range (stubborn gets exactly 1)

**Committed:** 9be147c (2026-06-27)
