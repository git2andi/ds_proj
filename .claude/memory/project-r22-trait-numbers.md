---
name: project-r22-trait-numbers
description: R22 fix — compact trait numbers added to speaker card so llama3.3 can calibrate persona voice (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

Compact trait line `Traits: extra=N agree=N neuro=N` appended to `runtime_speaker_card()` in `prompts.py`. Previously trait values were absent from the prompt (removed earlier to save tokens), leaving only the speaking-habit label. llama3.3 now has a numeric anchor to calibrate verbosity (extraversion), tone (agreeableness), and hedging (neuroticism).

**Why:** Speakers with very different trait profiles sounded similar; the speaking-habit string alone wasn't enough for the model to maintain distinct voices across turns.

**How to apply:** If voice differentiation regresses, check whether the Traits line is present in the speaker card. Don't expand to more than 3 traits without removing something else — prompt budget is tight.
