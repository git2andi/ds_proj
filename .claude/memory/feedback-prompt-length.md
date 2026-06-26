---
name: feedback-prompt-length
description: "User wants shorter prompts, not longer — adding more rules worsens output quality with llama3.3"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 45aa81ed-c223-4bc5-8692-c38d4cb41c5c
---

Do not simply lengthen existing prompts. They are already really long, just adding more stuff complicates and worsens discussions.

**Why:** The underlying model (llama3.3 via Ollama) can only follow so many instructions at once. When the prompt grows, the model starts ignoring more rules — especially the nuanced ones about tone and voice. Shorter, more direct prompts produce better outputs.

**How to apply:** When improving discussion quality, prefer trimming/consolidating existing prompt text over adding new rules. Move enforcement to deterministic validation checks where possible (validation.py catches things the prompt can't enforce). Every new rule added to sim_utterance should come with an old one being cut or merged.
