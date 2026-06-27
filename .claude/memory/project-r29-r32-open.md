---
name: project-r29-r32-open
description: Open issues R29 and R32 — Considering opener needs repair escalation (priority); ANSWER-turn ? falls through validation (2026-06-27)
metadata: 
  node_type: memory
  type: project
  originSessionId: 448d7ae6-0f14-48d8-b082-e4e42ac7e259
---

**R29: "Considering..." opener (TOP PRIORITY)** — The pattern `^\s*considering\b` is in `_ROBOTIC_TEMPLATES` (validation.py) but ROBOTIC_TEMPLATE is warn-only. The phrase appears 1-4 times per run despite being in the prompt's no-stock-phrases list. Fix: split the `^\s*considering\b` check out of the blanket warn, make it repair-level (same as POSSESSIVE_SUBJECT in R19, REPEATED_START in R20). Repair hint: "don't start with 'Considering' — open with the point itself, a reaction, or a question." Other ROBOTIC_TEMPLATE patterns can remain warn-only.

**R32: ANSWER-routed "?" falls through all validation** — When an ANSWER-routed speaker generates a question instead of an answer, no validation check fires: `UNWANTED_QUESTION` covers only `{VOTE, ACCEPT, REJECT, OPENING}`, and `QUESTION_IN_CONFIRMATION` covers only `ACCEPT`. The generated "?" then propagates via `_update_questions` as a new OpenQuestion, creating another ANSWER cycle. Fix: add `ActType.ANSWER` to the `statement_only` set in `_check_question_presence`, with repair hint: "you were asked a question — give an answer or say you can't confirm, don't re-ask."

**Why R29 is priority:** Appears in every batch of runs, visually degrades transcripts. R32 is more rare and partially mitigated by R26 (which blocks propagation when QUESTION_ECHO also fires).

**How to apply:** See fix candidates in docs/known_failures.md under R29 and R32.
