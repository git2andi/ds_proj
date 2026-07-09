# CLAUDE.md

Working instructions for coding agents in this repository.

## Project framing

This is a university project for an option-grounded multi-user decision simulator. The system simulates 2-7 LLM-driven participants discussing a fixed set of options. The goal is a configurable simulator whose participant parameters visibly affect turn-taking, stance movement, disagreement, compromise, and final outcomes.

The architecture should stay explainable:

```text
state -> route speaker/macro-act/target/focus -> generate one utterance -> validate -> parse visible state -> continue/narrow/vote/close
```

The LLM renders utterances. The controller owns phase logic, routing, narrowing, and final outcome rules.

## Current stance model

Private stance is stored as one per-sim/per-option rank table:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

Use `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` as computed helpers over the rank table. Do not add a second independent preference model.

Participant setup may provide short `reason_for` / `reason_against` fields for each option. Keep these short. Leave neutral options neutral. Hard rejects should be rare and grounded.

## Personas, hidden traits, age, and speech style

OCEAN traits are hidden setup traits. They only derive the four simulator parameters and plausible persona content; they are never passed into participant utterance prompts or routing.

The simulator parameters are the only numeric behavior controls:

```text
engagement    -> expected speaker frequency / turn share
verbosity     -> average utterance length (numeric word budgets only, never prose)
directness    -> blunt vs soft wording
stubbornness  -> resistance to changing stance, strength of stance defense
```

Age and `speech_style` are surface-realization metadata. `speech_style` is derived from age in four compact bands (18-27 young casual, 28-40 relaxed practical, 41-58 direct workplace, 59+ measured traditional wording) unless manually overridden. It may alter wording, formality, and conversational flavor, but must not alter routing, stance strength, vote choice, compromise willingness, or outcome logic. Hard blockers come only from `rejection`, not from `stubbornness`.

Profiles/backgrounds should be plausible for the generated or manually configured age. The builder performs deterministic checks for obvious contradictions such as a very young participant being described as a senior executive, long-term homeowner, married parent, or having decades of experience.

Manual participant profiles may include `age` and `speech_style`. If omitted, age/speech_style are filled by the builder. Manual `parameters` accept only `engagement`, `verbosity`, `directness`, and `stubbornness`.

## Discussion agenda

Do not reintroduce per-sim scripted agendas. The active pre-vote checklist is the chat-level `DialogueState.discussion_agenda`.

Persona-specific perspectives belong in:

```text
Persona.private_goal
OptionStance.reason_for
OptionStance.reason_against
```

The global agenda should track what the whole discussion still needs, not what one participant is scripted to say.

## Compact act vocabulary

The controller should reason over macro acts:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Do not grow the act list unless there is strong log evidence that a new act cannot be represented by the macro set.

## Implementation principles

- Prefer deterministic controller/state logic over new LLM calls.
- Keep prompts act-specific and compact.
- Use the option-rank table as the stance source of truth.
- Keep the four simulator parameters behavior-relevant; keep speech_style wording-only.
- Do not add more personality traits or simulator parameters unless required.
- Do not turn the project into a full agenda simulator.
- Do not reintroduce per-sim agenda steering.
- Normal auto-generated sims should have movable preferences, not categorical blockers.
- Manual hard constraints must remain binding.
- Unresolved outcomes are allowed, but must be earned after narrowing attempts.
- Decision turns should normally parse without repair.
- Invalid lines must not become transcript evidence.

## Development workflow

1. Work on one issue at a time.
2. Inspect the current code and latest logs before editing.
3. Make the smallest coherent change.
4. Run static checks:

```powershell
py -m py_compile main.py eval\run_eval_suite.py src\*.py eval\*.py
```

5. Before claiming stable behavior, run the eval suite (it always runs all cases) if LLM access is available:

```powershell
py .\eval\run_eval_suite.py
```

6. Inspect `transcript.md` and `run.json`, not only CSV metrics.
7. Update `README.md`, `docs/todo.md`, and relevant `info/*.md` files when behavior or workflow changes.

## What to inspect in logs

- Does the transcript read like a plausible group decision?
- Are turns grounded in the option board?
- Are direct questions answered by the addressed sim?
- Do stance switches have visible reasons?
- Do final votes respect the sim's rank/visible stance?
- Do rank movements make majority/success/unresolved outcomes plausible?
- Are blockers preventing false unanimity?
- Are age/profile/speech_style fields present and plausible?
- Is speech_style visible without overriding parameter-driven behavior?
- Are repair/fallback/token counts reasonable?
