# CLAUDE.md

## Current Project State

This project generates small, bounded, literature-informed group decision
chats. A short topic becomes four fictional option cards, 2-7 simulated
participants, a natural discussion, explicit votes, optional compromise checks,
and a logged outcome.

The project is not a full social simulator. The useful v1 target is:

- participants have stable traits, roles, goals, preferences, and reasons;
- chats sound casual and human without heavy roleplay;
- option facts are self-contained, so Sims do not invent missing information or
  ask people to check live facts;
- decisions are based on explicit votes, accepts, and rejects;
- rare failed or force-closed outcomes are allowed, but must be logged honestly;
- batch generation should stay bounded and inspectable.

## Runtime Flow

The live flow is:

```text
setup -> opening -> discussion -> vote -> confirmation_if_needed -> final
```

1. `Orchestrator` generates four grounded option cards and an opening question.
2. `PersonaBuilder` creates 2-7 participants from `config.yaml`.
3. Personas receive traits, roles, goals, private beliefs, reasons, acceptable
   options, rejected options, and reconsider conditions.
4. The deterministic moderator introduces the topic and options.
5. Each participant gives an opening priority.
6. The discussion runs until a readiness gate says enough substance exists, or
   until the configured maximum turn budget is reached.
7. The moderator asks for current picks.
8. Vote and confirmation turns use structured JSON internally, while the visible
   transcript keeps only the natural chat message.
9. If everyone votes for or accepts the same option, the dialogue succeeds.
10. If no shared option is accepted, the moderator force-closes or fails
    honestly.
11. `DialogueLogger` writes `.txt` and `.eval.json` outputs.

The discussion is not a fixed script length. It is bounded by config to keep
batch runs scalable, but narrowing is readiness-based.

## Active Modules

`main.py` is the entry point:

```bash
python main.py
python main.py scenarios.txt
```

Active `src/` modules:

- `config_loader.py`: loads and validates `config.yaml`.
- `llm_client.py`: provider wrapper for `uni`, `groq`, and `gemini`.
- `prompts.py`: the single registry for all prose sent to an LLM.
- `persona.py`: names, roles, traits, personas, and private belief kits.
- `orchestrator.py`: phase control, explicit outcome state, compromise checks.
- `simulator.py`: one participant turn, verification, one repair attempt,
  deterministic fallback for phase-critical failures.
- `policy.py`: speaker selection, addressee routing, repetition pressure, rare
  hard-blocker sampling.
- `state.py`: compact `DialogueMemory`, not a consensus engine.
- `prompt_context.py`: speaker cards, option cards, memory snippets, local
  context, output contracts.
- `verifier.py`: deterministic participant-turn checks.
- `utils.py`: option resolution, alias matching, vote extraction.
- `logger.py`: transcript and evaluation JSON.

## Configuration Rules

All adaptive numbers should live in `config.yaml`.

Important current sections:

- `llm`
- `simulation`
- `turns`
- `repetition`
- `personas`
- `response_length`
- `voice`
- `argument_kit`
- `divergence`
- `memory`
- `stubbornness`
- `option_generation`
- `prompt_budget`
- `structured_control`
- `prompt_contracts`
- `turn_policy`
- `output`
- `grounding`
- `verification`
- `closure`

Do not add new magic numbers directly inside runtime code. If a value controls
behaviour, add it to `config.yaml` and access it via `cfg`.

## Prompt Rules

`src/prompts.py` should contain every LLM-facing prompt.

Other modules may compose prompt sections, select context, or pass variables,
but they should not hide new prompt prose in random methods. This keeps prompt
auditing possible.

Current prompt families:

- option generation;
- names and roles;
- persona generation;
- private belief generation;
- free-form participant turns;
- structured vote turns;
- structured confirmation turns;
- targeted repair prompts.

No active prompt functions should exist for deleted moderator engines, surface
moves, challenge graphs, or conditional compromise scraping.

## Decision Model

The live decision model is explicit.

Important `DialogueState` fields:

```text
explicit_votes
explicit_accepts
explicit_rejects
candidate_option
preferred_option
pending_confirmation_target
pending_confirmation_candidate
outcome_reason
```

Vote round example:

```json
{
  "message": "I'd go with Option B because the timing is easier.",
  "action": "vote",
  "option": "B"
}
```

Confirmation example:

```json
{
  "message": "I still prefer Option C, but Option B works well enough.",
  "action": "accept",
  "option": "B"
}
```

The transcript stores only the visible message. The structured `action` and
`option` drive the outcome.

An option succeeds when every participant either voted for it or explicitly
accepted it. Private beliefs help choose plausible candidates to test, but they
do not pretend public consensus already happened.

## Persona Model

Personas are intentionally richer than a short name/personality tag because the
conversation needs grounded material.

Each participant has:

- name and role;
- primary/non-primary flag;
- Big Five traits plus response-length tendency;
- derived conversational controls such as initiative, flexibility, directness,
  detail level, and warmth;
- short backstory and goal;
- private preferred option;
- acceptable options;
- usually empty rejected list, unless a real hard line exists;
- key concern;
- concrete reasons;
- reservation about another option;
- `would_reconsider_if`.

The system supports rare stubbornness with
`stubbornness.hard_blocker_dialogue_probability`. A stubborn participant is not
trying to sabotage the chat; they simply should not accept a non-preferred
option if their private state makes that dishonest.

## Option Model

Options remain string cards, but they are validated and intentionally
self-contained.

Required shape:

```text
Option X - Name: attrs: key=value, key=value; upside: ...; tradeoff: ...; concern: ...; fit: ...; risk: ...; best for: ...
```

The option cards are the whole fictional decision world. They must not contain
live-checking hooks such as availability, waitlists, unknown policies, call
ahead, check online, or unspecified facts.

`OptionResolver` maps:

- `Option A`;
- bare `A`/`B` in vote-like contexts;
- option title aliases.

It is a practical resolver, not a broad stance parser.

## Verifier

`verifier.py` is deterministic and participant-focused.

It checks:

- empty output;
- accidental speaker name prefix;
- invalid option references;
- denial of valid options;
- invented option numbers or facts;
- direct mutation of listed attributes;
- fact-chasing questions;
- question chains;
- self-repetition;
- semantic point repetition;
- acknowledgement loops;
- missing explicit votes in narrowing;
- unclear or too-thin confirmation replies.

Repair is limited to one LLM attempt. If a vote or confirmation still fails,
`simulator.py` can use a deterministic fallback. Normal discussion turns should
not become deterministic templates.

## Logging

Each dialogue writes:

```text
logs/<dialogue_id>.txt
logs/<dialogue_id>.eval.json
```

The transcript is for human reading. The JSON stores:

- topic;
- outcome;
- token usage;
- participants and beliefs;
- turn records;
- verification/repair metadata;
- explicit votes, accepts, and rejects;
- memory summary;
- transcript-quality indicators.

`outcome_valid` means the final structure is coherent. It does not prove the
chat sounded good; naturalness still needs transcript inspection or a separate
offline evaluator.

## Research Grounding

Keep only implementation-useful insights:

- Sacks, Schegloff & Jefferson: one speaker at a time and local turn routing.
- Ouchi & Tsuboi: addressee/question response routing in multi-party chat.
- Toulmin: compact private belief kits with reasons and reservations.
- McCrae & John: Big Five as a lightweight voice/persona scaffold.
- Shanahan et al.: roleplay framing as prompt scaffolding, not dramatic acting.
- Park et al.: compact memory and anti-repetition only, not full reflection.

Do not reintroduce paper-by-paper runtime machinery. Literature belongs in the
thesis framing unless it directly simplifies or stabilizes generation.

## What Not To Reintroduce

Avoid adding back patterns that make the system hard to debug:

- broad inferred consensus instead of explicit votes and accepts;
- LLM-generated moderator interventions;
- heavy reflection memory;
- random question-heavy behaviour;
- new theory layers before the explicit control loop is stable.

The intended system is a bounded, scalable, inspectable generator for useful
synthetic decision discussions.
