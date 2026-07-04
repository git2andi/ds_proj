# System overview

These notes explain, in plain terms, what each part of the project does and how the
pieces fit together. They are a mental model for reading the code, not a thesis
chapter. They describe the system **as it is implemented today** (2026-07-05, after
the behavioral round: trait-weighted participation, stance/concern state,
mid-discussion movement, reservation negotiation, participant-owned procedure,
continuations, grounding). The one-line map: `info/` file ⇄ source module.

## What the project is

An **option-grounded multi-user decision simulator** — a multi-user dialogue
simulation framework for option-grounded group decisions. It is *not* an
open-ended group-chat simulator, not an agenda-based user simulator, and not a
human-realistic society simulation. You give it a short topic; it

1. builds a small decision **environment** (a topic → four factual option cards +
   shared context),
2. creates 2–7 **simulated users** (personas with traits, tunable behavior
   parameters, private preferences and goals),
3. runs a **controlled discussion** in which the personas compare the options,
   answer each other, and move toward a decision,
4. ends `successful`, `majority`, or `unresolved` **from visible votes only**, and
5. writes a readable transcript plus a structured trace and metrics.

The scope is deliberately narrow — small-group decisions over a generated option
board. That restriction gives the world a fixed source of truth, curbs
hallucination, and makes the outcome observable.

## The core idea: environment + controller loop, not one big prompt

The simulator is **not** one prompt that writes a whole conversation. It is a
controller loop where the LLM only ever writes **one message at a time**:

```text
one-line topic
  -> option-grounded scenario (the fact base)
  -> simulated users with private goals + behavior parameters
  -> controller picks: who speaks, to whom, which dialogue act, which option focus
  -> LLM realizes ONLY that next visible message
  -> observer parses the public text (votes, questions, blockers, option refs)
  -> visible state, coverage, obligations, outcome are updated
  -> repeat until a visible decision (or a bounded unresolved close)
```

The transcript is one output artifact. The real object is the framework behind it:
the controller decides *what should happen*; the LLM only decides *how to phrase it*.

## Module map (where each concern lives)

`DialogueRunner` in `src/dialogue.py` is the orchestration loop. It mixes in three
concern-specific modules (issue 8 split — the methods share one `self`/`DialogueState`,
so they read as one class but live in separate files):

```text
src/dialogue.py     orchestration: run(), phase loops, generation+repair pipeline,
                    moderator turns, pacing, printing, logging       (this + 03/04/05)
src/policy.py       PolicyMixin  — who speaks / which act / which target / which
                    option focus, vote readiness, candidate selection, word budgets,
                    surface-style intent flags                                  (03)
src/observer.py     ObserverMixin — parse a generated line, apply semantics
                    (votes, blockers, lean movement, switch events), response
                    obligations, open questions                                 (06)
src/validation.py   ValidationMixin — turn validation, grounding tripwire/judge,
                    deterministic fallback text                             (05/06)

src/builders.py     scenario + persona construction (setup)                     (01/02)
src/simulator.py    OCEAN -> behavior parameters, the weak private agenda        (02)
src/consensus.py    outcome logic from visible votes only                       (07)
src/parsing.py      the "observer's eyes": option refs + commitment/blocker parsing (06/07)
src/aliases.py      the single option-nickname contract (short_alias_map)       (01)
src/style.py        deterministic surface-style tracker (openings, repeats)     (03)
src/prompts.py      ALL LLM-facing prose (setup, moderator, per-turn, repair, judge)
src/models.py       typed state (Scenario, Persona, DialogueState, runtimes, …)
src/evaluation.py   per-run metrics                                             (08)
src/logger.py       transcript + run.json + metrics.csv                         (08)
src/config_loader.py  loads/validates config.yaml, exposes `cfg`               (09)
src/llm_client.py   provider abstraction (uni | groq | gemini | gpt)
```

The rest of these notes follow the flow of a run:

- `01` scenario/environment generation — the fact base
- `02` sim generation — the participants
- `03` routing & turn-taking — who speaks and why
- `04` moderator — the (configurable) facilitator voice
- `05` discussion & decision — phases, voting, compromise, bridged switches
- `06` consensus & outcomes — how a public decision is read from the text
- `07` evaluation & logging — what a run leaves behind
- `08` configuration & running — how to drive and tune it
- `09` topic examples — the same engine across domains

## Relationship to MUCA and ConvLab-style user simulation

MUCA is useful because multi-user interaction needs explicit control over *what*
should be said, *when* someone should speak, and *who* is addressed. This project
adapts that to **simulated** users: the controller decides speaker, addressee, act,
and timing, and the moderator (when enabled) intervenes state-awarely.

ConvLab-style user simulation is useful for its separation of goals, dialogue acts,
policy, state, and evaluation. We keep that spirit — sims have internal goals and
controllable behavior, not just decorative persona text — without a full ConvLab
implementation. (Honest caveat: the per-sim "agenda" is currently a weak hint list,
not a real goal stack — see `02`.)

## Design principles (the non-negotiables)

1. Option facts are fictional when generated, but **hard facts** afterward.
2. Sims must not invent concrete facts beyond the option board / shared context.
3. Internal state may *guide* behavior, but only **visible transcript text** decides
   the public outcome — hidden preference is never counted as support.
4. Never close before participants had a visible decision opportunity.
5. A hard blocker never accepts its rejected option through state mutation.
6. Keep prompts compact; prefer controller/parser/validator/state fixes over long
   prompt blocks. All LLM-facing prose lives in `src/prompts.py`.
7. Fixes must generalize across topics, group sizes, and option domains.

## What "tunable simulator" means here

The parameters visibly shape the interaction (validated per run by the
realization metrics, `07`): engagement/initiative/responsiveness drive **turn
share** and initiative-taking (`03`), responsiveness drives how promptly direct
questions are answered, verbosity drives utterance length, stubbornness and
compromise threshold drive how hard a sim is to move (`02`, `06`), and the tracked
stance state (commitment strength, concerns, pressure) drives defending,
conceding, softening, and switching (`02`, `05`). Continuity across turns comes
from that stance/concern state — not from executing an agenda checklist.

