# Master Group Discussion Simulator

Layout:

```text
root/
  main.py
  config.yaml
  logs/
  src/
    builders.py
    config_loader.py
    dialogue.py
    llm_client.py
    logger.py
    models.py
    parsing.py
    prompts.py
    router.py
    utils.py
    validation.py
```

Design boundaries:

- `config.yaml` stores tweakable numeric parameters.
- `src/prompts.py` stores all LLM prompts and moderator/chat templates.
- `src/models.py` stores typed state objects.
- `src/router.py` selects who speaks, when, to whom, and with what local move.
- `src/dialogue.py` owns orchestration, phase control, consensus, and state updates.
- `src/parsing.py` owns option resolution and trailer parsing.
- `src/validation.py` provides deterministic guardrails only; it should not become a second policy engine.

State extraction (the structured trailer):

Each generated turn ends with a small machine tag the chat reader never sees, e.g.
`[act=accept; opt=C; stance=accept]`. `parsing.parse_trailer` strips it and turns it
into a `TurnMove`. This replaces phrase "cue" regexes — deciding whether a message is a
vote/accept/reject is the model's job. If a turn omits the trailer, the parser falls back
to the routed intent. The option resolver only matches distinctive option names/ids
(shared words like "Night" in "Bowling Night"/"Game Night" are dropped automatically).

Convergence:

Agents have movable leanings. Strong arguments plus traits (compromise willingness) let a
persona move toward an option they can live with; the group narrows once leanings
concentrate (`conversation.concentration_to_narrow`), not by filling per-option counters.

Moderator facilitation:

The moderator does more than open/close. When the discussion circles (no *new*
information for `conversation.moderator_stall_window` turns) it summarizes where everyone
stands and pushes toward narrowing; in confirmation, if there is a single holdout, it asks
what would make the option work or whether another fits everyone. Interventions are
rate-limited (`moderator_cooldown_turns`, `moderator_max_interventions`).

Outputs (per run, under `logs/<run_id>/`):

- `transcript.md` — human-readable setup, chat, outcome, metrics, token totals.
- `run.json` — full structured run (scenario, personas, every turn, metrics).
- `logs/metrics.csv` — master file, one appended row per run for cross-run evaluation.
- `prompts.jsonl` — every prompt for the run, written only when `output.write_prompts: true`.

Failures are surfaced, not faked: if setup or generation fails there is no fabricated
fallback scenario/turn — you get a clear error.

Run:

```bash
py main.py
py main.py scenarios.txt
```

For local dry tests without an LLM endpoint, set `llm.provider: "mock"` in `config.yaml`.
The mock provider returns a schema-valid scenario and trailer-tagged turns so the whole
pipeline runs offline (it is a test double, not a fallback for failed real calls).
