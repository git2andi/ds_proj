# Group Discussion Simulator

## What this is

This is a university project exploring how well LLMs can simulate a small group of
people having a real discussion and arriving at a decision together. Give it a topic
("decide which board game to play at game night", "plan a weekend team offsite", "pick
a restaurant for the team dinner") and it generates a full multi-party chat: 2-7
personas with their own backstories, goals, and opinions talk it through, react to each
other, change their minds (sometimes), and either land on a shared choice or fail to.

The motivation is that LLM-driven "user simulators" are increasingly used to test and
train dialogue systems, recommender agents, and group-decision tools — but only if the
simulated conversations actually *behave* like group discussions: people have different
starting preferences, some form natural coalitions, opinions shift when a good point
lands, and a discussion converges (or doesn't) for understandable reasons. A simulator
that just has every persona repeat "I prefer X because Y" until a counter hits a
threshold isn't useful for that. This project is an attempt to get closer to the real
thing while staying fully topic-agnostic — nothing here is tuned for one scenario.

## How it works

Generates natural, casual multi-party discussions where several LLM-driven personas
talk through a decision from a one-line topic and try to reach a workable group choice.
A deterministic controller decides **who speaks, when, and with what intent**; the LLM
only renders the surface text of each turn. This split keeps the chat coherent and
on-topic instead of letting agents free-run.

Concretely: one LLM call builds the scenario (a handful of named options with
comparable attributes, e.g. four board games with playtime/players/complexity) and a
hidden belief state for each persona — what they privately prefer, what they could
live with, what they'd reject, why, and a short backstory. From there, every turn of
the chat is driven by code: a router picks who speaks next and what kind of thing they
should do (open, react, compare two options, support or push back on a point, propose
a compromise, vote, accept, ask/answer). Only the *wording* of that turn comes from the
LLM — it writes one chat-style line, plus a hidden tag reporting what it actually did
(which option, what stance), which the controller reads to update the group's state.
A moderator (also LLM-voiced, but only triggered by code) opens the discussion, steps
in if the group is going in circles or has clearly converged, nudges the last one or
two holdouts near the end, and closes things out once there's a decision (or once it's
clear there won't be one).

## What a chat should look like

A good run reads like a short, casual group chat: people state a leaning early without
over-explaining it, then actually respond to each other — agreeing, disagreeing,
building on a point, occasionally getting persuaded and shifting position. Some
participants may walk in already aligned (a 2v1 or 3v1 split is normal, not everyone
needs a unique favorite), and that alignment can still shift during the conversation.
The moderator's voice should vary run to run rather than reciting the same stock
phrases. By the end, the group either lands on one option for reasons that emerged in
the chat (consensus, or a majority fallback nobody hard-objects to), or — more rarely,
when someone is a genuine hard blocker — ends without an agreement. Throughout, the
text should stay grounded in the scenario's actual facts (no invented numbers), avoid
repeating itself or other speakers, and never leak the internal `[act=...]` bookkeeping
tag into the visible chat.
Chats should work for the various topics and any fix should not be tailored to fit the 
current specific topic only. Any update/change should be done witht the thought that it 
must also work for any other topic given. It should work for various amounts of Sims 
aswell.

## Layout

```text
root/
  main.py            # CLI entry point
  config.yaml        # every tunable number (see "Configuration")
  logs/              # one folder per run + master metrics.csv
  papers/            # background literature
  src/
    builders.py      # one LLM call builds the scenario + persona belief states
    config_loader.py # loads config.yaml, exposes `cfg`, validates ranges
    dialogue.py      # orchestration: phases, moderator, consensus, state tracking
    llm_client.py    # provider abstraction (uni | groq | gemini)
    logger.py        # transcript.md, run.json, metrics.csv, optional prompts.jsonl
    models.py        # typed state objects (the only things routing/consensus operate on)
    parsing.py       # option resolution + dialogue-act trailer parsing
    prompts.py       # ALL LLM prompts and moderator/chat text
    router.py        # turn-taking: who speaks, addressee, local move
    scoring.py       # shared "how much does the group back option X"
    utils.py         # deterministic helpers (sampling, token overlap, JSON)
    validation.py    # deterministic guardrails on generated messages
```

Design boundaries (kept deliberately): `config.yaml` holds every tunable number;
`src/prompts.py` holds every piece of prose sent to an LLM or printed as moderator text;
`src/models.py` holds typed state; `src/validation.py` is guardrails only, not a second
policy engine.

## How a run works

1. **Setup** (`builders.py`): a single LLM call produces the scenario (4 option cards with
   stable, comparable attributes) and each participant's hidden belief state (preferred /
   acceptable / rejected options, a 1–5 private utility per option, reasons, goal,
   backstory, reservation). The builder samples Big-Five + cooperative-control traits in
   code and tells the model to use them. If setup can't produce a valid world it raises —
   no required field is silently defaulted (names, roles, goals, option names/upsides,
   reasons, per-option scores, …) and there is no fabricated fallback scenario. The only
   reasons the code ever derives are for options it *structurally* assigns itself (a
   coalition-reassigned preference or a forced common compromise), never as a stand-in for
   content the model was asked for but omitted.

2. **Coalitions** (`builders.py`): before setup, a code-side *preference plan* decides this
   run's split. Most runs are all-distinct (1-1-1), but with `personas.coalition_probability`
   two or more participants share a preferred option (2v1, 3v1, 5v2 …). The plan is both
   described to the setup LLM and enforced afterwards, so coalitions actually appear.

3. **Dialogue loop** (`dialogue.py` + `router.py`): the controller advances through phases
   — opening → discussion → narrowing → confirmation → closure. The chat opens with a quick,
   optional greeting from a trait-driven subset (more extraverted personas are likelier to
   chime in, at least one always does) and ends with short sign-offs that acknowledge the
   outcome; these social beats are cosmetic — they don't affect stance, coverage, or
   convergence. Each turn the router emits
   a `MoveIntent` (speaker, addressee, local move such as compare/support/object/propose,
   length hint). `prompts.sim_utterance` turns that into a prompt; the LLM writes one chat
   line ending in a hidden status trailer. Asking is emergent, not quota'd: the `ask` move's
   weight is scaled per speaker by their openness (curious people ask more) and damped right
   after a question, and a direct question gets answered before the thread moves on. Turns
   may also carry the occasional rhetorical/open question that others simply pick up on.
   Turn length follows the speaker's traits — `response_length`/`detail` bias the length hint
   and the prompt's verbosity guidance, so a terse persona and a chatty one read differently
   instead of every line landing on the same one-liner. The router also makes sure real
   *alternatives* get aired before the group settles, but only options at least one person
   prefers or could accept; an option nobody wants is left unmentioned rather than forced in
   as filler.

4. **State extraction (the trailer)**: every generated turn ends with a machine tag the
   chat reader never sees, e.g. `[act=accept; opt=C; stance=accept]`. `parsing.parse_trailer`
   strips it and produces a `TurnMove`. The parser is tolerant — it handles the tag with or
   without surrounding brackets and ignores a stray option letter — and if the tag is
   missing it falls back to the routed intent. Deciding whether a message is a
   vote/accept/reject is the model's job, reported via the trailer, not guessed from prose.

5. **Convergence**: personas have movable leanings, but they don't fold on a dime. A strong
   point plus a cooperative personality lets someone shift toward an option they can live with —
   when the group is rallying behind an option a persona privately rates acceptable, the router
   can hand them an explicit "won over" turn that voices the change of mind and moves their lean.
   That shift is **gated for friction**: it only fires once they've actually spoken and held
   their pick a couple of times (`routing.persuasion_min_speaker_turns`), when a rival option
   *clearly* out-pulls their current lean (`routing.persuasion_support_margin`) and someone has
   genuinely argued for it — the chance itself then scales with `compromise_willingness` via
   `routing.persuasion_probability_factor`. So nobody capitulates in the first exchange, and the
   shift is voiced rather than only ever surfacing as a silent tally at vote time. Opening
   turns are clamped to a stated leaning only, so a first-round line can never be miscounted
   as a vote/accept. The group narrows once leanings concentrate
   (`conversation.concentration_to_narrow`), not by filling per-option counters. How long the
   discussion runs before that is **derived per run, not a fixed floor**: a target is computed
   from group size and composition — more distinct starting preferences, lower mean
   compromise, and more detail-oriented personas all lengthen it, plus per-run jitter — and
   the force-narrow and hard-stop caps scale per participant, so a quick-agreeing trio is
   short while a large, split, stubborn group runs proportionally longer.

6. **Moderator** (`dialogue.py` + `prompts.py`): beyond opening/closing, the moderator
   steps in when the discussion circles, when everyone has clearly converged ("sounds like
   we all want X — lock it in?"), or to nudge one or two holdouts in confirmation (each then
   answers in turn). **All moderator lines except the fixed opening option board are
   generated by the LLM** from the current standings, so they vary run to run. Interventions
   are rate-limited (`moderator_cooldown_turns`, `moderator_max_interventions`).

7. **Consensus / outcome** (`dialogue.py`): `consensus` when everyone votes/accepts the same
   option, `fallback` when a strong majority backs one and nobody hard-rejects it (a clear
   2/3 majority counts, via `consensus.majority_fallback_fraction`), otherwise `unresolved`. Confirmation churn is bounded (`conversation.max_confirmation_turns`) so the
   group can't flip-flop forever. Stubborn trait draws can legitimately end a chat unresolved
   — that's intended, just kept rare by the trait ranges.

8. **Validation** (`validation.py`): deterministic guardrails repair turns that are empty,
   carry a speaker prefix, reference an invalid option, invent numbers, repeat the speaker,
   duplicate another speaker's line verbatim, reuse a long verbatim phrase from another speaker
   (the *echo guard* — a shared 6+ word run with option/participant names masked, so naming the
   same pick is fine but lifting their sentence stem is not), trail off mid-thought, turn a
   *decision* move (vote/accept/reject) or an opening into a question, or stack too many
   questions in a row. Discussion turns may still carry the occasional rhetorical/open question.
   Warn-level flags (e.g. a repeated sentence start) are recorded but not repaired by default.

## Configuration

`config.yaml` opens with a **"DIALS THAT MATTER"** guide. The ~12 knobs worth touching:

| Dial | Effect |
|---|---|
| `llm.sampling.dialogue.temperature` | variety / looseness of the chat |
| `utterances.word_budgets` | length and feel of each turn kind |
| `simulation.num_participants` | group size |
| `personas.cooperative_controls.compromise_willingness` | the main "do chats conclude" dial |
| `personas.hard_blocker_probability` | how often a chat *can't* conclude |
| `personas.coalition_probability` | how often participants share a preference |
| `scenario.acceptance_score` | bar for "I can live with this" |
| `conversation.concentration_to_narrow` | how aligned before it goes to a vote |
| `conversation.discussion_depth` | shapes the derived, per-run discussion length |
| `conversation.force_narrow_turns_per_participant` / `hard_max_turns_per_participant` | narrow-anyway / hard-stop caps, scaled by group size |
| `conversation.moderator_stall_window` / `moderator_max_interventions` | moderator presence |

Everything below the guide is a structural constant (routing weights, validation
thresholds, prompt-window sizes) you normally set once.

Provider is `llm.provider`: `uni` (Bamberg Ollama endpoint), `groq`, or `gemini`. There is
no offline/mock provider and no fabricated fallback anywhere — every scenario, persona, and
turn comes from a real model call, and if a call returns something unusable the run raises
rather than papering over it. Evaluate and test against a real provider (`uni`). API keys
(`GROQ_API_KEY`, `GOOGLE_API_KEY`) come from `.env`.

## Outputs (per run, under `logs/<run_id>/`)

- `transcript.md` — human-readable setup, chat, outcome, metrics, token totals.
- `run.json` — full structured run (scenario, personas, every turn with its act/issues).
- `logs/metrics.csv` — master file, one appended row per run for cross-run evaluation.
- `prompts.jsonl` — every prompt for the run, only when `output.write_prompts: true`.

Key metrics: `outcome_status`, `final_support_fraction`, `repaired_turns` (turns an actual
repair ran on), `flagged_turns` (turns left with any validation flag — usually warn-level),
`question_density`, `avg_words_per_turn`, `option_coverage`, token totals.

## Running

Use the project virtualenv (`dspro`). Interactive single run:

```powershell
(dspro) PS C:\Users\Andi\Desktop\ds_proj> py .\main.py
Topic: Plan a weekend team offsite
```

Batch from a file (one topic per line, `#` comments ignored):

```powershell
(dspro) PS C:\Users\Andi\Desktop\ds_proj> py .\main.py scenarios.txt
```

A run needs a reachable provider: `uni` requires the Bamberg VPN; `groq`/`gemini` need their
API key in `.env`. There is no offline mode — if the endpoint is unreachable the run raises.

### Running and evaluating from the assistant

When the VPN is connected, `uni` runs work end to end, so the assistant can drive a run
itself instead of asking for transcripts:

```powershell
# pipe a topic into the interactive prompt, using the dspro venv directly
"Example (1-liner) Topic" | & .\dspro\Scripts\python.exe .\main.py
```

The run streams the header and every turn to stdout and writes the full logs under
`logs/<run_id>/`. After it finishes, read the newest `logs/<run_id>/transcript.md` (and
`run.json` for per-turn acts and validation issues) to evaluate naturalness, convergence,
and any remaining errors.

## Status & next steps (evaluation as of 2026-06-18)

A naturalness pass (more-friction target) landed and was verified on fresh 2-, 3-, 4-, and
7-person runs across varied topics. The previously-listed issues are addressed; recording the
current state here so the next session can continue without re-deriving it.

### Fixed in this pass
### Remaining (lower priority)
- **Thematic opener similarity** in the vote round (warn-level `REPEATED_START`): lexically varied
  but same theme — the genuinely hard, *thematic* part. Pushing it harder risks over-repair or
  stilted text.
- **Possessive "X's <feature>"** still appears (now alongside varied constructions); discouraged
  in the prompt but natural enough that it isn't forced out.

### How to reproduce the evaluation
Run a spread of group sizes (`simulation.num_participants`, e.g. 3 and 7) and topics from
different domains, then read the newest transcripts. Useful signals already logged:
`min_discussion_turns` (pacing), `question_density`, `avg_words_per_turn`, `repaired_turns`,
`flagged_turns`, `outcome_status`. For repetition, scan participant turns for high word-overlap
pairs (jaccard ≥ 0.6) and count distinct first-3-word openers; the echo guard now catches verbatim
cross-speaker lifts, so the remaining signal to watch is *thematic* (same dimension, varied words).

