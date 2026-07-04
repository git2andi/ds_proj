# Configuration and running

**Code:** `config.yaml`, `src/config_loader.py` (`Config`, validation),
`main.py`, `src/llm_client.py`.

`config.yaml` is the single place for tunable parameters; `config_loader` loads it,
validates it (fail-fast on bad values), and exposes it as `cfg`. Everything the other
notes describe is switched or tuned here.

## How to run

```powershell
py .\main.py                                   # reads the topic from a prompt / stdin
py .\main.py scenarios.txt                     # a file of topics
"Choose a coffee machine for the office" | py .\main.py
```

`main.py` forces UTF-8 stdout (for Windows consoles) and runs one `DialogueRunner`.
There is no LLM-free test suite; validation is by live runs (see `07`).

## Provider / model

```yaml
llm:
  provider: "gpt"                 # uni | groq | gemini | gpt
  models: { gpt: "gpt-4.1-mini", … }
  sampling:                       # setup/repair run cold, dialogue runs warm
    dialogue: { temperature: 0.82, … }
```

`src/llm_client.py` abstracts the provider. Keys are read from `.env`. The project
standard is `gpt` / `gpt-4.1-mini` — do not switch provider/model without an explicit
reason.

## The two input modes (the big levers)

These make controlled experiments possible — fix one side and vary the other:

```yaml
environment:
  mode: auto        # auto = topic -> scenario via LLM;  manual = author it below
  manual: { … }     # topic, opening_question, shared_context, exactly N option cards

participants:
  mode: auto        # auto = sample the cast;  manual = define profiles below
  profiles: [ … ]   # partial or complete; complete cast skips the persona LLM call
```

See `01` (environment) and `02` (participants) for the full field semantics. With a
manual environment **and** a complete manual cast, setup runs with **zero LLM calls**
— a fully deterministic world and cast.

## Moderator (issue 7)

```yaml
moderator:
  enabled: true
  opening: true
  mid_discussion_nudges: true
  final_vote_call: true
  closing: true
```

Gates only the moderator's visible turns; the controller still drives structure. Set
`enabled: false` (or individual flags) for lower-/no-moderator, peer-to-peer runs
(`04`).

## Pacing (`conversation:`)

Turn caps are derived per participant (`_derive_pacing`): `min` turns before any
narrowing, a `target` where the controller starts pushing to a vote, and a hard `max`
that forces a visible vote. Split/stubborn casts get bonus turns.

```yaml
conversation:
  min_discussion_turns_per_participant: 4.0
  target_discussion_turns_per_participant: 5.0
  max_discussion_turns_per_participant: 7.0
  max_vote_rounds: 2
  moderator_max_interventions: 1          # cap on mid-discussion nudges
  require_option_coverage_before_vote: true
```

## Scenario & personas shape

```yaml
scenario:
  option_labels: ["A","B","C","D"]        # -> four options per scenario
  public_attr_min/max: 3 / 5              # attributes per card
personas:
  trait_ranges: { … }                     # OCEAN sampling bounds (auto mode)
  hard_blocker_probability: 0.06
  preference_distribution:                # initial-preference split per group size
    shape_weights: { 3: {"1-1-1":0.5, "2-1":0.45, "3":0.05}, … }
    forced_shape: null                    # e.g. "2-1" to force a split
```

## Routing, style, utterances

```yaml
routing:  { direct_address_probability, quiet_speaker_boost, move_weights,
            target_window }               # speaker/target selection (03)
style:    { name_prefix_max_fraction, i_opening_max_fraction, … }  # surface variety (03)
utterances:
  recent_turns_in_prompt: 5
  word_budgets: { opening:22, discussion:24, ask:22, answer:24, vote:18, … }  # soft targets
```

## Validation & grounding

```yaml
validation:
  enabled: true
  grounding_check: true
  grounding_mode: "tripwire"   # tripwire = judge only suspicious lines; always = judge all
  grounding_acts: [ … ]        # which act types are ever fact-checked
consensus:
  majority_fraction: 0.51      # share of visible votes for "majority" (unanimity = "successful")
```

See `05`/`06` for how these drive turn validation and the outcome.

## Corpus presets (optional, default off)

```yaml
corpus:
  preset: null                 # null = behave exactly as configured; e.g. "delidata"
  presets: { delidata: { turns_per_participant, preferred_group_size,
                         top_speaker_share, dominance_range, imbalance_tolerance } }
```

An active preset folds corpus statistics into runtime parameters at load time:
discussion length → turn caps, preferred group size → `num_participants`, and
dominance targets switch the speaker router from strict equalization to share-aware
weighting (`03`). Soft targets, not hard constraints.

## Reproducibility & output

```yaml
simulation: { num_participants: 3, min/max_participants: 2/7, random_seed: null }
output:     { log_dir: "logs", write_prompts: false, transcript_file, json_file, metrics_csv }
limits:     { warn_total_input_tokens: 30000 }   # warn only; nothing is stopped
```

Set `random_seed` to an int to make the **controller's** decisions reproducible (LLM
text still varies with provider sampling). `write_prompts: true` dumps every prompt to
`prompts.jsonl` for debugging (large files).

## Current mismatch / intended correction

Manual environment and manual participant modes exist, but they should become the main development path for simulator behavior, not just convenience features. Auto/auto runs are useful for demos, but they change too many variables at once to debug whether parameters actually caused behavior changes.

The intended workflow should include fixed manual configurations for controlled development: same environment with high/medium/low engagement casts, a stubborn holdout cast, a no-moderator negotiation cast, and different responsiveness settings. A future routing config should also make the participation model explicit, for example equalized versus parameter-weighted routing.

