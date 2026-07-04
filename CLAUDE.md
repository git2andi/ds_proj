# CLAUDE.md

Use this file as project context for Claude Code or similar coding agents.

## What this project does

An **option-grounded multi-user decision simulator** (not an open-ended group-chat or agenda-based user simulator). A topic is turned into a controlled scenario with four factual options. Then 2-7 configurable simulated users discuss the options and either reach a unanimous decision, a visible majority, or remain unresolved.

The important design choice is separation of responsibilities: the controller decides who speaks, what kind of conversational move is needed, and when voting happens; the LLM writes exactly one natural message for that move.

## How to run

```powershell
py .\main.py
py .\main.py scenarios.txt
"Choose a coffee machine for the office" | py .\main.py
```

Change participant count and provider settings in `config.yaml`. The default provider is `gpt` (`gpt-4.1-mini`); keys are read from `.env`. There is no automated test suite: validation is by live runs — inspect the transcript **and** `run.json`/metrics (see `docs/todo.md` section 5 and `info/07`). Sanity-compile with `py -m py_compile src/*.py main.py`.

## Current source layout

- `main.py`: entry point (forces UTF-8 stdout for Windows consoles).
- `config.yaml`: tunable parameters.
- `src/config_loader.py`: loads and validates `config.yaml`; exposes `cfg`.
- `src/prompts.py`: all LLM prompts and moderator templates.
- `src/dialogue.py`: the orchestration loop — phase control, generation+repair
  pipeline, moderator turns, pacing, logging. `DialogueRunner` mixes in the three
  concern modules below (issue 8 split; methods share the same `self`/`DialogueState`).
- `src/policy.py`: `PolicyMixin` — routing policy (choose speaker/act/target, vote
  readiness, candidate selection, word budgets, surface-style intent flags).
- `src/observer.py`: `ObserverMixin` — parse generated text, apply semantics
  (votes/blockers/lean/switch events), manage response obligations and open questions.
- `src/validation.py`: `ValidationMixin` + `ValidationReport` — turn-text validation,
  grounding tripwire/judge, deterministic restate-first fallback.
- `src/consensus.py`: outcome logic (`ConsensusManager`, `participant_turn_count`), visible-vote only.
- `src/builders.py`: setup generation and persona parsing.
- `src/simulator.py`: OCEAN→parameter derivation and a per-persona private
  communicative-goal list (a weak hint system consulted in quiet moments — this is
  *not* agenda-based user simulation; never describe it as such).
- `src/models.py`: typed state (scenario, personas, runtime, obligations, coverage).
- `src/parsing.py`: option matching and visible-commitment/vote parsing.
- `src/aliases.py`: the single option-alias contract (`short_alias_map`).
- `src/style.py`: deterministic surface-style tracker (name/option openings, repeated templates).
- `src/llm_client.py`: provider abstraction (uni | groq | gemini | gpt).
- `src/evaluation.py`: per-run metrics incl. parameter-realization checks
  (participation gini, engagement/verbosity realization errors and correlations,
  response/question completion, repetition, compromise success, switch
  explanation/bridge rate). `run.json` carries a `runtimes` section (visible
  state + `switch_events` per sim) and a `phase_history` list (closure entries
  appear only on a resolved outcome).
- `src/logger.py`: run logging (transcript, JSON, metrics CSV). The transcript
  header and `run.json` carry the active LLM `Provider`/`Model`; the metrics
  CSV has `llm_provider`/`llm_model` columns (issue 9).
- `src/utils.py`: deterministic helpers (normalisation, weighted choice, JSON, `clean_generated`).
- `info/`: plain-language "how it works" notes, one per stage of a run
  (`00_overview` is the map; `01`–`09` follow scenario → sims → routing → moderator →
  discussion/decision → consensus → evaluation → config → topic examples). Kept in
  sync with the code.
- `docs/todo.md`: open issues and the per-issue implementation protocol.

## Key controller mechanisms (current as of 2026-07-04)

- **Environment input modes.** `environment.mode` in `config.yaml`: `auto` turns a
  CLI/stdin topic into a scenario via the setup LLM; `manual` builds the Scenario
  deterministically from `environment.manual` (topic, opening question, shared
  context incl. hard caps, exactly four option cards) with no scenario LLM call and
  no CLI topic needed. Authored cards are never rewritten — a cap violation is a
  startup config error. `run.json` records `environment_mode`. Combined with a
  complete manual cast, setup runs with zero LLM calls.
- **Participant input modes.** `participants.mode` in `config.yaml`: `auto` samples
  the cast as before; `manual` defines it via `participants.profiles` (group size =
  profile count). Profiles may be partial — missing fields are filled by the auto
  path; `parameters:` override trait-derived simulator values; a profile `rejection`
  makes a hard blocker (agreeableness pinned to 1). A fully specified cast skips the
  persona LLM call entirely (deterministic simulators for controlled experiments).
  `run.json` records `participants_mode`.

- **Transcript-safe fallbacks.** A turn that still carries blocking validation
  issues after the repair pass is never printed. `_safe_fallback_text` substitutes
  a deterministic, parser-clean line for the intent (a hard blocker commits to an
  allowed alternative, an unclear vote becomes one clear commitment, a coverage
  turn names the required option) from a pool of 8 direct-vote templates rotated
  via `avoid_phrases`. Metrics: `fallback_turns`, `invalid_printed_turn_count`
  (must stay 0).
- **Complete-sentence truncation.** Word budgets are style targets, not hard
  bounds: `utils.clean_generated` keeps a complete sentence that lands within a
  soft cap (budget + max(8, 40%)); a required cut falls on the last real sentence
  boundary inside the soft window (decimal-point safe); fragment salvage is a
  last resort for one runaway sentence. No printed line may end mid-thought.
- **Response obligations.** A direct question (moderator→participant or
  participant→participant), detected from visible text, sets
  `DialogueState.response_obligation`. The router consumes it before normal
  speaker selection in both the discussion and decision loops, so the addressed
  participant answers within the next turn. Obligations expire after a bounded
  number of turns and are counted as `unanswered_direct_questions`.
- **Surface-style control.** `src/style.py` tracks the last few participant turns
  for name-prefix density, option-name openings, first-person "I …" opening
  density, repeated opening words, and repeated
  concession/worry/trade-off templates. The controller strips non-functional name
  prefixes, biases act selection away from templated streaks, and passes compact
  prompt flags (suppress name/option opening, vary opening). Word budgets are
  trait-driven (`_word_bounds`) so verbosity/engagement produce a visible length
  spread with ±10% per-turn jitter; `sim_utterance` length/tone guidance is
  concrete and parameter-driven. Decision turns avoid commitment-phrase families
  used this round *or by the same speaker in any earlier turn* plus justification
  snippets from the round (`avoid_phrases` / `avoid_reasons`); the vote repair
  prompt builds its example menu dynamically from unused families; BUILD move
  purposes rotate.
- **Trait-weighted participation.** Each sim has a target turn share from
  `simulator.expected_turn_share` (engagement-dominant, plus initiative/
  responsiveness and a floor). `_choose_speaker` pulls actual share toward that
  target (`exp(routing.trait_share_adaptation * gap)`), with an anti-monopoly
  damp past `max_share_overshoot`, a minimum-visibility push after
  `max_silence_rounds` silent rounds, and the ping-pong penalty. Responsiveness
  gates how promptly an obligated sim answers (may defer one beat, never lapses
  from hesitation). Evaluation uses the same share targets
  (`expected_turn_share`/`realized_turn_share` in metrics).
- **Stateful stance tracking.** Each runtime carries `commitment_strength`
  (init `0.45 + 0.4*stubbornness`; eroded by challenges on the favorite and
  visible rival support, rebuilt by defending it), `challenges_received`,
  `concessions_made`. Visible objections open bounded **concern threads**
  (`DialogueState.open_concerns`, expire after ~3 participant turns): the
  router routes an advocate reaction within 1–2 turns — firm advocates defend,
  a shaken one (commitment ≤ 0.35) is told to concede honestly. Commitment
  feeds the latent-lean gate and vote-time compromise probability. Metrics:
  `concern_threads`, `concern_response_rate`. Blocker parsing is personal-veto
  only (speculative/other-directed "might be a dealbreaker for some" never
  binds the speaker). Group-directed questions pick their respondent by
  responsiveness + share deficit, not "previous speaker always".
- **Reactive act selection.** `_reactive_intent` fires before the agenda:
  open concern threads get an advocate reaction, answers get follow-ups,
  an unresolved blocker on the leading option is probed once, visible
  splits trigger head-to-head comparisons, and a circling thread (four turns
  with no question or parsed movement signal while two camps persist) gets one
  bounded criteria-level compromise/ask beat (`stagnation_break_done`). The
  per-persona agenda only fills quiet moments. Challenge reasons are
  stance-aware (never argue against the speaker's own pick).
- **Same-speaker continuations (issue 6).** After reactive checks, the previous
  speaker may take one short follow-up (afterthought/clarification/self-
  correction) with probability `0.03 + 0.07*initiative`; chain capped at 3
  (second continuation half as likely), reduced budget, and a blocking
  `CONTINUATION_REPEATS` check (Jaccard overlap with own previous line, or
  re-asking the same addressee) with a parser-neutral fallback. Normal turns
  still hard-exclude the last speaker. Metric: `continuation_turns`.
- **Thread-scored targeting.** `_choose_target_turn` scores the last few
  participant turns (open questions, objections/blockers, minority voices,
  leading/under-discussed options) instead of always answering the latest line;
  answer acts deterministically target the pending question.
- **Vote stability.** Later vote rounds only re-prompt unclear/non-voters, and
  `_set_vote` keeps an existing clear vote unless the text explicitly signals a
  change (e.g. "actually I vote for", "switch to").
- **Sanctioned switches.** On turns where the controller explicitly invites a
  vote change (`intent.allow_vote_change`, i.e. minority check and split-vote
  compromise), the parser accepts a commitment with a concessive bridge clause
  ("X works for me even though …", "as long as …"); questions and genuine
  prerequisites ("only if", "unless") still block. Everywhere else commitment
  parsing stays strictly conservative.
- **Coverage is bounded.** Each option gets at most one coverage nudge
  (`OptionCoverage.coverage_attempts`), so a detection miss cannot loop forever.
- **Visible-evidence vote readiness.** Early narrowing requires a visible support
  cluster (or visible support plus a visible compromise proposal) with no open
  question and no active blocker on the candidate; latent concentration never
  triggers voting. `_candidate_for_vote` scores visible votes/acceptances/
  proposals; latent lean only breaks ties. Hard turn caps still force a visible
  vote.
- **Visible-evidence stance movement.** Latent lean (`current_preference`) moves
  only on parsed signals (vote/acceptance, compromise offer, proposal,
  conditional support, softening line) gated by `_can_shift_to`, which also
  respects parsed runtime blockers. Mid-discussion movement (issue 3): a
  parsed softening phrase ("starting to make more sense to me") moves the lean
  without ever parsing as a vote, and a once-per-sim routed softening beat
  (`soften_toward` on the intent; eroded commitment or sustained pressure +
  visibly-backed attractor) makes a shaken sim say so during discussion
  instead of silently flipping at the vote. Metric: `discussion_lean_shifts`. Committing to an actively blocked option is a blocking
  validation issue (`BLOCKED_OPTION_ACCEPTED`) unless the same line resolves the
  blocker; sanctioned switches may only land on offered/current/initial options
  (`OFF_TARGET_SWITCH`). Vote-time compromise (`_should_compromise_to_candidate`)
  requires visible support or a visible proposal, never latent concentration.
  Vote movements are recorded as `switch_events` (from→to, has_reason, has_bridge).
- **Bridged switches.** A parsed commitment that lands on an option other than the
  sim's current internal lean must bridge the move: `parsing.switch_bridge_ok`
  requires the old option named or an explicit concession marker (`_CONCESSION`)
  plus a reason clause. A missing bridge is the blocking issue `UNBRIDGED_SWITCH`
  (repaired with the old pick named; if repair still fails, the restate-first
  fallback keeps the current lean rather than printing an unexplained flip).
  `switch_events` carry `has_bridge` (checked against the pre-turn lean) and
  `evaluation.py` reports `switch_bridge_rate`.
- **Blocker vocabulary.** The parser detects option-tied active blockers
  ("dealbreaker", "doesn't work for me", with negation guard), explicit blocker
  resolutions ("that fixes my concern; I can live with X"), conditional support,
  and compromise offers (incl. question forms). Parsed blockers accumulate in
  `ParticipantRuntime.hard_rejections`; a visible resolution clears parser-derived
  entries but never the persona-level setup rejection.
- **Alias safety.** Option aliases exclude stopwords/generic words (so "with",
  "data", "analytics", "warehouse", etc. never match) and include distinctive
  proper nouns ("Gin", "Rails").
- **Natural consensus calls.** The moderator asks for definite picks in plain
  language, never dictates a quoted vote formula, and vote calls are
  option-neutral (the candidate is never named in the question); participants
  commit in their own words, and the parser's commitment patterns cover the
  natural forms ("I'd go with", "my vote's on", "I'm all in for") while
  hedges/conditionals still block.
- **Reservation negotiation (issue 4).** The minority check and split-vote
  compromise both embed one two-turn reservation exchange
  (`_reservation_exchange`, once per run): the most movable holdout states a
  concrete reservation (no vote), one supporter responds honestly, then the
  normal closing beats run. With `final_vote_call` off, a high-initiative
  supporter asks the holdout probe (`_emit_peer_holdout_probe`) instead of the
  moderator. The hard turn cap forces the vote but no longer starves these
  bounded passes. Metric: `reservation_exchange`.
- **Targeted moderation.** Stall nudges prefer concrete visible issues: probe an
  unresolved blocker on the candidate once, ask the group to weigh a visible
  split head-to-head, or address the single holdout — generic "strongest
  remaining concern" only as last resort. The split-vote compromise probe
  (`_split_probe_candidate`) never targets an option with a visible unresolved
  dealbreaker and requires at least one dissenter who can actually move; its
  wording presents the front-runner as "currently has the most support" only when
  it is a strict plurality (a pure tie is announced as "evenly split with no option
  ahead"), with both answers explicitly fine. Closures are status-aware: a majority close
  names the holdouts and never implies full agreement; unresolved closes
  present nothing as chosen.
- **Configurable moderator.** `moderator:` in `config.yaml` gates the visible
  moderator *voice* with independent booleans — `enabled` (master switch),
  `opening`, `mid_discussion_nudges`, `final_vote_call`, `closing` — via
  `DialogueRunner._mod(part)`. The flags never touch controller policy: in
  lower-/no-moderator modes the run still decides because the decision loop keeps
  emitting participant vote turns and the participant-level narrowing acts carry
  the discussion (peer-to-peer). Participant-owned procedure (issue 5): with
  `final_vote_call` off, the highest-initiative sim casually calls for final
  picks (option-neutral) and a supporter asks the holdout probe; with
  `mid_discussion_nudges` off, a bounded stall beat (≤2/run) has a
  high-initiative sim summarize the visible split / suggest setting aside an
  unargued option / suggest deciding. Metrics: `participant_procedural_moves`,
  `peer_vote_call`. When `opening` is off the option board is shown as
  plain scaffolding (header + transcript `## Options`), not a turn. Defaults are
  fully-moderated; `run.json` records the resolved `moderator_config`. The split-vote
  probe only claims "the most support" on a strict plurality — a pure tie is
  announced as "evenly split with no option ahead".
- **Corpus presets (optional).** `corpus.preset` in `config.yaml` (default null)
  folds corpus statistics into runtime parameters at load time: typical turns per
  participant → turn caps, preferred group size → `num_participants`. Dominance
  targets (`top_speaker_share`, `dominance_range`, `imbalance_tolerance`) switch
  `_choose_speaker` from strict equalization to share-aware weighting
  (`utils.preset_dominance_weight`). Soft targets, not hard constraints; with no
  preset the simulator behaves exactly as configured.
- **Setup repair.** Persona rows that drop/reorder the controller-assigned primary
  preference are repaired deterministically (`builders.repair_preferred_options`)
  rather than retried; a rejection of the required option still fails the attempt.
- **Tripwire grounding.** `validation.grounding_mode: tripwire` (default) calls
  the LLM fact-judge only when a regex tripwire finds a suspicious concrete
  claim (a number, a policy/medical/weather-style term, or an experiential/
  operational claim — parking, wifi, crowds, traffic, staffing — absent from
  the option cards/shared context) or a cross-option fact transfer: a line that names
  option X while using another card's distinctive tokens, or mixes two cards'
  distinctive tokens inside an explicit comparison. The judge flags invented
  facts, wrong-option attribution, and unlike-unit comparisons; `always`
  restores per-turn judging. Decision acts are grounded too. A flagged line
  gets one extra grounding-only repair pass before printing (issue 7); metrics
  split `unsupported_fact_flags` (caught anywhere) from
  `unsupported_printed_turns` (must stay ~0). The per-turn
  prompt sends a voice capsule instead of raw OCEAN/parameter dumps (n=3 runs
  ≈ 25-35k input tokens with the cross-option gates and grounding repair).
- **Hard-cap enforcement.** Hard numeric caps in shared context (budget, distance,
  duration — soft "around $X" phrasings excluded) are extracted by
  `builders.shared_context_caps` with unit normalization inside a family
  (hours↔minutes, miles↔km), units read from the attribute key when the value
  is bare ("duration_minutes: 130"), and activity scoping ("within 15 minutes
  *walking*" never binds a wait time). Early setup attempts retry generation on
  a violation; only the final attempt clamps in place (`enforce_shared_caps`,
  floored, per-unit basis respected) — rewriting a number can fabricate a false
  fact about a real-world named option. Repairs are recorded in
  `Scenario.setup_notes`. Persona goals must be consistent with the assigned
  primary preference (setup prompt rule).

## Current direction

Keep the codebase small. Avoid reintroducing an over-complex rule stack. Improve quality by changing the compact controller, the prompt, or a narrow validator only when a transcript shows a repeated issue.

The discussion should contain natural multi-party behavior: agreement, challenge, questions, answers, comparisons, invitations to quieter participants, and plausible compromise. It should not become a rigid debate template.

## Non-negotiable rules

- Never count hidden preference as final support.
- Never close before participants have a visible decision opportunity.
- Never let a hard blocker accept their rejected option through state mutation.
- Never add facts outside option cards/shared context.
- Keep the moderator sparse and neutral (its voice is configurable via `moderator:`, but the default stays sparse).
- Put all LLM-facing prose in `src/prompts.py`.
- Prefer controller/parser/validator/state fixes over enlarging prompts.
