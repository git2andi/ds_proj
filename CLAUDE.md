# CLAUDE.md

Use this file as project context for Claude Code or similar coding agents.

## What this project does

It generates small group-decision transcripts. A topic is turned into a controlled scenario with four factual options. Then 2-7 personas discuss the options and either reach a unanimous decision, a visible majority, or remain unresolved.

The important design choice is separation of responsibilities: the controller decides who speaks, what kind of conversational move is needed, and when voting happens; the LLM writes exactly one natural message for that move.

## How to run

```powershell
py .\main.py
py .\main.py scenarios.txt
"Choose a coffee machine for the office" | py .\main.py
```

Change participant count and provider settings in `config.yaml`. The default provider is `gpt` (`gpt-4.1-mini`); keys are read from `.env`. Run the tests with `py -m pytest tests/ -q` (no LLM calls).

## Current source layout

- `main.py`: entry point (forces UTF-8 stdout for Windows consoles).
- `config.yaml`: tunable parameters.
- `src/config_loader.py`: loads and validates `config.yaml`; exposes `cfg`.
- `src/prompts.py`: all LLM prompts and moderator templates.
- `src/dialogue.py`: compact discussion controller — routing, obligations, voting, moderator.
- `src/consensus.py`: outcome logic (`ConsensusManager`, `participant_turn_count`), visible-vote only.
- `src/builders.py`: setup generation and persona parsing.
- `src/simulator.py`: OCEAN→parameter derivation and per-persona agenda.
- `src/models.py`: typed state (scenario, personas, runtime, obligations, coverage).
- `src/parsing.py`: option matching and visible-commitment/vote parsing.
- `src/aliases.py`: the single option-alias contract (`short_alias_map`).
- `src/style.py`: deterministic surface-style tracker (name/option openings, repeated templates).
- `src/llm_client.py`: provider abstraction (uni | groq | gemini | gpt).
- `src/evaluation.py`: lightweight metrics (separate from logging).
- `src/logger.py`: run logging (transcript, JSON, metrics CSV).
- `src/utils.py`: deterministic helpers (normalisation, weighted choice, JSON, `clean_generated`).
- `tests/`: deterministic, LLM-free tests.
- `info/`: conceptual design notes for intended behavior.
- `docs/todo.md`: open issues and the per-issue implementation protocol.

## Key controller mechanisms (current as of 2026-07-02)

- **Transcript-safe fallbacks.** A turn that still carries blocking validation
  issues after the repair pass is never printed. `_safe_fallback_text` substitutes
  a deterministic, parser-clean line for the intent (a hard blocker commits to an
  allowed alternative, an unclear vote becomes one clear commitment, a coverage
  turn names the required option). Metrics: `fallback_turns`,
  `invalid_printed_turn_count` (must stay 0).
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
  spread; `sim_utterance` length/tone guidance is concrete and parameter-driven.
- **Speaker balance.** `_choose_speaker` weights by turn-count deficit and
  penalizes the second-to-last speaker to stop two participants ping-ponging.
- **Reactive act selection.** `_reactive_intent` fires before the agenda:
  challenged options get defended by an advocate, answers get follow-ups,
  an unresolved blocker on the leading option is probed once, and visible
  splits trigger head-to-head comparisons. The per-persona agenda only fills
  quiet moments. Challenge reasons are stance-aware (never argue against the
  speaker's own pick).
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
  conditional support) gated by `_can_shift_to`, which also respects parsed
  runtime blockers. Committing to an actively blocked option is a blocking
  validation issue (`BLOCKED_OPTION_ACCEPTED`) unless the same line resolves the
  blocker; sanctioned switches may only land on offered/current/initial options
  (`OFF_TARGET_SWITCH`). Vote-time compromise (`_should_compromise_to_candidate`)
  requires visible support or a visible proposal, never latent concentration.
  Vote movements are recorded as `switch_events` (from→to, has_reason).
- **Blocker vocabulary.** The parser detects option-tied active blockers
  ("dealbreaker", "doesn't work for me", with negation guard), explicit blocker
  resolutions ("that fixes my concern; I can live with X"), conditional support,
  and compromise offers (incl. question forms). Parsed blockers accumulate in
  `ParticipantRuntime.hard_rejections`; a visible resolution clears parser-derived
  entries but never the persona-level setup rejection.
- **Alias safety.** Option aliases exclude stopwords/generic words (so "with",
  "data", "analytics", "warehouse", etc. never match) and include distinctive
  proper nouns ("Gin", "Rails").
- **Natural consensus calls.** The moderator asks where people land in plain
  language and never dictates a quoted vote formula; participants commit in
  their own words, and the parser's commitment patterns cover the natural forms
  ("I'd go with", "my vote's on", "I'm all in for") while hedges/conditionals
  still block.
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
- **Hard-cap enforcement.** Hard numeric caps in shared context (budget, distance,
  duration — soft "around $X" phrasings excluded) are extracted by
  `builders.shared_context_caps` and violating option attributes are clamped in
  place (`enforce_shared_caps`), respecting per-unit basis. Repairs are recorded
  in `Scenario.setup_notes`. Persona goals must be consistent with the assigned
  primary preference (setup prompt rule).

## Current direction

Keep the codebase small. Avoid reintroducing an over-complex rule stack. Improve quality by changing the compact controller, the prompt, or a narrow validator only when a transcript shows a repeated issue.

The discussion should contain natural multi-party behavior: agreement, challenge, questions, answers, comparisons, invitations to quieter participants, and plausible compromise. It should not become a rigid debate template.

## Non-negotiable rules

- Never count hidden preference as final support.
- Never close before participants have a visible decision opportunity.
- Never let a hard blocker accept their rejected option through state mutation.
- Never add facts outside option cards/shared context.
- Keep the moderator sparse and neutral.
- Put all LLM-facing prose in `src/prompts.py`.
- Prefer controller/parser/validator/state fixes over enlarging prompts.
