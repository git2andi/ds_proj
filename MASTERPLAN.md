# MASTERPLAN — Scientifically Grounded Multi-Party Dialogue Simulator

**Project root:** `c:\Users\andi\Desktop\ds_proj\`
**Source basis:** This plan merges two independent technical analyses of the current codebase against the five provided papers (Fisher 1970, Sacks/Schegloff/Jefferson 1974, MUCA 2024, Ouchi & Tsuboi 2016, McCrae & John 1992) and against [CLAUDE.md](CLAUDE.md). Where the two analyses disagreed, this document explicitly resolves the disagreement and explains why.

This is the single document to work from. Stages are independently shippable. Each stage cites the specific modules and config keys it touches.

---

## Part I — Diagnosis

### 1. Where the system is right

The architecture has the correct top-level objects: [Orchestrator](src/orchestrator.py), [PersonaBuilder](src/persona.py), [Simulator](src/simulator.py), [TurnManager](src/turn_manager.py), [ConsensusDetector](src/consensus.py), [ModerationEngine](src/moderation.py), [DialogueLogger](src/logger.py), [LLMClient](src/llm_client.py), centralized [config_loader.py](src/config_loader.py) reading [config.yaml](config.yaml), and a [prompts.py](src/prompts.py) registry. The system's operational phase backbone (greeting → opening → negotiation → narrowing → emergence → confirmation → closure), inspired partly by Fisher's four-phase decision-emergence model, the `AgentBeliefs` private stance object (preferred / acceptable / rejected / key_concern / concession), the layered consensus detector with reduced-opposition tier, and the moderator escalation ladder are all sound choices. The 4-option fixed-letter design simplifies stance extraction. [CLAUDE.md](CLAUDE.md) already documents the intent clearly and warns against pitfalls the team has already noticed (phrase blacklists, hollow validation, invented facts).

### 2. Where the system is wrong (root causes)

The four root structural problems:

**A. The transcript is the source of truth.** Almost every behavior — who voted what, who got addressed, whether a question is pending, who is repeating, who has a blocker — is re-extracted from `history: list[str]` via regex on every turn. See [utils.extract_preference_vote](src/utils.py#L23-L69), [turn_manager.extract_discourse](src/turn_manager.py#L198-L220), [orchestrator._fresh_unanswered_question](src/orchestrator.py#L714-L732). This produces brittle logic and forces every layer to re-parse strings.

**B. Personality expressed as named phrases.** [persona.personality_summary()](src/persona.py#L179-L208) inserts adversarial example phrases for low-A/high-N personas ("yeah but that's not actually true", "but what if that doesn't work out?"). The LLM mimics these as scripts. This violates [CLAUDE.md](CLAUDE.md)'s own rule about not listing specific filler phrases, and it is the structural reason stubbornness fires too often. Following McCrae & John's view of the Big Five as broad personality dimensions, this system should operationalize traits as **probabilistic behavioral biases rather than deterministic scripts**.

**C. Phases advance on turn counts, not content evidence.** [orchestrator._update_phase()](src/orchestrator.py#L256-L276) derives phases purely from `participant_turn_count`, `has_asked_narrowing`, and `has_entered_emergence`. Fisher's actual signal is the *ratio shift* between favorable / unfavorable / ambiguous statements. The current "emergence" phase means "all participants have voted via regex" — that is not Fisher.

**D. Turn-taking collapses SSJ rule cascade into a weighted sum.** [turn_manager._score()](src/turn_manager.py#L75-L126) is a sum of 9–10 hand-tuned constants. SSJ specifies a strict priority order: 1a (current selects next, obligated) > 1b (self-select, first-starter wins) > 1c (current continues). A question to Drew is not a +0.90 weight; it is a hard obligation.

Two cross-cutting consequences:

**E. Prompt bloat ≈ 60k in / 3k out.** Every sim turn re-sends persona block (backstory, goal, personality summary, beliefs ~250 tokens), the full options block (~150 tokens), the voice/style block (~400 tokens hardcoded in [prompts.sim_turn](src/prompts.py#L158-L229)), forbidden openers, forbidden frames, position discipline, interaction instructions. ~800 of every ~1500 input tokens are stable scaffolding that should be sent once and cached.

**F. LLM-facing text is not fully centralized.** [CLAUDE.md](CLAUDE.md)'s hard requirement is that all LLM prompts live in `prompts.py`, but [simulator.py](src/simulator.py#L88-L177) still constructs phase_instructions, narrowing_instructions, interaction_instructions, position_discipline, and skepticism_nudges as plain Python strings before passing them into `prompts.sim_turn`. This violates the project's own rule.

### 3. Specific failure modes (observed in [logs/](logs/) and [token_log.txt](token_log.txt))

1. **Force-close dominates.** Most recent batch entries in `token_log.txt` show `outcome=force_close`.
2. **Confirmation rollback loop.** A single "no" during confirmation invalidates the candidate, sets cooldown, re-routes priority to the rejecting speaker, often re-tests the same option, then fails again ([orchestrator._run_confirmation](src/orchestrator.py#L426-L496)). The sample log [20260521_142638_641507.txt](logs/20260521_142638_641507.txt) shows this firing twice on Option A and once on Option B before force-close.
3. **Stubbornness over-fires.** Low-A or high-N personas inherit adversarial phrasings, persistent skepticism nudges, and the "name your single concrete dealbreaker" path all at once. The compound effect is far stronger than any single rule. Target rate is ~0–10% of dialogues having a true hard blocker; current rate feels closer to 50–70%.
4. **Vote tracking is brittle.** [extract_preference_vote](src/utils.py#L23-L69) regex misses "yeah, A is fine", "let's go with C", "I'd go with B" (without the literal word "Option"). When the orchestrator thinks a participant hasn't voted, downstream consensus is starved.
5. **Hallucinated facts.** Personas invent attributes (page counts, budgets, ages, "they have a patio area") not present in the option text. [CLAUDE.md](CLAUDE.md) already flags this; there is no fact-check loop.
6. **Filler patterns rotate but don't disappear.** Forbidden openers cover 1–2 words; personas just lengthen them ("To be fair…", "Honestly speaking…").
7. **Force-close picks options for the wrong reason.** [orchestrator._force_conclusion](src/orchestrator.py#L613-L712) mixes private-belief scoring (acceptance += 2.0 for `beliefs.preferred`) with public votes (acceptance += count × 5.0) using a magic 5x weight to "make votes win". This conflates public and private state. Final decision must be justifiable from the **public transcript**, per [CLAUDE.md](CLAUDE.md).
8. **Closure templates ignore rejection.** [simulator._closure_line](src/simulator.py#L560-L570) injects `state.preferred_option` into goodbyes regardless of `rejected_options_by_speaker`.

### 4. Resolved disagreements between the two analyses

Both analyses agreed on most points. Where they diverged, the merged plan adopts:

| Issue | Decision | Why |
|---|---|---|
| "Prompt construction is entirely in `prompts.py`" | **False.** Phase/narrowing/interaction/position-discipline/skepticism text is still built in [simulator.py:91-170](src/simulator.py#L91-L170) and must move. | This violates the project's hard requirement. |
| Hard-cap stubbornness: "after N holdout turns, force concede" | **Rejected.** Replace with **rare explicit hard-blocker sampling at dialogue level**, with config cap (0–10%) and condition-based softening. | Forcing concession produces fake consensus; that contradicts the project goal of plausible failure modes. |
| Cache the system block via API-side prompt caching | **Treat as optional optimization, not the main lever.** Portable improvement is **prompt compression and structured speaker cards**. | Not equally available across `uni`/`groq`/`gemini` providers. |
| Remove fallback options entirely | **Soften to "explicit fallback with logging".** Fail fast in dev, log `option_generation_failed=true` in batch. | Silent fallback hides bugs; full removal hides recoverable failures. |
| `force_close_rate ≤ 30%` as a target | **Use as a diagnostic, not a scientific target.** | True north is "hard-blocker dialogue rate ≤ 10%" and "force-close is explainable from public stance evidence." |
| Fisher phase transition thresholds ("unfavorable > ambiguity → conflict") | **Use as rolling evidence trends, not hard switches.** Store phase confidence, allow short/skipped phases. | Fisher explicitly says phases are gradual and not universal. |
| Replace consensus regex now vs. run in parallel first | **Run new structured state in parallel for at least one full diagnostic batch before switching decision logic.** | This is the safest refactor pattern; it produces comparable A/B data on the same dialogues. |

---

## Part II — Target Architecture

### 5. Eight-layer separation

```
                ┌──────────────────────────────────────────────────────┐
1. State        │  DialogueState (Turn[], StanceTable, DiscourseGraph,│
                │   ParticipantState[], OptionState[], Phase, Budget) │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
2. Turn Policy  │  SSJ rule cascade:                                  │
                │   1a addressed_question > 1b self_select > 1c cont. │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
3. Act Planner  │  Pick DialogueAct given persona bias + stance +     │
                │   phase + obligations (probabilistic)               │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
4. Prompt Build │  Compose compact prompt:                            │
                │   speaker card + act plan + relevant state + local  │
                │   context (≤ 6 turns) + output contract             │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
5. Realize      │  LLMClient.generate() — natural language only       │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
6. State Update │  Deterministic parser (cheap regex + acts) for      │
                │   clear cases; LLM stance classifier only for       │
                │   ambiguous turns, batched every K turns            │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
7. Moderator    │  Phase detection (Fisher evidence), narrowing,      │
                │   compromise, confirmation, best_available_decision │
                │   — all from structured state                       │
                └──────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────────────────────┐
8. Logger /     │  Transcript + per-turn JSONL trace + per-dialogue   │
   Evaluator    │   summary JSON + batch metrics                      │
                └──────────────────────────────────────────────────────┘
```

### 6. New module layout

```
src/
├── config_loader.py           (existing, keep)
├── llm_client.py              (existing, extend with per-component token tracking)
├── prompts.py                 (existing, EXPAND — all LLM-facing text)
├── persona.py                 (slim down — keep dataclasses, move behavior to bias)
│
├── state/                     (NEW — central structured state)
│   ├── __init__.py
│   ├── dialogue_state.py      (DialogueState, Phase, BudgetMeter)
│   ├── turn.py                (TurnRecord, DialogueAct enum)
│   ├── stance.py              (StanceTable, StanceUpdate, OptionState)
│   ├── participant.py         (ParticipantState, participation debt)
│   ├── discourse.py           (DiscourseGraph, pending questions, reply edges)
│   └── tracker.py             (state_tracker — parses each Turn, updates state)
│
├── policy/                    (NEW — decision logic)
│   ├── __init__.py
│   ├── turn_policy.py         (SSJ rule cascade — replaces turn_manager)
│   ├── act_planner.py         (chooses DialogueAct given state + persona)
│   ├── personality_bias.py    (Big Five → probability biases)
│   ├── stubbornness.py        (rare hard-blocker sampling)
│   └── moderator_policy.py    (replaces ModerationEngine decision logic)
│
├── reasoning/                 (NEW — Fisher + consensus)
│   ├── __init__.py
│   ├── phase_detector.py      (Fisher evidence ratios → phase)
│   └── consensus_engine.py    (replaces consensus.py, uses StanceTable)
│
├── realize/                   (NEW — LLM realization layer)
│   ├── __init__.py
│   ├── simulator.py           (slimmed — calls prompt + LLM only)
│   ├── moderator.py           (LLM moderator line generation only)
│   └── grounding.py           (KNOWN_FACTS check + repair)
│
├── orchestrator.py            (slim down — loop coordinator only)
├── logger.py                  (extend with JSONL trace + summary JSON)
└── utils.py                   (keep regex helpers as deterministic fallback)

tools/
├── analyze_logs.py            (NEW — batch metrics computation)
└── run_eval.py                (NEW — fixed-seed eval harness)

eval/
└── eval_scenarios.txt         (NEW — 50 fixed topics)

config.yaml                    (EXPAND — see §9)
```

The existing [orchestrator.py](src/orchestrator.py) becomes the loop coordinator; all decision logic moves out. The existing [turn_manager.py](src/turn_manager.py), [consensus.py](src/consensus.py), [moderation.py](src/moderation.py) can stay temporarily behind a config flag for A/B testing during the transition.

### 7. Central data model

```python
# src/state/turn.py

class DialogueAct(Enum):
    GREET
    OPEN_PRIORITY
    ASSERT_SUPPORT          # Fisher: favorable
    ASSERT_OPPOSITION       # Fisher: unfavorable
    ASSERT_AMBIGUOUS        # Fisher: ambiguous
    ASK_CLARIFICATION
    ASK_PREFERENCE
    ANSWER
    CHALLENGE
    CONCEDE
    CONDITIONAL_ACCEPT
    PROPOSE_COMPROMISE
    COMMIT_VOTE
    CONFIRM
    REJECT_WITH_REASON
    SUMMARIZE
    GOODBYE
    SILENCE

@dataclass
class StanceUpdate:
    speaker: str
    option: str                          # "A".."D"
    stance: Literal["support", "oppose", "ambiguous",
                    "conditional_support", "blocker", "neutral"]
    confidence: float                    # 0..1
    condition: Optional[str] = None      # for conditional_support / blocker

@dataclass
class TurnRecord:
    turn_id: int
    speaker: str                         # "Moderator" or participant name
    text: str
    phase: str
    is_moderator: bool
    addressees: list[str]                # may be empty (open turn)
    reply_to: Optional[int]              # turn_id being responded to
    dialogue_act: DialogueAct
    mentioned_options: list[str]
    stance_updates: list[StanceUpdate]
    is_question: bool
    answers_question_id: Optional[int]
    selected_reason: str                 # why this speaker was picked
    tokens_in: int
    tokens_out: int
    prompt_type: str                     # "sim_turn" | "moderator" | "consensus" ...
```

```python
# src/state/stance.py

@dataclass
class OptionState:
    option_id: str
    text: str                            # the canonical option text (KNOWN_FACTS)
    supporters: set[str]
    opponents: set[str]
    ambiguous: set[str]
    conditional_supporters: dict[str, str]   # name → condition text
    hard_blockers: dict[str, str]            # name → blocker reason
    support_score: float
    opposition_score: float
    last_mentioned_turn: Optional[int]

class StanceTable:
    """Per-(speaker × option) current stance + history. Single source of truth."""
    def current(self, speaker: str, option: str) -> StanceUpdate: ...
    def apply(self, update: StanceUpdate) -> None: ...
    def fisher_ratios(self, window: int = 8) -> dict:
        """Returns {'favor': float, 'disfavor': float, 'ambiguous': float,
                    'conditional': float} over the last `window` participant turns."""
    def majority_favor(self, min_supporters: int) -> Optional[str]: ...
    def holdouts(self, option: str) -> list[str]: ...
    def unresolved_blockers(self, option: str) -> dict[str, str]: ...
```

```python
# src/state/participant.py

@dataclass
class ParticipantState:
    name: str
    persona: Persona                     # existing persona dataclass
    public_preference: Optional[str]     # most recent committed vote
    public_stances: dict[str, str]       # option → stance label
    unresolved_conditions: dict[str, str]
    hard_blockers: dict[str, str]
    open_questions_to_answer: list[int]  # turn_ids of unanswered Qs addressed to me
    turn_count: int
    last_spoke_turn: Optional[int]
    participation_debt: float            # for SSJ self-selection bias
    recent_dialogue_acts: list[DialogueAct]
    strategy_cooldowns: dict[DialogueAct, int]   # MUCA-style cooldown
    is_true_hard_blocker: bool           # set at dialogue start by sampler
```

```python
# src/state/discourse.py

@dataclass
class DiscourseGraph:
    pending_questions: dict[int, list[str]]   # turn_id → expected addressees
    reply_edges: dict[int, int]               # answer_turn_id → question_turn_id
    last_addressed: Optional[str]
    open_invitations: list[int]               # group-directed questions
```

```python
# src/state/dialogue_state.py

@dataclass
class PhaseEvidence:
    phase: Literal["orientation", "conflict", "emergence", "reinforcement", "closure"]
    confidence: float
    favor_rate: float
    disfavor_rate: float
    ambiguous_rate: float
    conditional_rate: float
    rounds_in_phase: int
    entered_at_turn: int

@dataclass
class DialogueState:
    turn_id_counter: int
    turns: list[TurnRecord]
    participants: dict[str, ParticipantState]
    options: dict[str, OptionState]
    stance_table: StanceTable
    discourse: DiscourseGraph
    phase: PhaseEvidence
    consensus_state: Literal["none", "candidate_emerging", "majority_candidate",
                              "conditional_consensus", "full_consensus",
                              "blocked", "failed"]
    candidate_option: Optional[str]
    moderator_style: str
    outcome: Literal["pending", "success", "force_close", "failed"]
```

This data model is the single most important change. Once it exists, every other layer stops re-parsing transcript strings.

### 8. Paper → Implementation Mapping (final merged table)

| Paper / Concept | Idea | Current weakness | Concrete change | Module | Diff. | Effect |
|---|---|---|---|---|---|---|
| Fisher: fav/unfav/ambig units | Decision emerges from changing ratios | Consensus is regex + phrase lists | Track per-(option × speaker) stance + Fisher ratios over rolling window | `reasoning/phase_detector.py`, `state/stance.py` | M | Grounded consensus; fewer premature endings |
| Fisher: emergence | Dissent dissipates via ambiguity → conditional acceptance | "Emergence" only means "everyone voted" | Detect emergence when opposition falls + conditional/ambiguous around one candidate | `reasoning/phase_detector.py` | M | Natural compromise transitions |
| Fisher: reinforcement | Final phase = visible support dominates, blockers gone | Closure can feel forced | Separate "candidate emerging" from "reinforced/confirmed"; require visible support + no unresolved blocker before closure | `reasoning/consensus_engine.py` | M | Better endings |
| Fisher caveat | Phases are gradual, not universal | Hard phase switches | Store phase confidence; allow short/skipped phases | `reasoning/phase_detector.py` | M | Less mechanical |
| SSJ rule 1a (current selects next) | Addressed party has obligation | One-shot pending_question_target evaporates next turn | Persist Q in `DiscourseGraph.pending_questions` until answered | `state/discourse.py`, `policy/turn_policy.py` | S | Q→A rate ↑ |
| SSJ rule 1b (self-select) | First-starter wins; biased by traits | Pure weighted-random | Rule cascade with personality-biased self-selection probability | `policy/turn_policy.py` | M | More natural participation |
| SSJ rule 1c (continue) | Rare current-speaker continuation | Always switches speaker | Allow rare continuation if prior act incomplete | `policy/turn_policy.py` | S | Better local coherence |
| SSJ last-as-next bias | Q→A pair returns to questioner | Recency penalty negates this | Boost prior speaker after Q→A | `policy/turn_policy.py` | S | Natural back-and-forth |
| SSJ adjacency pairs | Q/A, greeting/greeting, invitation/accept-decline | Only Q→A is tracked | First-class adjacency-pair object | `state/discourse.py` | M | Robust pending tracking |
| MUCA 3W (What/When/Who) | Separate decisions | All blurred in `sim_turn` | Dedicated turn_policy / act_planner / addressee selection | `policy/` | L | Testable, controllable |
| MUCA Dialog Analyzer | Periodic state extraction | All extraction every turn via regex | One deterministic pass + LLM classifier every K turns for ambiguous cases | `state/tracker.py` | M | Vote accuracy ↑, redundant work ↓ |
| MUCA Strategy Arbitrator | Priority-ranked strategies | Nested if/elif in orchestrator | `Strategy` objects with `eligible(state)` + `execute(state)` + priority | `policy/moderator_policy.py` | M | Testable, extensible |
| MUCA Strategy cooldowns | Repeated strategies = bot loops | Forbidden openers only | Per-participant act cooldowns | `state/participant.py`, `policy/act_planner.py` | S | Less scripted |
| MUCA Grounding | Pin known facts | Options re-sent but personas invent attributes | KNOWN_FACTS block + post-turn fact-check + one repair pass | `realize/grounding.py`, `prompts.py` | M | Fewer invented facts |
| Ouchi & Tsuboi | Multi-party = (addressee, content) | Implicit addressee | `Turn.addressees` + `reply_to` first-class fields | `state/turn.py` | S | Trivial Q tracking |
| Ouchi & Tsuboi dynamic speaker | Speaker state changes over context | Mostly static persona + raw history | Maintain `ParticipantState` with debt, recent acts, cooldowns | `state/participant.py` | M | Better adaptation |
| McCrae & John FFM | Broad probabilistic traits | Named adversarial phrases in prompts | Replace with bias parameters consumed by act_planner + turn_policy | `policy/personality_bias.py` | M | Distinct voices, less scripted |
| McCrae factors orthogonal | Stubbornness ≠ low-A | Conflated | Hard-blocker = sampled latent at dialogue level, capped | `policy/stubbornness.py` | M | Holdout rate ≤ 10% |
| McCrae factor blends | High-O × high-C = sustained inquiry; high-O × low-C = idle curiosity | Single-trait cues only | Register cues for trait *interactions*, never named phrases | `persona.py` | S | More distinct voices |

---

## Part III — Stage-by-Stage Refactor Plan

Each stage is independently shippable and produces measurable progress. Earlier stages enable measurement; later stages depend on the new state objects.

### Stage 1 — Diagnostics and structured logging

**Goal:** measure first, refactor second. No behavior change.

Add per-turn JSONL trace next to each transcript (`logs/{dialogue_id}.jsonl`) with one record per turn containing:
- `turn_id`, `phase`, `speaker`, `candidate_speakers`, `speaker_scores`, `selected_reason`
- `addressees`, `reply_to`, `dialogue_act_estimated`
- `mentioned_options`, `stance_updates_estimated`, `is_question`, `answers_question_id_estimated`
- `repetition_pressure`, `consensus_tier_used` ("soft" | "regex" | "reduced_opposition" | "llm" | "none")
- `moderator_intervention_type` if any
- `tokens_in`, `tokens_out`, `prompt_type`, `prompt_component_estimates` (persona / options / voice / history / instruction)

Add per-dialogue summary (`logs/{dialogue_id}_summary.json`):
- `outcome`, `force_close_reason`
- `confirmation_rejection_count` (specifically log this — the confirmation rollback loop is a known failure mode)
- `same_candidate_retested_count`
- `reopened_after_confirmation: bool`
- `force_closed_after_confirmation_failure: bool`
- `phase_durations` (turns per phase)
- `hard_blocker_present: bool`
- `holdout_turns_by_speaker`
- `consensus_tier_used`
- `vote_flips_per_speaker`
- `participation_gini`
- `tokens_setup_in/out`, `tokens_dialogue_in/out`, `total_in/out`

Build `tools/analyze_logs.py` to read all logs and emit:
- `force_close_rate`, `success_rate`
- `confirmation_rollback_rate`
- `hard_blocker_dialogue_rate` (estimated from public behavior: persistent opposition to the current candidate across multiple relevant turns, with a stated blocker reason or unmet condition; private beliefs may be used only as diagnostic context, not as the definition)
- `participation_gini` distribution
- `question_answer_rate` (Q answered by addressee within 2 turns)
- `addressee_reply_compliance` rate
- `phase_progression` (does ambiguity actually rise in current "emergence"? — i.e., does Fisher's pattern even appear today?)
- `tokens_per_dialogue` distribution
- `tokens_per_successful_dialogue`
- `repeat_opener_rate`
- `invented_fact_rate` (LLM-judged on a sample)

Run a **batch baseline of 50 fixed-seed scenarios** (use the topics already in `scenarios.txt` and extend if needed). Save the baseline metrics. Every later stage compares against this.

**Files touched:** [src/logger.py](src/logger.py), [src/orchestrator.py](src/orchestrator.py), new `tools/analyze_logs.py`, new `eval/eval_scenarios.txt`.

**Acceptance:** baseline metrics file exists; `analyze_logs.py` runs end-to-end on existing logs.

---

### Stage 2 — Config centralization

**Goal:** every behavior-changing number lives in [config.yaml](config.yaml).

Specific magic numbers to move (non-exhaustive, file:line references):

| Number | Current location | Meaning |
|---|---|---|
| `0.65` | [orchestrator.py:329](src/orchestrator.py#L329) | repetition_pressure threshold for max_speakers=1 |
| `[0.30, 0.50, 0.20]` | [orchestrator.py:333](src/orchestrator.py#L333) | speaker-count weights |
| `0.45` | [simulator.py:409](src/simulator.py#L409) | Jaccard self-repetition threshold |
| `0.55` | [moderation.py:317](src/moderation.py#L317) | outlier word-overlap threshold |
| `4`, `5` | [moderation.py:259-260](src/moderation.py#L259-L260) | speculative-loop thresholds |
| `0.65` | [simulator.py:562](src/simulator.py#L562) | closure template-vs-goodbye probability |
| `0.50, 0.90, 0.80, 0.22, 0.10, 0.08, 0.05, 0.12, 0.35, 0.30` | [turn_manager.py:75-126](src/turn_manager.py#L75-L126) | all turn-selection weights |
| `5.0, 0.5, 2.0, 1.0, -2.0, -3.0, -4.0` | [orchestrator.py:498-547, 613-680](src/orchestrator.py#L613-L680) | force-close and compromise scoring |
| `3` | [simulator.py:347](src/simulator.py#L347) | turns_since_rejection escalation threshold |
| `0.55, 0.75, 0.80` | various | repetition pressure thresholds |
| `14, 24, 36, 48, 60` | [persona.py:94-100](src/persona.py#L94-L100) | response_length word budgets |
| `8, 10, 18, 40, 42` | [persona.py:166-178](src/persona.py#L166-L178) | phase-specific word caps |
| `120` | [llm_client.py:122](src/llm_client.py#L122) | HTTP timeout |
| name pool `_DEFAULT_NAMES` | [main.py:235-238](main.py#L235-L238) | participant name pool |
| `cfg.repetition.forbidden_frames` | already in config but should be retired (see Stage 3) | |

Proposed reorganized [config.yaml](config.yaml) structure (sections, not exhaustive keys):

```yaml
llm:
  provider, models, endpoints, sampling, gemini_rpm_delay,
  timeouts: { request_seconds: 120 }
  retries:  { max_json_repairs: 2 }

simulation:
  num_participants, num_participants_random, num_participants_min/max
  moderator_style
  name_pool: [Alex, Jordan, Morgan, Taylor, Casey, Riley, Drew, Quinn, Avery, Blake]

turns:
  hard_ceiling: 40
  min_before_narrowing: 9
  escalation_rounds: { level_1: 2, level_2: 5, level_3: 9 }
  max_speakers_weights: [0.30, 0.50, 0.20]

personality:
  trait_min, trait_max
  trait_ranges: { ... }
  enforce_diversity: true
  diversity_thresholds: { agreeableness: 4, extraversion_min: 4 }

stubbornness:                              # NEW (see Stage 8)
  hard_blocker_dialogue_probability: 0.05  # target ≤ 10%
  max_hard_blockers_per_dialogue: 1
  require_public_reason: true
  allow_softening_if_condition_met: true

response_length:
  word_budgets:  { 1: 14, 2: 24, 3: 36, 4: 48, 5: 60 }
  phase_caps:
    greeting: 8
    confirmation: 10
    narrowing: { min: 18, max: 40 }
    emergence: 42

turn_policy:                               # NEW (Stage 6)
  rule_priority: [addressed_question, addressed_mention, self_select, current_continues]
  obligation_weights:
    addressee_of_question: 0.90
    addressee_of_statement: 0.80
  self_selection_weights:
    extraversion: 0.30
    primary_boost: 0.10
    unspoken_boost: 0.35
    participation_debt: 0.20
    novelty: 0.10
    phase_relevance: 0.15
  penalties:
    last_speaker: 0.50
    recent_speaker_per_turn: 0.12
    own_repetition: 0.30
    introvert_off_turn: 0.08
  windows: { recent_speakers: 4, recent_turns: 8 }

phase_policy:                              # NEW (Stage 7)
  window_size: 8
  min_turns_before_conflict: 6
  emergence:
    opposition_decline_threshold: 0.20
    ambiguity_rise_threshold: 0.30
  reinforcement:
    support_rate_threshold: 0.60
  use_confidence: true                     # Fisher caveat — store confidence, not hard switch

consensus:
  llm_check_every_n_turns: 5
  regex_window: 8
  stall_rounds_to_force: { active: 2, minimal: 3, passive: 999 }
  max_dissenters_active: 1
  max_dissenters_other: 0
  stance_weights:
    support: 1.0
    conditional_support: 0.7
    ambiguous: 0.3
    oppose: -1.0
    blocker: -2.0
  closure_requires_no_blocker: true

moderation:
  narrowing:
    min_turns_per_participant: 2
  interventions:
    fresh_question_grace: true
    outlier_overlap_threshold: 0.55
    speculative_loop_threshold: 4
  closure:
    template_probability: 0.65

repetition:
  pressure_window: 8
  min_word_length: 3
  jaccard_threshold_self: 0.45

prompt_budget:                             # NEW (Stage 4)
  recent_turns_short: 4
  recent_turns_long: 8
  max_turn_prompt_tokens: 900
  summarize_after_turn: 12

grounding:                                 # NEW (Stage 10)
  enable_fact_check: true
  repair_attempts: 1

logging:                                   # NEW
  write_jsonl_trace: true
  write_summary_json: true
  write_csv: true
  write_token_log: true
  log_prompt_components: true

evaluation:                                # NEW (Stage 12)
  eval_scenarios_path: eval/eval_scenarios.txt
  seed: 42
```

The `repetition.forbidden_frames` list should be retired (see Stage 3). The current `cfg.consensus.*` keys mostly stay but get new siblings.

**Files touched:** [config.yaml](config.yaml), every module that currently has a magic number.

**Acceptance:** a manual audit, supported by grep or a small AST-based check, finds no behavior-changing numeric constants outside `config.yaml`. Simple grep is only an audit aid; it is not sufficient proof by itself. Remaining numeric literals should be structural constants, test data, enum-like labels, or obvious loop/index mechanics.

---

### Stage 3 — Prompt centralization (hard requirement)

**Goal:** every LLM-facing string lives in [prompts.py](src/prompts.py).

Currently violating the rule (move these out of [simulator.py](src/simulator.py)):
- `narrowing_base` and the three `narrowing_instruction` variants ([simulator.py:91-109](src/simulator.py#L91-L109))
- All `phase_instructions` strings ([simulator.py:111-141](src/simulator.py#L111-L141))
- The repetition-pressure appendix ([simulator.py:144-146](src/simulator.py#L144-L146))
- `_interaction_instruction` body ([simulator.py:306-389](src/simulator.py#L306-L389))
- `_position_discipline` text ([simulator.py:206-294](src/simulator.py#L206-L294))
- `_skepticism_nudge` text ([simulator.py:537-548](src/simulator.py#L537-L548))
- `forced_block` text ([prompts.py:192-196](src/prompts.py#L192-L196)) — already in prompts.py but driven by a boolean; move the logic too

New prompts.py structure:

```python
# prompts.py

# Stable per-dialogue/per-persona blocks (kept compact, sent once via speaker card)
def speaker_card_block(persona, beliefs) -> str: ...
def options_block(options) -> str: ...
def known_facts_block(options) -> str: ...
def voice_register_block(style_rule, max_words) -> str: ...   # NO named filler phrases

# Per-turn blocks (composed by simulator from structured inputs)
def sim_turn(
    speaker_card: str,         # pre-built
    options_relevant: str,     # only the options that matter for this turn
    public_state_summary: str, # one paragraph from StanceTable
    local_context: str,        # last 4-6 turns
    act_plan: TurnPlan,        # see Stage 7
    output_contract: str,      # one short line about word budget + formatting
) -> str: ...

def moderator_intervention(intervention_plan, public_state, context) -> str: ...
def moderator_emergence(intervention_plan, public_state) -> str: ...
def moderator_compromise_test(intervention_plan, public_state) -> str: ...
def moderator_force_close(intervention_plan, public_state) -> str: ...

def consensus_check(stance_summary, recent_turns) -> str: ...
def stance_classifier(turn_text, options) -> str: ...   # NEW (Stage 5)

def option_generation(topic) -> str: ...                # existing
def persona_group_generation(topic, names, traits) -> str: ...   # NEW (Stage 11)
def agent_beliefs_group(topic, personas, options) -> str: ...    # NEW (Stage 11)
```

Other modules pass structured values (TurnPlan, public_state_summary, etc.). They do not assemble prose. Inside prompts.py, *all* prose lives.

Drop the named forbidden-frames list from the prompt entirely. Replace with positive register guidance: *"Vary how you open each turn. Do not reuse the same opener across turns."* The `forbidden_openers` list is still useful internally to detect repetition, but **do not name examples to the model.**

**Files touched:** [src/prompts.py](src/prompts.py), [src/simulator.py](src/simulator.py), [src/moderation.py](src/moderation.py).

**Acceptance:** a manual audit, supported by grep or an AST-based string scan, finds no LLM-instruction-like prose outside `prompts.py`. Grep is only a helper because it can miss concatenated strings, normal quoted strings, and f-strings.

---

### Stage 4 — Prompt and token compression (speaker card pattern)

**Goal:** drop per-turn input cost from ~1500 to ~500–900 tokens.

Replace the current giant per-turn template with the **speaker-card pattern**:

```
SPEAKER CARD
- Riley, primary traveler
- Reserved, practical, short turns
- Prefers Option B; would accept C if cost is handled

KNOWN FACTS (options are the only shared facts you may cite)
- Option B: Convenience — easier or faster to execute; trade-off: moderate cost
- Option C: Quality — strongest expected outcome; trade-off: higher effort

CURRENT GROUP STATE
- Candidate: B (Casey supports B, Blake conditionally accepts B if timing is clear)
- Riley: unresolved concern about baggage fee

YOUR PLANNED MOVE
Act: CONDITIONAL_ACCEPT  Target: Option B  Condition: only if baggage fee is included
Address: Blake (replying to turn 17)

RECENT TURNS
[15] Blake: "I'd be in for B if we can sort out the timing question."
[16] Casey: "Timing is fine; I checked."
[17] Blake to Riley: "What about cost — does B work for you?"

OUTPUT
One chat message from Riley, ≤ 24 words. No name prefix. Reply to Blake.
```

This replaces:
- 80 lines of voice/style rules
- Full persona block
- Full options block (only relevant options shown)
- Full forbidden_openers + forbidden_frames listing
- The "do not repeat / do not invent / do not stack questions" warnings (these become deterministic post-checks in Stage 10)

Stable per-dialogue context (the speaker card content beyond the planned move) **can** be sent via a system message if the provider supports it (groq/openai-style). For `uni` (Ollama) and `gemini` without prompt caching, this is still useful because the prompt is shorter; even without API caching, going from ~1500 to ~700 tokens per turn is the win, not server-side cache.

Local context: last 4 turns by default, rising to 6 only when relevant. Replace anything older than turn 12 with a 2-sentence rolling deterministic summary maintained by the orchestrator (MUCA-style accumulative summary, but kept deterministic — see Stage 5).

**Realistic target:** input tokens per typical 25-turn 3-participant dialogue drops from ~50k to **~15–22k**.

**Files touched:** [src/prompts.py](src/prompts.py), [src/realize/simulator.py](src/simulator.py), new `src/realize/prompt_context.py` (rolling summary builder).

**Acceptance:** baseline token-cost batch metric drops by ≥ 50%.

---

### Stage 5 — Dialogue state in parallel (no behavior change yet)

**Goal:** populate the new structured state alongside existing logic. The orchestrator continues using the old code paths; the new state is logged for comparison.

Add `src/state/` package with the dataclasses from §7. Add `state/tracker.py::update(state, raw_turn)` that:

1. Assigns a `turn_id`.
2. Detects `is_moderator` from speaker.
3. Extracts `addressees` via name-match in first 1–3 words (Ouchi & Tsuboi convention).
4. Resolves `reply_to` via `DiscourseGraph.pending_questions` lookup or fallback to last participant turn.
5. Detects `is_question` deterministically (presence of `?`).
6. Detects `mentioned_options` via [extract_option_letters](src/utils.py#L72-L73).
7. Estimates `dialogue_act` via:
   - deterministic rules first (greeting words → GREET; "I prefer Option X" → COMMIT_VOTE; "yes/no" in confirmation phase → CONFIRM/REJECT_WITH_REASON; pure `?` → ASK_*; etc.)
   - LLM `stance_classifier` call **only for the ambiguous remainder**, batched every K=3 turns over a window of N recent ambiguous turns
8. Extracts `stance_updates` per mentioned option using the same hybrid approach.
9. Updates `StanceTable`, `OptionState`, `ParticipantState`, `DiscourseGraph`.
10. Updates `participation_debt = -1 for the speaker, +1/(n-1) for everyone else` (or a configured formula).
11. Writes the full `TurnRecord` to JSONL trace.

Crucially: log both the **old** decisions (from current consensus.py / orchestrator.py logic) and the **new** decisions (from StanceTable) side by side. Don't switch yet — compare A/B over a baseline batch.

Add deterministic rolling summary: every K=6 turns, build a 2-sentence summary by template from current state (who supports what, current candidate, unresolved blockers). No LLM call.

**Files touched:** new `src/state/` package, [src/orchestrator.py](src/orchestrator.py) to wire the tracker in, [src/logger.py](src/logger.py) to write `TurnRecord` JSONL.

**Acceptance:** for every existing baseline dialogue, the new state's vote/stance recovery matches or exceeds the old regex-based recovery, measured by `state_recovery_agreement_rate` in `analyze_logs.py`.

---

### Stage 6 — Turn-taking redesign (SSJ rule cascade)

**Goal:** replace weighted-sum scoring with the SSJ priority order.

New `policy/turn_policy.py::select_next_speakers(state) -> list[Participant]`:

```
Rule cascade (in order, stop at first that fires):

1a. If DiscourseGraph has a pending_question with explicit addressees,
    those addressees have FIRST priority (in mention order, capped at
    max_speakers).

1a'. If DiscourseGraph has a recent direct-mention without ?, addressed
     speaker has SECOND priority (slightly weaker than question).

1b. If no obligated addressee, run self-selection:
    - Score = self_selection_weights · (extraversion, primary_boost,
              unspoken_boost, participation_debt, phase_relevance,
              novelty, stance_relevance)
    - Apply penalties: last_speaker, recent_speaker_per_turn,
                       own_repetition_penalty, introvert_off_turn
    - Sample from softmax over scores
    - "Last-as-next" bias: if last turn was a Q→A pair, the questioner
      gets a +bias to be chosen next (SSJ §4.5)

1c. If no useful speaker (all scores near zero), allow current speaker
    to continue OR insert moderator turn.

Greeting/opening phases: round-robin (uncovered participants first).
Confirmation phase: each participant gets one shot to confirm/reject.
Closure phase: primary first, then others.
```

Keep the old [turn_manager.py](src/turn_manager.py) behind a `cfg.turn_policy.use_legacy: true` flag during A/B comparison.

**Files touched:** new `src/policy/turn_policy.py`, [src/orchestrator.py](src/orchestrator.py) (swap selector behind config flag).

**Acceptance:** `question_answer_rate` (Q answered by addressee within 2 turns) rises to ≥ 85% in the new path, vs. baseline.

---

### Stage 7 — Act Planner (the "What")

**Goal:** the simulator stops deciding what to do; it only realizes a planned act.

New `policy/act_planner.py::plan_turn(speaker, state) -> TurnPlan`:

```python
@dataclass
class TurnPlan:
    speaker: str
    act: DialogueAct
    target_option: Optional[str]
    addressee: Optional[str]
    reply_to: Optional[int]
    condition: Optional[str]      # for CONDITIONAL_ACCEPT / blocker
    max_words: int
    rationale: str                # for logging — why this act?
```

The planner picks the act using:
- **Obligations** (from DiscourseGraph): if the speaker owes an answer → ANSWER.
- **Phase**: in narrowing without a vote → COMMIT_VOTE; in emergence with a candidate ∈ acceptable + condition exists → CONDITIONAL_ACCEPT; in confirmation → CONFIRM or REJECT_WITH_REASON.
- **Persona bias** (from Stage 8): high-O is more likely to PROPOSE_COMPROMISE; high-A more likely to CONCEDE; high-N more likely to ASK_CLARIFICATION with risk framing.
- **Cooldowns**: if speaker just used ASK_CLARIFICATION twice in their last 4 turns → de-prioritize ASK_*.
- **Stance**: if the candidate is in `beliefs.acceptable` and a condition is on the table that matches the persona's `beliefs.concession` → CONDITIONAL_ACCEPT becomes very likely.

The act planner is **probabilistic but constrained**. It is not a rigid script. A high-O speaker with no obligations and the candidate already acceptable still has, e.g., 15% chance of CHALLENGE — but the dominant act is shaped by state.

The planner outputs a single `TurnPlan` passed into `prompts.sim_turn(act_plan=...)`. The LLM realizes that one move in persona voice. No more "be natural / vary your move / answer if asked / don't repeat / ..." pile.

**Files touched:** new `src/policy/act_planner.py`, [src/realize/simulator.py](src/simulator.py), [src/prompts.py](src/prompts.py).

**Acceptance:** measurable distinctness of personas in batch (e.g., share of CONCEDE acts for high-A is significantly higher than for low-A); repetition rate drops.

---

### Stage 8 — Big Five redesign + rare hard-blocker

**Goal:** remove caricatured phrasings; introduce probabilistic biases; cap stubbornness explicitly.

**Part A — strip caricatures.**

Rewrite [persona.personality_summary()](src/persona.py#L179-L208) to produce **register descriptors only**:

| Trait level | Register cue (good) | Caricature (BAD — remove) |
|---|---|---|
| High openness | "considers angles others haven't raised; comfortable reframing the question" | "'wait, what about...' or flip the framing" ❌ |
| Low openness | "prefers concrete options on the table; impatient with speculation" | "'let's stick to what's actually on the table'" ❌ |
| High extraversion | "energetic, quick to react, thinks aloud" | "'oh that's actually a good point'" ❌ |
| High agreeableness | "acknowledges before pushing back; seeks common ground" | "'I get that, but...'" ❌ |
| Low agreeableness | "direct, skeptical, blunt" | "'yeah but that's not actually true'" ❌ |
| High neuroticism | "worry shows through; sensitive to unresolved uncertainty" | "'but what if that doesn't work out?'" ❌ |

The register cues describe *what kind of person* the persona is. The caricatures **prescribe phrases the LLM mimics literally**. Same for [_skepticism_nudge](src/simulator.py#L537-L548): drop it. Skepticism becomes an `act_planner` bias.

**Part B — `policy/personality_bias.py`.**

```python
@dataclass
class PersonalityBias:
    talkativeness: float           # E
    self_selection_propensity: float
    concession_propensity: float   # A
    objection_propensity: float    # low A
    risk_salience: float           # N
    detail_orientation: float      # C
    consistency: float             # C
    reframing_propensity: float    # O
    clarification_propensity: float

def derive(persona: Persona) -> PersonalityBias:
    """Map Big Five 1..5 values to bias floats. Probabilistic, not deterministic."""
```

These biases enter `turn_policy` (self-selection score) and `act_planner` (act sampling weights). They never enter the prompt as named phrases.

**Part C — rare hard-blocker sampling.**

```python
# policy/stubbornness.py
def sample_hard_blockers(participants, cfg) -> list[str]:
    """At dialogue start, decide whether this dialogue has a hard blocker.
    p(dialogue has blocker) = cfg.stubbornness.hard_blocker_dialogue_probability
    If yes, pick at most cfg.stubbornness.max_hard_blockers_per_dialogue participants.
    The chosen participant gets is_true_hard_blocker=True and a public reason
    (drawn from their beliefs.key_concern or backstory)."""
```

A true hard blocker:
- Has `is_true_hard_blocker=True` on their `ParticipantState`.
- Their `beliefs.rejected` contains the leading candidate.
- Their `act_planner` is biased toward REJECT_WITH_REASON when the candidate is their rejected option.
- They CAN still soften if `cfg.stubbornness.allow_softening_if_condition_met` AND their `beliefs.concession` condition is publicly addressed.
- They are **the only mechanism by which the dialogue legitimately fails or force-closes due to stubbornness**.

A non-blocker low-A persona is just blunt and skeptical. They do not automatically reject; they argue, then concede when their concern is met.

**No "after N turns, force concession" hard cap.** That would produce fake consensus, which contradicts the project goal of plausible failure modes.

**Files touched:** [src/persona.py](src/persona.py), new `src/policy/personality_bias.py`, new `src/policy/stubbornness.py`, [config.yaml](config.yaml).

**Acceptance:** `hard_blocker_dialogue_rate` roughly tracks `cfg.stubbornness.hard_blocker_dialogue_probability`. For reliable estimates, use a larger batch of 200–500 dialogues or report confidence intervals. On the 50-topic eval batch, treat this as a coarse sanity check rather than a strict ±3 percentage-point criterion.

---

### Stage 9 — Consensus and phase redesign (Fisher-aligned)

**Goal:** consensus and phase derive from `StanceTable`, not from regex over transcript.

**Phase detector** (`reasoning/phase_detector.py`):
- Computes Fisher ratios over `StanceTable.fisher_ratios(window=cfg.phase_policy.window_size)`.
- Transitions (with confidence):
  - `orientation → conflict`: when `disfavor_rate` first exceeds `ambiguous_rate` AND we've had `min_turns_before_conflict` participant turns.
  - `conflict → emergence`: when `disfavor_rate` falls and `ambiguous_rate + conditional_rate` rises around one candidate (deltas configured in `cfg.phase_policy.emergence`).
  - `emergence → reinforcement`: when `favor_rate` dominates AND `unresolved_blockers(candidate)` is empty.
  - `reinforcement → closure`: when confirmation succeeds.
- Store **confidence**, not just a label. The orchestrator can show "emergence (conf 0.7)". Allow phases to be short or skipped if evidence supports it. Fisher caveat respected.

**Consensus engine** (`reasoning/consensus_engine.py`):
- Replaces [src/consensus.py](src/consensus.py).
- Uses `StanceTable.majority_favor(min_supporters)` + `StanceTable.unresolved_blockers(candidate)`.
- Consensus states (replacing the boolean `agreement_reached`):
  - `none`: no candidate has plurality.
  - `candidate_emerging`: one option has plurality but ≥ 1 ambiguous or conditional supporter still pending.
  - `majority_candidate`: plurality + minority is conditional/ambiguous (not active opposition).
  - `conditional_consensus`: all participants support or conditionally support, conditions on the table.
  - `full_consensus`: all participants support; no unresolved conditions.
  - `blocked`: a true hard blocker actively opposes the leading candidate.
  - `failed`: hard ceiling reached without progress.
- Closure requires:
  - `consensus_state ∈ {conditional_consensus, full_consensus}`, AND
  - No `unresolved_blockers(candidate)` (unless `cfg.consensus.closure_requires_no_blocker: false`), AND
  - Confirmation does not introduce new opposition.

**Confirmation rejection becomes a stance update**, not a separate scan:
- A "no" during confirmation = `StanceUpdate(speaker, candidate, "blocker", condition=text_reason)`.
- The new consensus_state is recomputed; if it drops below `conditional_consensus`, the orchestrator routes to a `PROBE_BLOCKER` moderator intervention rather than re-entering the same confirmation loop.
- This eliminates the **confirmation rollback loop** as a runtime failure mode.

**Force-close redesign — rename to "best-available decision":**
- Triggered only when phase confidence in `reinforcement` cannot be reached within `cfg.turns.hard_ceiling`.
- Scoring uses **public stance** only:
  - `score(opt) = Σ_speaker stance_weights[stance_of(speaker, opt)]`
  - `cfg.consensus.stance_weights` provides the mapping.
- Private beliefs **do not enter** the force-close score. They only enter the act planner (biasing the speaker's own behavior).
- Restricted to options that received support/conditional_support from at least one speaker (fallback to participant-mentioned options if none, then to all — but always log which tier was used).
- Logged outcome is `best_available_decision`, not `force_close`. The label "force close" implies arbitrary choice; the new label is honest about what happened.

**Files touched:** new `src/reasoning/phase_detector.py`, new `src/reasoning/consensus_engine.py`, [src/orchestrator.py](src/orchestrator.py), [src/logger.py](src/logger.py).

**Acceptance:** `confirmation_rollback_rate` drops to ≤ 5%; phase progression follows Fisher's pattern (ambiguity actually rises in emergence) in ≥ 70% of dialogues.

---

### Stage 10 — Grounding + fact check

**Goal:** stop personas inventing attributes (page counts, budgets, ages).

Add KNOWN_FACTS block to every sim_turn prompt (already in Stage 4 speaker card). Add a post-turn deterministic fact-check:

```python
# realize/grounding.py
def fact_check(turn_text: str, options: dict[str, OptionState], topic: str) -> list[str]:
    """Return list of suspicious factual claims:
       - numbers not in any option text
       - named attributes not in any option text
       - locations / proper nouns not in option text or topic
    """
```

If any are found:
- Log `invented_fact_warning` in the turn record.
- If `cfg.grounding.repair_attempts >= 1`, regenerate the turn ONCE with an added directive in the prompt: *"Use only the listed option attributes. Do not invent numbers or named details."*
- If repair still fails, keep the turn but log the warning.

LLM-based fact-checking is **optional and disabled by default** due to cost. The deterministic check catches most cases.

**Files touched:** new `src/realize/grounding.py`, [src/realize/simulator.py](src/simulator.py), [src/prompts.py](src/prompts.py).

**Acceptance:** `invented_fact_rate` in metrics drops by ≥ 50%.

---

### Stage 11 — Setup-call compression

**Goal:** reduce setup LLM calls from `2N + 2` to ~3 for typical N=3.

Current setup ([persona.py:13-14](src/persona.py#L13-L14)): `1 options + 1 roles + N persona concept + N beliefs = 2N + 2`. For N=3, that's 8 calls.

Replace with:
1. `option_generation(topic)` — 1 call (existing).
2. `persona_group_generation(topic, names, sampled_traits)` — 1 call returning all N personas' roles + backstories + goals in one structured JSON.
3. `agent_beliefs_group(topic, personas, options)` — 1 call returning all N belief states in one JSON.

Trade-off: a single longer call vs N short calls. Net token savings come from not re-sending the topic + options + format spec N times. Net wall-clock savings come from fewer round-trips.

Validation:
- The grouped beliefs prompt must still enforce per-persona consistency rules (preferred ↔ key_concern, acceptable set size 2–3, etc.).
- If JSON validation fails, fall back to per-persona calls.

**Files touched:** [src/persona.py](src/persona.py), [src/prompts.py](src/prompts.py).

**Acceptance:** setup tokens drop from ~3k to ~1.5–2k per dialogue.

---

### Stage 12 — Evaluation harness

**Goal:** every refactor stage produces measurable progress against a fixed reference.

Build:
- `eval/eval_scenarios.txt` — 50 fixed topics covering decision domains (travel, food, group activity, technical choice, social, scheduling, creative).
- `tools/run_eval.py` — runs all 50 scenarios at fixed seed (`cfg.evaluation.seed`) and emits a metrics report.

**Quantitative metrics** (computed by `tools/analyze_logs.py`):
- `force_close_rate` / `best_available_decision_rate`
- `natural_consensus_rate`
- `confirmation_rollback_rate`
- `hard_blocker_dialogue_rate` (target: roughly tracks config; use larger batches or confidence intervals for precise estimates)
- `participation_gini`
- `question_answer_rate`
- `addressee_response_compliance`
- `phase_progression_fidelity` (does ambiguity rise in emergence?)
- `repeat_opener_rate`
- `invented_fact_rate`
- `vote_flip_distribution`
- `tokens_per_dialogue`, `tokens_per_successful_dialogue`
- `llm_calls_per_dialogue`

**Qualitative rubric (1–5)**, applied by sampling 10 dialogues per batch:
- Naturalness
- Local coherence (does turn N respond to turn N-1?)
- Persona distinctiveness (can you tell who is speaking without the name?)
- Disagreement plausibility (is the disagreement grounded?)
- Compromise plausibility (are the proposed compromises sensible?)
- Ending quality (does the outcome make sense given what happened?)
- Topic grounding (no invented facts?)

**Comparison protocol:**
1. Save baseline metrics BEFORE any refactor (immediately after Stage 1 diagnostics).
2. After each subsequent stage, re-run the eval set and compare deltas.
3. Maintain a small regression set: 5 "known-good" dialogues + 5 "known-bad" dialogues. After each stage, verify the known-good ones are still good and the known-bad ones improve.

**Files touched:** new `tools/run_eval.py`, new `eval/eval_scenarios.txt`, extend `tools/analyze_logs.py`.

**Acceptance:** metrics report runs end-to-end and produces deltas vs baseline.

---

## Part IV — Cross-Cutting Decisions

### Config-as-source-of-truth (the hard requirement)

Every behavior-changing number lives in [config.yaml](config.yaml). No exceptions for "well, it's only used in one place." If changing the value changes dialogue behavior, it belongs in config.

The reorganized config (§Stage 2) groups by domain (`llm`, `simulation`, `turns`, `personality`, `stubbornness`, `turn_policy`, `phase_policy`, `consensus`, `moderation`, `repetition`, `prompt_budget`, `grounding`, `logging`, `evaluation`). Comment every key with its meaning and range.

### Prompts-only-in-prompts.py (the second hard requirement)

After Stage 3, no module other than `prompts.py` constructs LLM-facing text. Other modules pass structured values; `prompts.py` builds prose. This can be checked with a lightweight lint script or AST-based string scan. Grep is useful for auditing, but it should not be treated as complete enforcement because it can miss ordinary quoted strings, concatenation, and f-strings.

### Determinism vs LLM

LLM is for:
- option generation
- persona text (backstory, goal)
- belief generation
- natural utterance realization
- nuanced moderator wording
- stance classification for genuinely ambiguous turns
- optional final evaluator

Deterministic logic is for:
- speaker selection
- addressee + reply-to extraction
- option-letter extraction
- dialogue act classification for clear cases
- phase detection from Fisher ratios
- consensus state from StanceTable
- best-available-decision scoring
- fact-check against option text
- repetition detection

When in doubt, prefer deterministic. LLM calls are expensive and add variance to debugging.

### A/B safety pattern

For each stage that changes decision logic (Stages 6, 7, 9):
1. Add the new module.
2. Run it **alongside** the existing logic.
3. Log both decisions.
4. Switch the orchestrator to the new logic **only after** the eval batch confirms the new path is at least as good as the old.
5. Keep the old code path behind `cfg.*.use_legacy: true` for one release cycle in case of regression.

---

## Part V — Things NOT to Do (resolved disagreements)

1. **Do NOT add a hard cap "after N holdout turns, force concession."** This produces fake consensus. The correct mechanism is the rare hard-blocker sampler (Stage 8) plus condition-based softening.

2. **Do NOT rely on API-side prompt caching as the main token saving.** It's not portable across `uni`/`groq`/`gemini`. The portable wins are: prompt compression (Stage 4), setup-call merging (Stage 11), and rolling deterministic summary (Stage 5).

3. **Do NOT remove fallback options entirely.** Add `cfg.simulation.fallback_options_mode: "fail_fast" | "log_and_use"`. Silent fallback hides bugs; loud fallback is acceptable in batch.

4. **Do NOT target a fixed `force_close_rate`.** That metric is a *diagnostic*, not a goal. The goal is "hard_blocker_dialogue_rate ≤ 10%" + "best_available_decision is explainable from public stance evidence."

5. **Do NOT use Fisher's phase transitions as hard switches.** Store confidence; allow short/skipped phases. Fisher explicitly says phases are gradual.

6. **Do NOT name specific filler phrases in prompts**, even as forbidden examples. [CLAUDE.md](CLAUDE.md) is explicit. Any named phrase becomes a future crutch.

7. **Do NOT mix private beliefs and public votes in scoring.** `_force_conclusion`'s 5x vote multiplier is a symptom of this mixing. Public stance is the only basis for outcome justification.

8. **Do NOT delete the old `turn_manager.py` / `consensus.py` / `moderation.py` immediately.** Keep them behind a legacy flag for one release after the new path is verified.

9. **Do NOT skip Stage 1.** Diagnostics before behavior changes. Without baseline metrics, every later change is unmeasured guesswork.

10. **Do NOT mix multiple stages in a single commit.** Each stage is independently shippable and independently reversible.

---

## Part VI — Concrete Next Tasks (prioritized checklist)

This is the ordered task list. Do them in order; each unlocks the next.

### Immediate (Stage 1 — diagnostics)
1. Add JSONL trace logging in [src/logger.py](src/logger.py) with all per-turn fields listed in Stage 1. **No behavior change.**
2. Add per-dialogue summary JSON output.
3. Build `tools/analyze_logs.py` computing all quantitative metrics.
4. Run baseline on existing 6 dialogues in [logs/](logs/). Record baseline metrics in `eval/baseline_metrics.json`.
5. Specifically log `confirmation_rejection_count` and `force_closed_after_confirmation_failure`.

### Token quick wins (Stage 3 + Stage 4 partial)
6. Move all LLM-facing strings from [src/simulator.py](src/simulator.py) into [src/prompts.py](src/prompts.py). (Stage 3 hard requirement.)
7. Drop the named `forbidden_frames` and named filler examples from prompt text. Replace with positive register guidance.
8. Strip caricatured personality cues from [persona.personality_summary()](src/persona.py#L179-L208). Replace with register descriptors only.
9. Re-run baseline. Compare token cost.

### Config cleanup (Stage 2)
10. Move every magic number listed in Stage 2 into [config.yaml](config.yaml) under the new section structure.
11. Verify config is loaded correctly (run a smoke test dialogue).

### State scaffolding (Stage 5)
12. Create `src/state/` package with `TurnRecord`, `StanceUpdate`, `OptionState`, `ParticipantState`, `DiscourseGraph`, `DialogueState`, `Phase`.
13. Create `state/tracker.py::update(state, raw_turn)` running deterministic extraction.
14. Wire the tracker into [src/orchestrator.py](src/orchestrator.py) — populate state in parallel with existing `history: list[str]`. **No decisions use the new state yet.**
15. Log the structured `TurnRecord` to JSONL.

### Speaker card and grouping (Stage 4 full + Stage 11)
16. Refactor [prompts.sim_turn](src/prompts.py#L158-L229) to consume a speaker card + act plan + relevant state + local context.
17. Implement rolling deterministic summary in `state/tracker.py`.
18. Merge persona setup into a single `persona_group_generation` LLM call.
19. Merge belief setup into a single `agent_beliefs_group` LLM call.
20. Re-run baseline. Token target: 50% reduction vs original baseline.

### Behavioral redesign (Stages 6, 7, 8) — parallel-first
21. Implement `policy/turn_policy.py` (SSJ cascade). Run alongside existing TurnManager; log both choices.
22. Implement `policy/personality_bias.py` (Big Five → bias floats).
23. Implement `policy/act_planner.py`. Run alongside; log planned acts. Don't gate generation on it yet.
24. Implement `policy/stubbornness.py` (rare hard-blocker sampler).
25. **Switch over.** Set `cfg.turn_policy.use_legacy: false` and `cfg.act_planner.enabled: true`. Re-run eval. Compare against baseline.

### Consensus and phase redesign (Stage 9)
26. Implement `reasoning/phase_detector.py` (Fisher ratios → phase + confidence).
27. Implement `reasoning/consensus_engine.py` with the new consensus states.
28. Replace confirmation rejection scan with a `StanceUpdate(stance="blocker")` flow.
29. Rename `force_close` → `best_available_decision`; use public-stance-only scoring.
30. Re-run eval. Target: `confirmation_rollback_rate ≤ 5%`.

### Grounding (Stage 10)
31. Implement `realize/grounding.py` deterministic fact-check.
32. Add one-shot repair pass on flagged turns.
33. Re-run eval. Target: `invented_fact_rate` drops ≥ 50%.

### Final evaluation (Stage 12)
34. Build `eval/eval_scenarios.txt` (50 topics).
35. Build `tools/run_eval.py`.
36. Run full eval batch. Produce final metrics report comparing to original baseline.
37. Sample 10 dialogues, apply qualitative rubric. Document remaining issues.

### Optional follow-ups
38. Compare token cost across providers (uni vs groq vs gemini).
39. If a real-dialogue dataset is available, run distributional comparison.
40. Document the final architecture for the written report.

---

## Part VII — Summary

The current system has the right concepts but the wrong central data model. Personas, phases, options, beliefs, and the moderator escalation ladder are correct in spirit. The core fix is to make **dialogue state a first-class structured object**, derive every decision from that state, push the LLM into a pure realization role with compact prompts, and replace named-phrase personality cues with probabilistic biases. Stubbornness is reframed as a rare sampled latent variable, not an emergent side-effect of compounded heuristics. Force-close is reframed as `best_available_decision` based on public stance evidence only.

The five papers map cleanly onto the architecture: Fisher gives the phase + consensus evidence model; SSJ gives the turn-allocation rule cascade; MUCA gives the What/When/Who separation and dialog-analyzer pattern; Ouchi & Tsuboi justifies addressee/reply-to as first-class state; McCrae & John grounds the Big Five framework while leaving the bias mapping as your engineering design.

The 12-stage plan is conservative: each stage is independently shippable, each preserves the existing behavior path during the transition, and each is gated on measured improvement vs. baseline. Stage 1 (diagnostics) is non-negotiable and comes first. After that, the order of value-per-effort is: Stage 3 (prompt centralization) + Stage 4 (token compression) for immediate wins, then Stage 2 (config) for hygiene, then Stage 5 (state scaffolding) as the load-bearing refactor everything else depends on, then Stages 6–9 to implement the papers, then Stages 10–12 to polish and validate.

The target outcome: a scientifically defensible multi-party decision simulator with hard-blocker rate ≤ 10%, force-close rate explainable from public stance evidence, ~15–22k input tokens per typical dialogue (down from ~50k), Fisher-aligned phase progression visible in the data, and a clean separation of state / policy / realization that supports both the written report and future research extensions.
