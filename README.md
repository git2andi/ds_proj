# Autonomous Multi-User Decision Simulator

University project: **Implementation of User Simulators in Multi-User Conversational AI**.

The program simulates several configurable users discussing a fixed public option board. Each simulator independently decides whether to speak, what communicative action to perform, which option or participant to address, and whether to change stance. A lightweight environment coordinates the floor and guides the group through:

```text
OPENING → DISCUSSION → NARROWING → VOTING → CLOSED
```

A single bounded return from voting to narrowing is allowed only when the first vote has no majority and the final discussion produces visible acceptance or switching. Otherwise the run closes unresolved without a pointless duplicate vote.

## Core ownership

### User simulator

Each `UserSimulator` owns:

- voluntary participation;
- the complete structured `UserAction`;
- support, concern, concern-based question, answer, soft comparison, final-position, and compromise choices;
- option focus and addressee;
- personal reason;
- visible stance movement;
- clear visible vote.

### Environment

The environment owns:

- mandatory opening order;
- direct-answer obligations;
- categorical floor arbitration;
- broad turn budgets;
- one active issue;
- soft option coverage;
- public narrowing;
- vote collection and outcome computation.

### LLM

The LLM only realizes one selected action as natural language. It does not choose the speaker, action, stance change, or vote.

Ordinary pro/con facts already established publicly are not offered again as new standalone contributions; required answers, issue reactions, and stance movement remain available. The two long diagnostic cases can temporarily allow reason reuse to expose what degrades when deliberation is deliberately extended; normal defaults remain unchanged.

## Simplified policy

There are no urgency scores, floor multipliers, candidate-score formulas, or public-pressure scores.

- Engagement maps to a configurable probability of submitting a self-selected voluntary bid. Required answers are separate; any later unsolicited participant comment is self-selected.
- Stubbornness maps to a configurable movement probability after a concrete trigger. Movement may mean making an option acceptable or visibly switching preference.
- Every acceptance or switch stores one concrete movement reason from the persona stance or resolved issue. The acceptance must make that reason visible; a later vote may stay short because the rationale is already public.
- A rank-3 neutral option may become compromise material directly. A rank-2 disliked option becomes eligible only after that participant's concrete concern was visibly resolved or softened. Rank-1 and hard-blocked options never become acceptable.
- Verbosity maps to a configurable maximum word count.
- Directness maps to one short wording instruction.
- The floor uses categorical priority: required answer, active-concern or moderator-stimulus response, ordinary contribution. Concern owners and other responders remain voluntary.
- A direct question clearly names one participant in a natural position and creates one mandatory answer. The same addressee/option/concern question is not asked again. Afterward, at most one ordinary voluntary reaction may continue the exchange; otherwise the question closes immediately.
- Concern responses are voluntary. Up to two distinct non-owners may respond, and the concern owner may voluntarily accept, soften, or maintain the concern. The same semantic concern is opened only once during discussion and may be reopened at most once during narrowing; otherwise it remains a stale public reservation.
- Ties inside one category are resolved with seeded random selection.
- Ordinary discussion limits each simulator to one newly opened concern by default and also deduplicates the same concern across the whole group.
- Stagnation exposes one simulator-owned compromise opportunity; it never forces a switch. During ordinary discussion, a visible moderator compromise prompt is committed only together with a successfully realized participant response. During a complete public split, narrowing states the split once and names only options that somebody publicly prefers or accepts. If nobody moves, the moderator acknowledges that silence and moves directly to the vote instead of asking a question and immediately speaking again.
- Soft coverage remains voluntary. The moderator opportunity does not bypass engagement, and an untouched option that receives no bid is marked as having no expressed interest rather than receiving manufactured content.
- Public unanimity after roughly one post-opening contribution round may proceed to voting before the minimum budget, so liveness handling does not add filler to an already settled discussion.
- Once a movement action wins the floor, language failure cannot erase it: after one focused repair the runtime commits one of several concise grounded movement fallbacks and records that fallback explicitly. Movement prompts allow natural acceptance/switch wording instead of requiring one fixed contrast formula.

## Language realization

The realization prompt keeps behavioral state unchanged while improving surface language. Reaction-like actions receive compact relational context: the previous speaker, that speaker's visible point, and how it supports, conflicts with, or matters differently to the current simulator's priority. Each persona also keeps two stable wording tendencies, such as leading with the conclusion, acknowledging another view first, using contractions, or preferring one compact sentence. These tendencies affect wording only.

Stance movement distinguishes concern resolution, a benefit outweighing a remaining trade-off, group compromise, and choosing an option that was already publicly acceptable. This prevents unrelated benefits from being described as if they solved a concern. Recent sentence openings are supplied as soft patterns to avoid, while fallback movement wording uses clause-safe two-part forms rather than inserting arbitrary reason fragments after `because`. Formal votes remain short and may not transfer a reason from another option.

## Reasons and grounding

Ordinary actions primarily use:

1. persona-specific `reason_for` and `reason_against`;
2. option `upside` and `concern` as fallback;
3. raw option attributes only when a direct question, active issue, persona reason, or concrete comparison needs them.

Persona reasons are whitespace-normalized but never truncated by word count. The setup prompt still asks for brief grounded reasons, while the full generated or manually supplied reason remains available to simulator policy and realization.

The structured action is authoritative. Validation blocks only hard failures such as unusable output, unknown options, unsupported concrete values, unrelated direct answers, genuinely ambiguous votes, invisible required stance changes, hard-blocker contradictions, and near-verbatim self-repetition. During voting, a short message that visibly names exactly the intended option is sufficient even without a fixed vote verb. An incomplete comparison is accepted as a useful one-sided contribution and is not recorded as public comparison evidence.

No validator LLM is used. Realization prompts require literal option names and treat supplied facts as atomic. They explicitly forbid invented option subtypes, facilities, schedules, costs, use cases, consequences, guarantees, absences, unsupported relative claims such as “shortest” or “best value,” and stronger or weaker mutations of supplied facts. Narrow deterministic checks reject high-risk unsupported strengthening and clear cross-option reason transfer during a vote or visible stance movement. They do not claim complete semantic entailment for every paraphrase. Failed ordinary realizations remain diagnosable in `run.json`. A selected stance movement or formal vote can never disappear: after one generation and one focused repair fail, the runtime renders a minimal grounded statement for the simulator’s already-authoritative action.

## Running

Install dependencies for the configured LLM provider (`.env` supplies the API key), then run:

```powershell
py .\main.py "Choose a Saturday study location"
```

With no CLI topic, the program uses the manual scenario when `environment.mode: manual`; otherwise it asks interactively for a topic. Automatic setup creates one paragraph of one or two sentences describing the shared situation and constraints, followed by the fixed option board. The context must be compatible with every option and cannot assign an option-specific cost, duration, capacity, availability, or outcome to the whole scenario. Setup then repairs invalid short aliases and falls back to deterministic natural phrases from the full option names, so valid boards are not discarded because an abbreviation such as `BA via LHR` was rejected.

## Deterministic tests

```powershell
py -m pytest -q
```

The deterministic suite currently contains 154 passing tests covering configuration, simulator authority, opening variation, mandatory direct answers, optional question follow-ups, voluntary concern participation, rare unknown-information answers, visible issue wording, rank-2 concern gating, adaptive narrowing, optional compromise, authoritative movement and vote fallback, deterministic tied choices, hard blockers, logging, bounded pacing from two through seven participants, and the definitions of the evaluation scripts (scenario file format, sweep-variant validity).

## Evaluation tooling

All LLM-backed evaluation lives in `eval/`; the shared metrics flattener `src/eval.py` belongs to the runtime because it exposes the runtime's own metrics schema. Every script overrides configuration in memory only — `config.yaml` on disk is never modified — and writes incremental CSV plus Markdown summaries, so interrupted batches keep their partial results.

### 1. Focused case suite — `eval/run_eval_suite.py`

Seventeen hand-built cases across ten topics and every group size from 2–7 with pinned personas, seeds, and expected structural properties (hard-blocker integrity, direct-answer ordering, bounded re-voting, movement commitment). Fifteen use normal pacing; two `long_*` diagnostic cases use isolated stress overrides.

```powershell
py .\eval\run_eval_suite.py --list
py .\eval\run_eval_suite.py --case grounding_sensitive_shipping_n4
py .\eval\run_eval_suite.py
```

Outputs land in `eval/logs_eval_suite/` and are zipped as `eval/logs_eval_suite.zip`.

### 2. Scenario batch — `eval/run_scenarios.py`

Runs each `participant_count | topic` line of `eval/scenarios.txt` (102 deliberately diverse everyday topics, counts 2–7 balanced) as one complete automatic run: scenario generation, persona generation, full dialogue.

```powershell
py .\eval\run_scenarios.py --list
py .\eval\run_scenarios.py --limit 5
py .\eval\run_scenarios.py --counts 5 --seed 500
```

Outputs: `eval/logs_scenarios/scenario_runs.csv` and `scenario_summary.md`.

### 3. Config sweep — `eval/run_config_sweep.py`

One-knob-at-a-time sensitivity analysis over every numeric value in the `conversation:`, `simulator:`, and `language:` config sections. Each parameter is run with a smaller value, the shared current-config baseline, and a larger value (default 3 runs per variant, same topic, same seeds), so differences between variants come from the knob. Derived values always satisfy the config-validation constraints; knobs that cannot affect the chosen group size (for example large-group caps in a three-person run) are skipped with a note.

```powershell
py .\eval\run_config_sweep.py --list
py .\eval\run_config_sweep.py --params voluntary_turns --runs 1
py .\eval\run_config_sweep.py --participants 6 --params large_group
```

Outputs: `eval/logs_config_sweep/sweep_runs.csv` and `sweep_summary.md`.

### 4. Transcript judge — `eval/judge_transcripts.py`

Post-hoc LLM judging in the style of ChatEval (Chan et al., ICLR 2024): up to three judge agents with diverse role personas (conversation analyst, behavioral scientist, fact auditor) assess each run one-by-one — later judges see earlier assessments — and their 1–5 scores for naturalness, coherence, groundedness, persona consistency, and decision quality are averaged.

The judge defaults to the `uni` provider (the local Ollama endpoint from config.yaml) rather than the dialogue provider, so the runtime model is not grading its own writing; `--provider`/`--model` override this.

```powershell
py .\eval\judge_transcripts.py --logs eval\logs_scenarios
py .\eval\judge_transcripts.py --judges 1 --limit 3
py .\eval\judge_transcripts.py --provider gpt   # judge with the runtime provider instead
```

Outputs: `eval/logs_judge/judge_scores.csv` and `judge_summary.md`. The judge is diagnostic only and never influences the runtime.

## Logs

Each normal run writes:

- `transcript.md`: option board, participant trait summary, visible chat, outcome, compact metrics;
- `run.json`: structured run state, core diagnostics, and compact failed-generation records;
- `metrics.csv`: one flat comparison row per run.

Deep generation attempts and action traces are optional:

```yaml
output:
  debug_metrics: false
  write_action_trace: false
```

## Main files

```text
main.py                  CLI entry point (topic → one full run)
config.yaml              all behavioral probabilities and language limits
src/config_loader.py     validated configuration access
src/models.py            structured data model (UserAction is authoritative)
src/builders.py          scenario and persona setup
src/simulator.py         participant-local policy and floor selection
src/dialogue.py          phase loop and public state updates
src/prompts.py           compact action-to-language prompts
src/validation.py        minimal deterministic validation/grounding
src/aliases.py           canonical option-alias contract and vote resolution
src/consensus.py         public narrowing and vote outcomes
src/logger.py            compact transcript and structured logging
src/llm_client.py        single provider abstraction (uni/groq/gemini/gpt)
src/eval.py              flat metrics view consumed by the eval scripts
eval/run_eval_suite.py   17 pinned LLM-backed evaluation cases
eval/run_scenarios.py    scenarios.txt batch runner
eval/run_config_sweep.py one-knob-at-a-time config sensitivity sweep
eval/judge_transcripts.py ChatEval-style multi-judge transcript scoring
eval/experiment_common.py shared in-memory config overrides and run helpers
eval/scenarios.txt       102 diverse `count | topic` batch cases
```

See `info/` for the detailed architecture.
