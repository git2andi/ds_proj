# Logging and evaluation

Each accepted participant turn records:

- phase and speaker;
- mandatory, voluntary, or liveness-forced status;
- the authoritative action and urgency;
- realized text and repair count;
- issue event;
- stance update;
- vote;
- intended versus realized word count;
- approximate input/output tokens.

Every generation attempt is also retained with the structured action, raw output, validation errors, repair output, repair errors, and final accepted/dropped status. This is diagnostic logging only; it does not reconstruct semantic state from text.

Each run writes:

- `transcript.md` with the public option board, persona cards, visible transcript, result, and metrics;
- `run.json` with complete structured state, accepted actions, generation attempts, issue history/outcomes, per-round vote records, validation-failure categories, and metrics;
- one compact `metrics.csv` row.

Metrics include participant and moderator turn counts, voluntary turns per simulator, comparable utterance lengths, intended/realized word targets, repairs and drops, near-verbatim repetition repairs, structured reason sources, question keys, distinct supporters/concern raisers, switch opportunities and cooldown decisions, issue provenance and relevant responders, public acceptances, visible switches, narrowing-focus adherence, complete per-round vote records, vote-switch validation, outcomes, LLM calls, and token usage. Engagement uses voluntary open-floor turns only. Verbosity uses comparable voluntary action types rather than mandatory votes or acknowledgments. Stubbornness is reported per actual switch/acceptance opportunity. Hard blockers report nonpreferred acceptances and votes, which must remain zero. Any unclear or generation-failed vote marks the run as protocol-degraded even when a mathematical majority can still be counted.

`transcript.md` contains a compact human-facing metric subset. Complete per-turn word budgets, issue provenance, switch decisions, generation attempts, and validation details remain in `run.json` so the transcript does not become an opaque debug dump.

`eval/run_eval_suite.py` provides an LLM-backed end-to-end suite. It uses the configured `llm.dialogue` provider to realize every selected action; simulator policies, floor arbitration, issue lifecycle, validation, phase progression, voting, and logging remain the production Python implementations. The suite requires the same provider credentials or endpoint as `main.py` and writes per-case logs plus CSV, JSON, Markdown, and ZIP summaries under `eval/logs_eval_suite`. Cases and diagnostics cover valid formal vote switches, final-turn direct questions, rapid-switch detection, concern relevance, structured reason/question diversity, persona distinctness, isolated verbosity/directness/style realization, early unanimous convergence, grounded option facts, hard blockers, majority closure, and one bounded no-majority re-vote. Policy-only calibration independently checks engagement bid frequency and stubbornness-dependent switching. Realization diagnostics hold the action/context constant while changing one language trait. The deterministic unit tests remain the reproducible architectural verification layer.

Validation is intentionally narrow: hard output correctness, grounding, formal protocol visibility, issue-effect visibility, and near-verbatim same-speaker repetition. It is not a semantic completeness grader or style judge. Rich social modeling, multiple simultaneous issues, unrestricted factual inference, and stable provider-independent persona voice remain outside this project's scope.
