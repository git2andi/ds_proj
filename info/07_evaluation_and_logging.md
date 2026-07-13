# 07 — Evaluation and logging

Each run writes `transcript.md`, a detailed `run.json`, and one concise `metrics.csv` row.

`run.json` contains the scenario, personas, visible turns, current runtime stance, threads,
repair history, controller trace, final outcome, grouped metrics, and mutually exclusive token
usage buckets. Detailed validation issue codes and state transitions remain in the trace rather
than becoming dozens of summary columns.

The metric schema is grouped into:

- run structure: participant/moderator turns, word lengths, question density;
- participation: expected engagement/share and realized counts/shares;
- traits: defensible verbosity and switch-opportunity measurements;
- interaction: question/concern threads, direct/group response rates, functional direct-address and participant-reference rates, pairwise interaction density, self-selected act mix, normal-discussion compromise/movement, repair attempts/switches/holdouts, repetition;
- decision behavior: visible votes, outcome, switches, lean shifts, coverage, compromise, blockers;
- validation/grounding: repairs, fallbacks, drops, critical grounding interventions, runtime validator calls;
- token usage: setup, participant, moderator, repair, runtime validation, and total.

Rates are null when their denominator is zero. Private rank preference is explicitly labelled
private and never reported as public support. The suite CSV keeps stable scalar aggregates and
flags excessive turns, post-vote loops, repairs, fallbacks, drops, vote inconsistency, blocker
violations, missing votes, repetition, poor thread completion, validator calls, and abnormal tokens.

Run deterministic tests first:

```powershell
py -m pytest -q
```

The ten-case live suite is:

```powershell
py .\eval\run_eval_suite.py
```


The metric schema version is `3.1`. A `floor_autonomy` section reports the authority split and floor behavior: authority-source distribution, self-selected vs protocol-forced vs direct-answer turns and the self-selected ratio, bid rounds and no-bid rounds, per-simulator claim rate / average willingness / floor wins, the submitted-act distribution, intended-vs-realized act match rate, invalid-bid counts by reason, next-best-bid substitutions, the maximum speaker chain, and the engagement-vs-floor-win correlation. Verbosity reporting separates configured verbosity, the average word budget actually assigned to each participant, realized average words, and budget adherence. `verbosity_budget_correlation` tests whether configuration reaches the assigned budget; `verbosity_behavior_correlation` tests whether assigned budgets reach visible length.

`critical_grounding_interventions` counts accepted turn lifecycles whose original or repaired candidate triggered a high-confidence deterministic grounding check. It is not a claim that the final transcript has undergone exhaustive semantic fact auditing.

## Realization and obligation integrity metrics

The evaluation distinguishes genuine `true_no_claim_rounds` from `generation_failure_rounds`. It also records `valid_bid_attempts`, `final_dropped_intents`, `protocol_obligation_failures`, `repeated_bid_rejections`, accepted versus expected openings, accepted formal votes, and discussion-phase conditional acceptances. Failed wording must not be interpreted as participant silence or open-floor engagement.
