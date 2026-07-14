# Autonomous multi-user decision simulator

This project implements option-grounded user simulators for multi-user conversational AI. Each simulated user owns its communicative action, option focus, reason, addressee, stance changes, and vote. A lightweight environment manages only protocol concerns such as phases, direct-answer obligations, floor arbitration, narrowing, and vote counting. The dialogue LLM realizes an already-selected structured action as natural language.

The active runtime has one authority path:

```text
private simulator state + public dialogue state
    -> seeded Python simulator policy proposes UserAction or silence
    -> floor selects one intact bid
    -> one dialogue-LLM realization call
    -> minimal hard-failure validation; at most one repair
    -> structured action commits public state, issue effects, stance update, or vote
```

Natural-language parsing is not used to reconstruct hidden dialogue state. There is no validator LLM, expected-turn-share correction, global participant-act controller, multi-thread engine, or unanimity repair. Public persuasion uses distinct participants rather than repeated mentions, and preferred-option switches require new external evidence plus a short hysteresis window. Questions are selected from visible information needs, concerns retain their public provenance, and only relevant mitigation, trade-off, or acknowledgment can affect concern resolution. A pending direct answer is drained before narrowing. Minimal realization checks cover hard consistency, grounding, formal protocol visibility, and near-verbatim self-repetition; they do not judge ordinary style or reconstruct state. Any unclear or failed formal vote is retained as an explicit runtime protocol degradation.

## Run

Install the dependencies used by the configured provider, set its API key where required, then run:

```powershell
py .\main.py "Choose a study location"
```

With `environment.mode: manual`, running without a topic uses the configured manual option board:

```powershell
py .\main.py
```

Logs are written under `logs/` by default.

## Test

```powershell
$env:PYTHONPATH = "src"
py -m pytest -q
```

## LLM-backed evaluation

The evaluation suite uses the configured `llm.dialogue` provider for every selected participant utterance. Simulator bidding, floor arbitration, structured state updates, validation, voting, and outcomes remain Python-controlled:

```powershell
$env:PYTHONPATH = "src"
py .\eval\run_eval_suite.py
```

It requires the same provider credentials or endpoint as `main.py`. It writes per-case logs, a CSV/JSON/Markdown summary, and `eval/logs_eval_suite.zip`. Because language realization is live, transcript wording, repairs, drops, and some quality checks may vary between runs even though simulator policies are seeded. The suite covers formal vote switches, adjacency-pair completion, switch stability, concern relevance, reason/question diversity, persona distinctness, isolated realization traits, early convergence, and the valid-majority/re-vote protocol. Multi-seed policy calibration remains separate for engagement and stubbornness. Full structured diagnostics stay in `run.json`; the human transcript keeps a compact metric view.

See `info/00_overview.md` through `info/08_configuration_and_running.md` for the final architecture.
