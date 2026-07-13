"""Controller package: runtime state, thread engine, floor arbitration, and flow.

- state.py   — controller runtime dataclasses (phases, threads, repair)
- threads.py — issue keys, thread lifecycle, primary-thread selection
- floor.py   — floor arbitration: collects/validates/scores simulator bids and
               selects a winner without rewriting it (no participant authoring)
- flow.py    — phases, protocol obligations, open-floor bid orchestration, and
               the bounded repair state machine

Participant behavior (act/target/focus/reason/vote) is decided by the simulator
policy in src/simulator.py, not by this package.
"""
