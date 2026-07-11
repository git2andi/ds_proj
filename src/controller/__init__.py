"""Controller package: runtime state, thread engine, routing policy, and flow.

- state.py   — controller runtime dataclasses (phases, threads, repair)
- threads.py — issue keys, thread lifecycle, primary-thread selection
- policy.py  — pure route/speaker/act/option selection returning MoveIntent
- flow.py    — phase transitions, narrowing/vote readiness, repair state machine
"""
