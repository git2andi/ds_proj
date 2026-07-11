# 03 — Routing and turn-taking

The router decides who speaks next, which macro act they perform, who they address, and which option/thread they focus on. Routing is read-only over dialogue state: it returns a `MoveIntent` and never mutates persistent state; effects only count after the final accepted utterance is parsed and observed.

## Routing order

One readable routing function (`controller/policy.py::_route_discussion_turn`) applies this priority order:

```text
1. required direct answer            (hot question thread's required respondent)
2. hot primary thread                (deterministic primary selection)
3. cooling primary thread            (probabilistic continuation, never scripted)
4. option coverage                   (only when no hot thread exists)
5. rare same-speaker continuation    (short addendum, chain-capped)
6. normal weighted discussion act
```

Phase/progress nudges run in the discussion loop before the router; narrowing/voting gates live outside act sampling (`controller/flow.py`).

## Threads

Local interaction is tracked as threads (`question`, `concern`, `blocker`, `comparison`; `repair` is phase-specific) with statuses `hot / cooling / resolved / stale`. Thread identity is `(type, focus options, deterministic issue key)`; the engine in `controller/threads.py` owns all lifecycle transitions and selects one deterministic primary thread per route decision (repair > direct question > group question > hard blocker on candidate > candidate concern > other hot > cooling).

A hot thread drives the next local move: concern threads route an advocate's defense (or honest concession, depending on tracked commitment), blocker threads route one bounded probe of the blocker and then honest mitigation responses, comparison threads route engagement with the same trade-off. Cooling continuation lets the raiser visibly accept or push back once (concerns/blockers), another participant react to an answer (questions), or a new voice join a comparison — with probabilities from `threads:` config, bounded to freshly cooled threads.

## Speaker choice

Normal turns combine:

```text
engagement-based expected turn share (actual share vs own target)
+ recent-speaker penalty and anti-monologue damping
+ minimum visibility for quiet sims
```

Thread turns use relevance, not engagement alone: stance/option relevance dominates, the turn-share deficit corrects imbalance, engagement is only a secondary prior, and just-spoke/ping-pong penalties keep the floor moving. A direct question's required respondent overrides normal ranking.

`engagement` is the only participation-share parameter: each sim gets an expected share (`0.30 + engagement`, normalized). Age/speech_style must not be used as a routing signal.

## Questions

The parser assigns question *scope* from visible text only: a named or "you"-directed question is direct; a genuine question without an addressee is a group question with no target. The controller (never the parser) assigns the group respondent by relevance, engagement, and turn-share deficit. The required respondent answers on the next turn; an unrelated turn by that respondent does not close the question, and a fallback line never resolves it.

## Macro acts

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Normal sampling is limited to `support, concern, ask, compare, comment`. `answer` is route-driven by question threads; `process`/`compromise` belong to narrowing and repair; softening is an observed stance effect parsed from visible text, never a routed act.

## Parameter influence

- higher directness increases concern/challenge behavior;
- higher stubbornness raises discussion-phase stance defense and resistance;
- higher switch_resistance raises final-movement resistance (switches, compromise acceptance, holdout concession);
- engagement decides how often a sim participates.
