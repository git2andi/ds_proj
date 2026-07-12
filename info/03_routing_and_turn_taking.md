# 03 — Routing and turn-taking

The router decides who speaks next, which macro act they perform, who they address, and which option/thread they focus on. Routing is read-only over dialogue state: it returns a `MoveIntent` and never mutates persistent state; effects only count after the final accepted utterance's validated visible evidence is observed (the observer consumes exactly the evidence object that passed validation — it never reparses text).

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

A hot thread drives the next local move, and the routed act always matches the decided objective. Concern threads: a low-stubbornness advocate concedes (CONCERN), a committed advocate defends (SUPPORT), a bystander who shares the dislike adds a grounded doubt (CONCERN), a neutral bystander grounds the issue in the listed facts (COMMENT). Blocker threads route one bounded probe (ASK), then a backer points to the addressing fact (SUPPORT) or anyone else acknowledges the blocker's weight (COMMENT). Comparison threads route engagement with the same trade-off (COMPARE). Cooling continuation lets the raiser visibly accept (SUPPORT) or push back once (CONCERN) — direction picked by a stubbornness-weighted draw — another participant react to an answer (questions), or a new voice join a comparison; probabilities come from `threads:` config, bounded to freshly cooled threads. There is no hidden commitment float anywhere in these decisions: they read ranks, traits, and thread state only.

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

Question *scope* comes from validated visible evidence only: a named or "you"-directed question is direct; a genuine question without an addressee is a group question with no target (rhetorical tags open nothing). The controller (never the interpreter) assigns the group respondent by relevance, engagement, and turn-share deficit. The required respondent answers on the next turn; an unrelated turn by that respondent does not close the question, and a fallback line resolves it only when its accepted answer evidence says it addressed the target (the deterministic listed-fact answer families do; nothing else does).

## Macro acts

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Normal sampling is limited to `support, concern, ask, compare, comment`. `answer` is route-driven by question threads; `process`/`compromise` belong to narrowing and repair; softening is an observed stance effect parsed from visible text, never a routed act.

## Parameter influence

- engagement -> contribution frequency (expected turn share);
- verbosity -> average utterance length (numeric word budgets, soft targets);
- directness -> wording bluntness (and a higher concern/challenge prior);
- stubbornness -> discussion-phase defense, concession, and softening — never final switching;
- switch_resistance -> final movement only: switches, compromise acceptance, holdout concession, vote/repair resistance;
- speech_style -> lexical and register variation, never a routing signal.
