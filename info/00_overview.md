# 00 — Overview

The project is an option-grounded multi-user decision simulator. It creates 2-7 simulated participants, gives them tunable behavioral parameters, and lets them discuss a fixed option board until the run ends as `successful`, `majority`, or `unresolved`.

The target is not arbitrary chat. The target is an explainable simulator:

```text
topic/manual environment
  -> option board
  -> simulated users with initial option ranks
  -> controller routes speaker / macro act / target / focus
  -> LLM renders one utterance
  -> validation checks intent and grounding
  -> observer updates visible state and option ranks
  -> consensus manager computes outcome from visible evidence
```

## Current stance model

Each sim stores one rank for every option:

```text
4 = preferred
3 = acceptable
2 = neutral / untested
1 = disliked but negotiable
0 = rejected / hard blocked
```

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from this table. There are no separate runtime preference/rejection containers.

## Current act model

The controller uses a compact macro-act vocabulary:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Only the macro set is used. Legacy act aliases were removed so routing and prompts cannot silently rely on the old taxonomy.
