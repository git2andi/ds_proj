# 00 — Overview

The project is an option-grounded multi-user decision simulator. It creates 2-7 simulated participants, gives them tunable behavioral parameters, and lets them discuss a fixed option board until the run ends as `successful`, `majority`, or `unresolved`.

The target is not arbitrary chat. The target is an explainable simulator:

```text
topic/manual environment
  -> option board
  -> simulated users with hidden traits, age, speech_style, profile, and initial option ranks
  -> explicit phases, option coverage, and local interaction threads
  -> controller routes speaker / macro act / target / focus
  -> LLM renders one utterance
  -> deterministic critical interpretation and grounding checks
  -> observer updates visible state, option ranks, threads, and coverage
  -> consensus manager computes outcome from visible evidence
```

## Current stance model

Each sim stores one rank for every option:

```text
5 = preferred
4 = acceptable
3 = neutral / untested
2 = disliked but negotiable
1 = rejected / hard blocked
```

Derived helpers such as `top_option()`, `acceptable_options()`, `disliked_options()`, and `rejected_options()` are computed from this table. There are no separate runtime preference/rejection containers.

## Current participant model

A sim has hidden OCEAN traits that derive five simulator parameters (engagement, verbosity, directness, stubbornness, switch_resistance) for behavior. Age, speech_style, and profile are descriptive metadata for plausibility and surface wording.

Core rule:

```text
the five parameters decide behavior; speech_style changes wording only
```

Age/profile plausibility is checked during setup so generated personas do not contain obvious contradictions.

## Current progress model

There is no content agenda. Global progress is represented by explicit phases,
per-option visible coverage, and local question/concern/blocker/comparison
threads. Persona-specific reasons remain attached to option stances.

## Current act model

The controller uses a compact macro-act vocabulary:

```text
opening, support, concern, ask, answer, compare, comment, compromise, process, vote, closing
```

Only the macro set is used so routing, prompts, and logs remain aligned.
