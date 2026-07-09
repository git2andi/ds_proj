# 01 — Scenario and environment generation

The environment defines the factual decision space. It should provide enough shared facts for sims to argue concretely without requiring external knowledge.

## Modes

`environment.mode = auto`:

- the CLI topic is sent to the setup LLM;
- the setup LLM creates the shared context and option cards (nothing else — there is no generated opening question or decision-kind category);
- generated options are validated before the discussion starts; an invalid attempt (including a missing/unusable/duplicate `short_name`) is rejected and retried.

`environment.mode = manual`:

- `environment.manual` in `config.yaml` defines the topic, shared context, and options;
- the CLI topic is ignored;
- the scenario setup LLM call is skipped;
- this is the preferred mode for controlled tests.

## Scenario schema

A scenario is exactly:

```text
topic
shared_context   (public facts every participant knows — the public source of truth)
options          (one card per option label)
```

Each option card has: `id`, `name`, `short_name`, `attrs`, `upside`, `concern`.

- `attrs` are concrete, stable, topic-natural attributes. The setup LLM chooses which attributes fit the topic; the prompt gives no example dimensions and the code hard-codes no preferred dimensions.
- `upside` is one positive argument; `concern` is one stable likely objection. There is no separate `tradeoff` or `best_for` field.
- `short_name` is a required concise natural alias copied from the name, unique across options. It is never derived by clipping words from the full name; full names are never mutated.

The board should expose real trade-offs. Avoid one obvious winner unless the goal is to test quick consensus.

## Moderator opening

The moderator opening is fixed and neutral: it introduces the topic, lists the option board and shared context, then ends with the criteria-free line "Let's discuss which option fits best overall." The setup never selects concrete decision dimensions for the first turns.

## Grounding rule

Sims may use only:

- option facts;
- shared context;
- prior visible dialogue;
- explicit uncertainty.

They must not state unsupported concrete facts as known. Hypothetical mitigation is acceptable only when marked as uncertain.

## Current relevance

The option board remains central. The controller may move holdouts only when the tested candidate is plausible according to visible votes, blockers, resistance, rank state, and traits. It does not create new options or blended plans.
