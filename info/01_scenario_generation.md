# 01 — Scenario and environment generation

The environment defines the factual decision space. It should provide enough shared facts for sims to argue concretely without requiring external knowledge.

## Modes

`environment.mode = auto`:

- the CLI topic is sent to the setup LLM;
- the setup LLM creates the topic framing, shared context, opening question, and option cards;
- generated options are validated before the discussion starts.

`environment.mode = manual`:

- `environment.manual` in `config.yaml` defines the topic, context, and options;
- the CLI topic is ignored;
- the scenario setup LLM call is skipped;
- this is the preferred mode for controlled tests.

## Option board

The option board is the source of truth. Each option should have an ID, name, concrete attributes, upside, tradeoff, concern, best-for note, and preferably a short alias.

The board should expose tradeoffs. Avoid one obvious winner unless the goal is to test quick consensus.

## Grounding rule

Sims may use only:

- option facts;
- shared context;
- prior visible dialogue;
- explicit uncertainty.

They must not state unsupported concrete facts as known. Hypothetical mitigation is acceptable only when marked as uncertain.

## Current relevance

The option board remains central. The controller may move holdouts only when the tested candidate is plausible according to visible votes, blockers, resistance, rank state, and traits. It does not create new options or blended plans.
