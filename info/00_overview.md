# Runtime overview

The project is an option-grounded multi-user decision simulator. It is not a generic chatbot and not an open-ended social simulation. A fixed public option board defines the objective facts. Several simulated users discuss those options and close with a `successful`, `majority`, or `unresolved` result.

The central design decision is that structured simulator actions are authoritative:

```text
OPENING -> DISCUSSION -> NARROWING -> VOTING -> CLOSED

all eligible simulators evaluate their local policy in Python
    -> each proposes silence or one complete UserAction
    -> the floor selects one bid without rewriting it
    -> the dialogue LLM realizes that action as natural language
    -> minimal validation accepts, repairs once, or drops the rendering
    -> state updates are committed from UserAction
```

Each simulator privately owns its preference state, goal, background, traits, possible hard-blocker status, action choice, stance evolution, and final vote. The environment owns only protocol: phases, mandatory openings and votes, direct-answer obligations, light floor arbitration, broad group pacing, one active issue, neutral moderator messages, candidate derivation, vote counting, and one bounded re-vote. Public candidate evidence is participant-distinct, direct-answer obligations outrank phase transitions, and preferred switches require new external evidence plus a short hysteresis window.

The runtime does not parse utterances to infer acts, reasons, stance changes, issues, or votes. Text checking is limited to hard failures such as malformed output, missing required option mentions, premature formal-vote language, invented concrete facts, contradictory concrete comparisons, irrelevant direct answers, ambiguous votes, hard-blocker contradictions, issue-effect visibility, and near-verbatim repetition. Pairwise comparisons use the named/focused peer, while lowest/highest and shortest/longest claims are checked globally. Natural acceptance and switch expressions are checked only for broad visible consistency with the authoritative structured action; a discussion switch may name only the new preference because the old preference is already public, whereas a formal vote switch uses one vote-specific old-to-new bridge contract.

The active source modules are:

```text
src/
    aliases.py
    builders.py
    config_loader.py
    consensus.py
    dialogue.py
    llm_client.py
    logger.py
    models.py
    prompts.py
    simulator.py
    utils.py
    validation.py
```

The former controller package, parser, semantic interpreter, observer, style tracker, complex thread engine, and validator-LLM infrastructure have been removed.
