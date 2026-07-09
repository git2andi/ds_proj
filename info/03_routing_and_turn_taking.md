# 03 — Routing and turn-taking

The router decides who speaks next, which macro act they perform, who they address, and which option/thread they focus on.

## Speaker choice

Speaker choice combines:

```text
trait-derived participation pressure
+ response obligations
+ unresolved questions / concerns
+ relevance to current option camps
+ anti-monologue damping
+ minimum visibility for quiet sims
```

The router should not force perfect balance. High-engagement/high-initiative sims may speak more, but quieter sims must still show visible stances.

Age/style must not be used as a routing signal. A young/casual participant should not speak more because of style, and an older/formal participant should not become more stubborn because of age. Those effects belong to traits and simulator parameters.

## Macro acts

Current macro acts:

```text
opening, support, concern, ask, answer, compare, soften_toward, compromise, process, vote, closing
```

Routing, prompts, and logs use only the macro set.

## Local thread priority

Direct questions, recent answers, and unresolved concerns should usually be developed before a new issue is introduced. The remaining pre-vote work is tracked as a chat-level discussion agenda, not as per-sim agenda scripts.

## Chat-level agenda

`DialogueState.discussion_agenda` tracks global work the discussion still needs, such as option coverage. It is not a participant script. It helps the controller decide what is missing, then normal speaker selection chooses a suitable participant.

Personal perspective should come from option stances and private goals, not from hidden per-sim agenda items.

## Trait influence

- higher directness increases concern/challenge behavior;
- higher compromise tendency increases compromise and softening moves;
- stubbornness raises resistance;
- responsiveness helps with answers and invitations;
- engagement and initiative affect how often a sim participates.
