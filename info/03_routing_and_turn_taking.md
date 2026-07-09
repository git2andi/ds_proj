# 03 — Routing and turn-taking

The router decides who speaks next, which macro act they perform, who they address, and which option/thread they focus on.

## Speaker choice

Speaker choice combines:

```text
engagement-based expected turn share (actual share vs own target)
+ response obligations
+ unresolved questions / concerns
+ relevance to current option camps
+ recent-speaker penalty and anti-monologue damping
+ minimum visibility for quiet sims
```

`engagement` is the only participation-share parameter: each sim gets an expected share (`0.30 + engagement`, normalized) and the router boosts sims behind their own target and damps sims ahead of it. It does not equalize turn counts — a low-engagement sim is never pulled up to a high-engagement sim's count.

Age/speech_style must not be used as a routing signal. A young/casual participant should not speak more because of style, and an older/formal participant should not become more stubborn because of age. Those effects belong to the simulator parameters.

## Question obligations

If a participant is directly asked a question, they answer on the next turn unless a stronger validation/safety condition prevents it — there is no personality-based delay.

Group-directed questions pick a respondent by a weighted score: relevance to the question's option focus, engagement, expected-share deficit relative to the sim's own target, a recent-speaker penalty, and small randomness. Not simply the quietest person.

Proactive moves (questions, process beats, compromise tests) are routed from dialogue state: an unresolved issue makes a question worthwhile, a stuck thread or a silent participant makes a process move worthwhile, concentrating support makes a compromise test worthwhile. Opening order uses engagement plus light randomness.

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

## Parameter influence

- higher directness increases concern/challenge behavior;
- lower stubbornness increases compromise and softening moves;
- higher stubbornness raises resistance and stance defense;
- engagement decides how often a sim participates.
