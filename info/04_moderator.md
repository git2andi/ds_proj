# 04 — Moderator behavior

The moderator is optional visible scaffolding. The framework still owns phase/protocol logic and the simulators still own their behavior even when the moderator voice is disabled.

## Configurable jobs

The moderator can perform:

- opening / option board presentation (fixed neutral wording: board + shared context, then "Let's discuss which option fits best overall." — never criteria selected by the setup);
- occasional mid-discussion nudges (a targeted nudge opens a direct question thread: the named participant owes the next answer);
- moderator-led narrowing summaries when the discussion was circling or the target length forced narrowing;
- final vote calls and vote-clarification prompts;
- closing.

Disabling moderator turns does not disable open-floor bidding, narrowing, vote logic, repair, or outcome computation. With `mid_discussion_nudges` off, a public `stall`/`coverage` stimulus lets simulators self-select a useful contribution. With `final_vote_call` off, framework vote-count narration and narrowing summaries are omitted rather than attributed to a participant; any visible participant process turn must originate from a simulator bid or obligation.

## Low-/no-moderator mode

In low-/no-moderator mode the framework may still change phases and schedule formal obligations, but it does not put procedural wording into a participant's mouth. Outcome closure may therefore have no visible closing line.

## Closure

When moderator closing is enabled, the final line is deterministic and status-aware: it cannot describe a majority as unanimity or announce a winner for an unresolved result. When disabled, closure is recorded in state/logs without a framework-authored participant acknowledgement.

## Style interaction

Moderator style is separate from participant speech_style. Participant age/speech_style should affect participant utterances, not moderator scaffolding.
