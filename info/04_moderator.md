# 04 — Moderator behavior

The moderator is optional visible scaffolding. The controller still owns the decision policy even when the moderator voice is disabled.

## Configurable jobs

The moderator can perform:

- opening / option board presentation (fixed neutral wording: board + shared context, then "Let's discuss which option fits best overall." — never criteria selected by the setup);
- occasional mid-discussion nudges (a targeted nudge opens a direct question thread: the named participant owes the next answer);
- moderator-led narrowing summaries when the discussion was circling or the target length forced narrowing;
- final vote calls and vote-clarification prompts;
- closing.

Disabling moderator turns should not disable routing, narrowing, vote logic, or outcome computation. With `mid_discussion_nudges` off, an engaged participant owns the stall beat; with `final_vote_call` off, participants own the narrowing summary, split summary, and holdout probe.

## Low-/no-moderator mode

When final vote call or closing is disabled, participants own the procedural move. This should sound like a group member, not like a hidden controller.

If a participant voices a split summary, the first reservation response should not be that same participant answering their own prompt.

## Closure

Unresolved outcomes get a short participant or moderator acknowledgement before closure. The final moderator closing line is deterministic and status-aware: it cannot describe a majority as unanimity or announce a winner for an unresolved result. This makes endings socially clear without changing the already computed outcome.

## Style interaction

Moderator style is separate from participant speech_style. Participant age/speech_style should affect participant utterances, not moderator scaffolding.
