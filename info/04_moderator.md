# Moderator

The moderator has one configuration switch:

```yaml
moderator:
  enabled: true
```

When enabled, deterministic moderator messages may:

- open the discussion;
- issue one stall or neutral compromise prompt;
- issue one soft-coverage prompt;
- announce narrowing;
- request votes;
- report the outcome.

The moderator never chooses a participant's reason, stance, compromise, or vote.

When disabled, the environment still changes phases and collects votes, but no moderator turns are visible.


Moderator prompts create public opportunities only. A clear-leader prompt is directed conceptually at dissenters, while supporters may still react voluntarily to a concrete concern or proposal.

Soft-coverage prompts do not bypass simulator engagement. A complete split is announced once during narrowing so the visible transcript explains why a compromise window occurs, but the moderator still does not assign a mover or require a concession.

When a visible complete-split compromise prompt receives no participant bid, the moderator acknowledges the lack of movement (for example, “No one? All right, let’s record the final votes.”). This is a transition message, not a forced participant response.
