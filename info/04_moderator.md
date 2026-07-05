# 04 — Moderator behavior

The moderator is a configurable visible voice, not the whole controller. The hidden controller still manages state and routing whether or not the moderator appears in the transcript.

## Config flags

`moderator:` in `config.yaml` controls visible moderator turns:

```yaml
moderator:
  enabled: true | false
  opening: true | false
  mid_discussion_nudges: true | false
  final_vote_call: true | false
  closing: true | false
```

## Fully moderated mode

The moderator can:

- present the option board;
- nudge stalled discussion;
- call for final votes;
- ask holdout/split-vote questions;
- close the discussion.

## Light/no-moderator modes

When moderator functions are reduced, participants should perform more procedural work. High-initiative participants may call for picks, ask holdouts, or summarize split positions.

## Procedural style

Moderator and participant-owned procedure should be short and concrete. It should not add unnecessary conversational bulk.

Good procedural lines:

```text
We are split: A has two votes, B and C have one each. Let's test A first.
```

```text
Can the two B voters live with A, or is that still blocked?
```

Bad procedural lines:

```text
Maybe we should all reflect again on the broader implications of the decision and consider whether there is some integrated solution.
```

## Current open issue

Procedure exists, but it can still contribute to long, summary-like dialogue. Keep it deterministic where possible, but make it shorter. Candidate selection should follow visible support and blocker state.
