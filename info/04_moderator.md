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

## Deterministic summaries

Split-vote and no-majority summaries should be deterministic or tightly controlled. They must be complete and concrete. Avoid vague prompts such as:

```text
Could those who prefer B or C live?
```

Correct style:

```text
We are split: A has two votes, B and C have one each. Let's test A first because it currently has the most support. Ben and Clara, what would still block A for you?
```

## Current open issue

Moderator/peer narrowing now happens, but candidate selection and follow-up logic still need improvement. The moderator must not test arbitrary candidates when a visible leading option exists.
