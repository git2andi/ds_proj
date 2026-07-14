# Moderator

Configuration exposes one flag:

```yaml
moderator:
  enabled: true
```

When enabled, deterministic moderator messages may:

- introduce the task and public option board;
- issue at most one neutral stall prompt;
- issue at most one neutral prompt for an uncovered option;
- announce a leading option or top pair;
- request formal votes or the single re-vote;
- report the final result.

The moderator never selects a participant action, reason, concern, compromise, concession, stance change, or vote. It does not select a respondent for a group question and does not force an option to receive support.

When disabled, no visible moderator turns are emitted. Opening obligations, phase transitions, direct-question obligations, narrowing, voting, vote counting, and closure still run internally.
