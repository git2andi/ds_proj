# Topic examples

The automatic setup accepts a plain decision topic and creates four topic-specific options. Suitable topics have a clear choice and a small set of public trade-offs.

Examples:

```text
Choose a restaurant for a group dinner
Choose a weekend activity for four friends
Choose a shared coffee machine for an office
Choose a film for a movie night
Choose a charity project to support
Choose accommodation for a short group trip
Choose a board game for six participants
```

The system is not limited to fixed domains. Topic generality comes from structured option cards, not from hard-coded restaurant, travel, or product rules.

For reliable setup, topics should avoid requesting live prices, current schedules, or external facts that the configured LLM cannot verify. The generated option board is treated as the shared world for the discussion.

Manual mode is preferable when exact factual values or a specific experimental setup must be preserved.
