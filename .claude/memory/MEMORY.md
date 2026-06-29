# Memory Index

- [File scope rule](feedback-file-scope.md) — only CLAUDE.md; never touch AGENTS.md or .agents/
- [KF23 persona refactor](project-kf23-fix.md) — minimal persona schema (preferred_options list, optional rejection); scores/lists/role/etc removed; resolved 2026-06-29
- [P0 naturalness](project-p0-naturalness.md) — background+goal in per-turn card; hard word limits; reactive move guidance; casual greeting/farewell; alias variation; resolved 2026-06-29
- [KF04 alias contract](project-kf04-alias-contract.md) — single `validated_short_alias` gate in aliases.py; all consumers use `short_alias_map`; resolved 2026-06-29
