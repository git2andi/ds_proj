# Discussion, issues, and stance updates

The runtime uses five phases only: `OPENING`, `DISCUSSION`, `NARROWING`, `VOTING`, and `CLOSED`.

Every participant first receives one mandatory opening action naming its initial preference and a reason. Discussion then uses autonomous simulator bids. Broad pacing is group-level: a minimum, soft target, and hard maximum count only voluntary open-floor turns. Openings, mandatory answers, votes, re-votes, and liveness-forced turns are excluded.

## One active issue

The state contains at most one `ActiveIssue` plus an append-only history. Supported kinds are `QUESTION`, `CONCERN`, and `COMPARISON`; statuses are `OPEN`, `RESOLVED`, and `STALE`.

A direct question creates both an issue and a mandatory response obligation. A group question creates an issue without choosing a respondent. Answered questions are recorded as answered/resolved when no relevant follow-up remains rather than being mislabeled stale. A concern retains the opening action's reason source and normalized issue key. A response counts as relevant only when its structured action supplies same-issue mitigation, explicitly weighs the drawback against another public benefit, or agrees that the concern remains. An unrelated upside does not accumulate resolution pressure. The concern owner alone may maintain, partially address, or resolve the concern, and reevaluates when new relevant evidence arrives. Resolution may atomically make the option acceptable or switch preference after the visible utterance passes validation. A comparison becomes an issue only when the same trade-off is independently developed or challenged.

Issues normally receive zero to three follow-ups. Continuation becomes less likely after the configured normal window, while the environment enforces a hard cap. An issue becomes stale when nobody continues it, a new issue takes priority, narrowing ends, or the cap is reached. Resolved and stale records remain in history.

## Structured stance changes

`UserAction.stance_update` may make an option acceptable, remove acceptance, switch the preferred option, or explicitly reject an option. The update is validated before generation and committed only after an aligned visible utterance passes validation. Other participants' turns cannot directly change someone else's stance.

During narrowing, candidates are derived only from public structured preferences, acceptances, distinct supporters, distinct concern raisers, comparisons, and visible switches. Repeated support from one speaker remains a raw logging occurrence but contributes only one persuasive participant. Private ranks are not used to manufacture a public leader. A preferred switch normally requires new evidence from another participant, a minimum accepted-turn distance, and a meaningful evidence advantage before the stubbornness-dependent probability is evaluated. Re-voting itself adds no switch pressure. Candidate actions are phase-specific: finalists, active issues, mandatory answers, explicit non-finalist defence, compromise, acceptance, rejection, and switching remain relevant; unrelated ordinary questions do not. A participant whose own option is already a finalist may still make another finalist acceptable or switch voluntarily.

When every participant has publicly converged on one option, no obligation or concern remains, and at least one voluntary discussion contribution has genuinely tested or confirmed that agreement, the system moves to a brief final-concern/narrowing step instead of producing redundant support. A newly raised concern still blocks early closure.
