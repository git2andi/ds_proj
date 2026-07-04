# Consensus and outcomes

**Code:** `src/consensus.py` (`ConsensusManager.finalize`), `src/parsing.py` (the
observer's parsing), `src/observer.py` (`ObserverMixin`: apply semantics, `_set_vote`),
`src/models.py`.

The outcome is read from what the transcript **visibly says** — never from hidden
state. This is the project's most important rule.

## Outcome types

`ConsensusManager.finalize` returns one of three, from visible votes only:

```text
successful   every participant visibly committed to the same option
majority     one option has a unique majority of visible commitments
             (support >= ceil(consensus.majority_fraction * n) AND no tie at the top)
unresolved   no unique majority after bounded discussion + finalization
```

A `majority` is **not** full agreement, and the transcript must show that difference:
the majority close names the holdouts (`04`).

## The visible-text-only rule

If a sim privately prefers B but never says so clearly, the group has **not** reached
support for B. Hidden metadata, initial goals, current lean — none of it counts.
`finalize` reads only `explicit_vote` per sim. Latent preferences are carried in the
outcome metadata for analysis, never for the decision.

## How the observer reads a line (`parsing.py`)

`parse_dialogue_act` turns each printed line into a `DialogueAct`: option references,
addressee, whether it's a genuine question (creates an obligation), and — the key
part — a **visible commitment** if any. Parsing is deliberately conservative: a false
"they voted" is worse than a missed vote.

**Clear commitments** (count as a vote):

```text
"I vote for B."   "B gets my vote."   "my pick is B."   "I'd go with B."
"B works for me as the final choice."   "let's go with B."   "count me in for B."
```

**Weak / conditional support** (do *not* close the outcome):

```text
"I can support B, but are we okay with the cost?"   "maybe B could work."
"I lean toward B."   "B sounds interesting."   "only if…"   "unless…"
```

A conditional line can trigger a clarification prompt, but it stays unresolved.

## Vote stability

Once a sim casts a clear vote, `_set_vote` keeps it unless the text explicitly signals
a change ("actually I vote for…", "switch to…", "changed my mind") or the controller
sanctioned a change. One exception: a formal direct vote replaces an earlier casual
*acceptance* ("X works for me") so a soft line can't lock out the real vote round. A
direct vote never silently overwrites another direct vote.

## Sanctioned switches and bridge clauses

On turns where the controller explicitly invites a vote change
(`intent.allow_vote_change` — the minority check and split-vote compromise, `04`), the
parser accepts a commitment carrying a **concessive bridge** ("X works for me even
though…", "as long as…"); genuine prerequisites ("only if", "unless") and questions
still block. Everywhere else, commitment parsing stays strict.

The bridge itself is enforced by the validator (`switch_bridge_ok`, issue 5, `05`):
a switch away from the current lean must name the old stance (or concede) **and** give
a reason. Recorded per sim as `switch_events` (`from`, `to`, `has_reason`,
`has_bridge`).

## Blockers bind like a rejection

The parser detects option-tied active blockers ("dealbreaker", "doesn't work for me",
with a negation guard), explicit resolutions ("that fixes my concern; I can live with
X"), conditional support, and compromise offers (including question forms). A visible,
unresolved blocker binds exactly like a setup rejection: the sim cannot commit to that
option until a visible resolution line exists in the same or an earlier turn.
Committing to an actively blocked option is a blocking validation issue
(`BLOCKED_OPTION_ACCEPTED`) — the line is replaced by a fallback, never printed.

A hard blocker's **setup-level** rejection is never cleared by a casual line; only
parser-derived blockers can be resolved in-dialogue.

## Latent lean moves only on visible signals

The sim's internal `current_preference` moves only on a parsed signal in the visible
text — a vote/acceptance, a compromise offer, a proposal, or explicit conditional
support — gated by `_can_shift_to` (which respects rejections and stubbornness). It
never moves from routing intent alone. This keeps the private state honest, but
remember: the *outcome* still uses only `explicit_vote`, never the lean.

## Split votes don't loop

If votes split with no majority, the controller does not re-poll endlessly. It makes
**one** bounded compromise attempt (`04`) and otherwise closes `unresolved` with a
clear reason. Hard turn caps guarantee termination.
