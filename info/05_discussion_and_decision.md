# 05 — Discussion and decision flow

The discussion should not be a sequence of isolated preference statements. Sims should react to each other, raise and answer concerns, soften during the discussion, and narrow before voting.

## Normal discussion phase

During discussion, participants may:

- build on an option;
- agree with a reason;
- challenge a tradeoff;
- ask or answer a concrete question;
- compare two options;
- invite a quiet participant;
- propose a bounded compromise;
- soften toward another option.

Local rhythm is enforced since 2026-07-06: a follow-up to an answer stays on the same point (agree/challenge/build, never a fresh ask), and once two questions are open in the recent window, statement-type acts are told to end on a statement. Direct Q→A adjacency stays strong; Q→A→new-question chains are the exception now.

## Pacing (P9)

Discussion length adapts to the conflict state instead of one fixed schedule:

- A cast that starts on the same option gets a lower minimum
  (`min_discussion_turns_per_participant − 1.5`, floor 3.0), so quick agreement
  may close earlier.
- Cast drive (mean engagement + 0.5·initiative) shifts the force/hard points by
  ±0.5·n turns: an engaged, high-initiative group earns more free turns, a flat
  one narrows sooner.
- Split initial preferences and low compromise tendency still add the
  contention/low-compromise extras.
- At the forced-narrowing point, a run that is still multi-camp **and** has an
  unaddressed open concern keeps discussing, bounded by the hard cap.

Rough sanity ranges (not targets): n=2 8-18 turns, n=3-4 16-35, n=5-7 24-50.
Observed targeted runs: n=2 19, n=3 26/31, n=4 36, n=5 45.

## Utterance length

Verbosity affects average length, not every turn: the word-budget logic mixes in deterministic short beats for all sims. Realized target bands (validated in the 2026-07-06 suite, avg 12.4-16.4 words/turn overall):

```text
low verbosity:       many turns around 5-10 words
medium verbosity:    many turns around 9-15 words
high verbosity:      many turns around 14-22 words
vote turns:          often under 12 words
continuations:       around 4-10 words
```

Short acknowledgements, direct disagreements, and quick answers are allowed and useful.

## Mid-discussion stance movement

A sim can move its internal lean before final voting if visible discussion supports that movement. Softening should not count as a final vote. It should prepare later votes.

A switch should have a visible reason. Examples of valid triggers:

- a concern was answered;
- the current favorite was challenged;
- another option gained visible support;
- a higher-priority group constraint became clearer;
- the sim explicitly accepts a tradeoff;
- visible majority pressure affects a high-compromise sim.

A bridge phrase alone is not enough if the actual blocker remains unresolved.

## Compromise and option combinations

Sims may propose bounded combinations, such as “A works if we borrow B's simpler setup.” This is allowed, but it must not always happen and it must not create a hidden fifth option.

The final winner should still be one concrete option from the option board, optionally with a visible condition. Since the 2026-07-06 naturalness round a deterministic tripwire (HYBRID_COMPROMISE) blocks compromise turns that weld two options into one plan ("X and also Y", "combined with"); such drafts are repaired to pin one option.

## Final voting

Every sim should produce a visible final stance. Final votes are not trait-weighted; all participants need an observable stance for the outcome to be valid.

Vote prompts suggest a rotating menu of parser-recognized commitment phrasings not yet used in the round, so vote lines stay varied but observable; subject-form lines ("X still gets my vote — Y hasn't fixed my concern") bind to the option before the phrase. When no moderator closing exists, one participant speaks a short deterministic wrap-up line ("So X wins for most of us, with N still not sold." / "Looks like we're not landing this one today.").

## No-majority handling

If final votes produce no majority, the controller runs bounded narrowing rather than closing immediately.

Current intended sequence:

```text
1. detect the vote split and visible vote counts;
2. rank concrete candidates deterministically from visible votes;
3. if one option has a strict plurality, test it first unless every relevant dissenter hard-blocks it;
4. if leaders are tied, choose by fewer blockers, lower resistance, and higher compromise fit;
5. ask relevant non-candidate voters for concrete reservations about the tested candidate;
6. let a supporter respond using only the candidate's known facts and explicit uncertainty for unknowns;
7. route each relevant holdout into a visible switch / stay / alternative decision;
8. if the first candidate fails, optionally test one alternative candidate;
9. close as successful, majority, or unresolved.
```

## Current validation focus

Inspect whether the transcript shows earned movement. A reader should be able to explain why each participant stays, switches, or refuses.
