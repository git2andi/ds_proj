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

The next quality target is local rhythm. The discussion should often spend several turns on one point before opening a new issue. Direct Q→A adjacency is good, but repeated Q→A→new question chains are not.

## Utterance length

Turns should be shorter than the current logs. Verbosity should affect average length, not force every high-verbosity sim to write long turns.

A plausible target for the next round:

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

The final winner should still be one concrete option from the option board, optionally with a visible condition.

## Final voting

Every sim should produce a visible final stance. Final votes are not trait-weighted; all participants need an observable stance for the outcome to be valid.

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
