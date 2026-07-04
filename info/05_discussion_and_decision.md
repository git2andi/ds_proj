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
- propose a compromise;
- soften toward another option.

## Mid-discussion stance movement

A sim can move its internal lean before final voting if visible discussion supports that movement. Softening should not count as a final vote. It should prepare later votes.

Examples:

```text
I still like the bike ride, but the museum is starting to make more sense if we care about leaving time in the evening.
```

```text
I am not fully sold, but I could see Piazza working better for the whole group than my first pick.
```

## Final voting

Every sim should produce a visible final stance. Final votes are not trait-weighted; all participants need an observable stance for the outcome to be valid.

## No-majority handling

If final votes produce no majority, the controller should run bounded narrowing rather than close immediately.

Correct sequence:

```text
1. detect the vote split;
2. choose a candidate or top-two pair;
3. ask non-candidate voters concrete reservations;
4. let a supporter respond;
5. ask each relevant holdout to switch, stay, or propose an alternative;
6. optionally test one alternative candidate;
7. close as successful, majority, or unresolved.
```

## Current open issues

The latest full suite showed better split handling, but still not enough. Candidate selection can be wrong, post-reservation decisions are too vague, and some compromise turns attach concerns to the wrong option.
