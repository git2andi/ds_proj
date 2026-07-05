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

The split summary in no-moderator mode is deterministic participant-owned procedure. This avoids one extra LLM call and prevents the candidate/counts from drifting in a paraphrase.

## Current validation focus

Run the full eval suite and inspect `q01`, `q02`, `q04`, `f03`, `f04`, and `f06`. The tested candidate should follow visible support, and unresolved outcomes should feel earned after explicit holdout decisions.
