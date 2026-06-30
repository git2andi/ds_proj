# Consensus and outcomes

## Outcome types

The simulator uses three outcome states:

```text
successful
  all participants visibly commit to the same option

majority
  a unique option receives visible support from a majority of participants

unresolved
  no unique majority is visible after bounded discussion/finalization
```

## Visible-text-only rule

Outcome calculation must use visible transcript text only. Hidden metadata, private preferences, initial goals, or internal simulator state cannot count as public support.

This is essential because the transcript is the observable interaction. If a sim privately prefers Option B but never says so clearly, the group has not visibly reached support for B.

## Clear commitments

Clear commitments include:

```text
I vote for Option B.
I vote for the evening layover.
B works for me as the final choice.
Let's choose B.
I am fine with B as the final decision.
```

## Weak or conditional support

These should not count as final votes:

```text
I can support B, but are we okay with the cost?
Maybe B could work.
I lean toward B.
B sounds interesting.
I do not hate B.
```

A conditional statement can trigger a clarification prompt, but it should not close the outcome.

## Vote stability

Once a participant gives a clear vote, that vote should remain stable unless the participant explicitly changes it.

Explicit change markers:

```text
I changed my mind.
Actually, I vote for C.
Then I switch to C.
```

Without an explicit change marker, later discussion should not accidentally overwrite the vote.

## Split votes

If all participants vote for different options, the moderator should not loop through repeated voting. It should either:

```text
make one bounded compromise attempt
```

or:

```text
close as unresolved with a clear reason
```

## Majority vs successful

A majority is not full agreement. The transcript should make that difference visible.

Good majority close:

```text
Two participants chose C, while Kenji still preferred D. The group proceeds with C by majority.
```

Good successful close:

```text
All three participants now support C, so the group chooses C.
```
