# TODO: active implementation plan

This file is the live work queue for the option-grounded multi-user decision simulator. It should contain current open work only. Remove an item only when deterministic code or fresh evaluation logs prove it is fixed.

## 0. Project framing

The project is an **option-grounded multi-user decision simulator**, not a generic chatbot or open-ended society simulator.

A run should follow this pipeline:

```text
one-line topic or manual environment
  -> fixed option-grounded decision environment
  -> 2-7 configurable simulated users
  -> controller selects speaker / addressee / dialogue act / option focus
  -> LLM renders one visible utterance
  -> observer updates public state from visible text
  -> discussion narrows through reactions, concerns, stance movement, reservations, and votes
  -> outcome = successful / majority / unresolved from visible votes only
```

Participant parameters must remain behaviorally visible: engagement, initiative, responsiveness, verbosity, stubbornness, directness, and compromise tendency should affect turn-taking, response timing, stance movement, and willingness to compromise.

## 1. Implementation protocol

1. Work one issue at a time, in priority order.
2. Prefer deterministic controller logic over additional LLM calls.
3. Keep prompts smaller and act-specific; do not solve negotiation quality by adding broad prompt text.
4. Validate behavior with transcripts and `run.json`, not execution success alone.
5. Test the four mode combinations across a full round:

```text
auto environment + auto participants
manual environment + auto participants
auto environment + manual participants
manual environment + manual participants
```

6. Include `n=2`, `n=3`, `n=4`, and at least one `n=5` run before claiming completion.
7. Update `README.md`, `CLAUDE.md`, and relevant `info/*.md` when behavior or workflow changes.
8. Run static checks before every handoff:

```powershell
py -m py_compile main.py run_eval_suite.py src\*.py
```

9. Before claiming behavioral completion, run and inspect:

```powershell
py run_eval_suite.py --full
```

## 2. Priority 1 — Improve split-vote and tie narrowing

### Problem

The system detects split votes and starts a narrowing pass, but the latest full logs still show weak social negotiation. In a visible `2-1-1` restaurant split, the controller tested a one-vote option instead of the two-vote leader. In tied `1-1-1` or `2-2` structures, the candidate can still feel arbitrary.

### Correct behavior

After final votes produce no majority:

```text
1. detect the vote structure, e.g. 1-1, 1-1-1, 2-1-1, 2-2;
2. choose a concrete candidate or top-two pair from visible votes;
3. if one option has a strict plurality, test that leader first unless it is visibly blocked by all relevant dissenters;
4. if the lead is tied, choose the candidate with fewer blockers and lower resistance / better compromise fit;
5. ask relevant non-candidate voters for targeted reservations;
6. let a supporter answer at least one concrete reservation honestly;
7. route dissenters into visible switch / stay / alternative / no-compromise decisions;
8. run at most one alternative candidate attempt if the first candidate fails and the turn budget allows;
9. close unresolved only after relevant dissenters had a chance to move or explain why they cannot.
```

### Validation

Inspect split cases around the first final vote. The tested candidate must be explainable from visible vote counts and reservations. For `2-1-1`, the two-vote option should normally be tested first.

## 3. Priority 2 — Fix compromise candidate scoring

### Problem

Candidate scoring currently lets flexibility and mover estimates overpower visible vote structure. It can also call stochastic shift logic while ranking candidates, which makes selection less stable than it should be.

### Correct behavior

Candidate ranking should be deterministic and roughly follow:

```text
strict plurality -> test the leading option first;
tied leaders -> fewer hard blockers, lower average resistance, higher compromise fit;
all tied -> least objectionable / most compromise-compatible option, not arbitrary order;
failed first candidate -> optionally test one alternative, then stop.
```

Resistance should use visible blockers, participant compromise threshold, stubbornness, current commitment, and whether the candidate appears in the participant's preferred options.

## 4. Priority 3 — Add explicit post-reservation decision steps

### Problem

After a holdout states a reservation and a supporter responds, the follow-up sometimes reads like another vague re-vote rather than a decision step.

### Correct behavior

Every relevant holdout after a reservation response should visibly do exactly one of these:

```text
switch to the tested candidate;
stay with the original option and name the blocker;
propose one concrete alternative candidate;
state that no acceptable compromise exists.
```

The observer must update visible vote state from that line. Avoid hidden state updates and avoid generic “I’m still thinking” phrasing in decision beats.

### Validation

In `q01`, `q02`, `q04`, `f03`, `f04`, and `f06`, the split-resolution section should show explicit switch/stay/alternative decisions before closure.

## 5. Priority 4 — Validate and improve `n=2` deadlock handling

### Problem

The latest full suite did not trigger the two-person deadlock protocol because the two-person case converged before deadlock. The protocol therefore remains unvalidated.

### Correct behavior

Add or adjust a deterministic eval case with two stubborn manual participants and opposing fixed preferences. A `1-1` vote should trigger:

```text
1. each person states their strongest blocker;
2. each person proposes one condition or concession;
3. each person makes a final switch/stay decision;
4. unresolved is valid if neither moves.
```

### Validation

The relevant run should set `two_person_deadlock_attempted = true` and its transcript should include the blocker/concession exchange before unresolved closure or a justified switch.

## 6. Priority 5 — Fix option-target confusion in compromise prompts

### Problem

Some reservations attach a tradeoff to the wrong option. Example pattern from the latest logs: a concern like “less flexible once booked” can appear while testing a Museum compromise, although that tradeoff belongs to Escape Room.

### Correct behavior

When testing candidate `X`:

- reservation prompts should bind strongly to `X`;
- supporter responses should use only `X`'s actual facts and clearly mark unknowns;
- comparisons to the holdout's original favorite are allowed, but unrelated attributes must not be transferred to `X`;
- grounding should flag wrong-option fact transfer when it appears.

## 7. Priority 6 — Reduce token cost

### Problem

The latest full suite still used roughly 460k input tokens across 12 runs. Approximate distribution:

```text
utterance calls: ~65% of input tokens
grounding calls: ~27%
repair calls:    ~5%
setup/moderator: small share
```

### Correct behavior

Reduce cost without removing behavioral controls:

```text
smaller participant prompts;
fewer grounding calls;
compact option facts;
deterministic pre-checks before LLM grounding;
skip LLM grounding for short vote lines unless suspicious;
avoid full option-board repetition when only one candidate is being discussed;
reduce repair rate by making prompts simpler and more act-specific;
avoid extra LLM calls for deterministic split summaries.
```

### Validation

`run.json` should show lower `tokens_utterance_in` and `tokens_grounding_in` by call type. Grounding should not remain near one quarter of total input tokens in ordinary runs.

## 8. Priority 7 — Tighten grounding for invented logistical workarounds

### Problem

Sims still sometimes state unsupported logistical details or fixes as if known, for example shelters, quiet corners, parking, route conditions, booking workarounds, or weather reliability.

### Correct behavior

Allowed:

```text
Maybe we could pick a quieter corner, but we do not know if that is possible.
```

Not allowed:

```text
We can pick a quieter corner.
```

Concrete unsupported workarounds should trigger repair or fallback. Hypothetical mitigations are allowed only when uncertainty is explicit.

## 9. Priority 8 — Monitor trait-weighted participation

Trait-weighted participation is much better than before, especially in manual trait-spread cases. Do not prioritize it above narrowing and cost unless a clear bug appears.

Correct behavior remains:

```text
low-engagement sims are quieter but visible;
high-engagement sims are more active but not dominant by accident;
opening and voting phases naturally compress turn-share differences, so discussion phase behavior matters most.
```

## 10. Non-goals for the next round

Do not prioritize broad new features, new traits, new paper integrations, open-domain chat, cosmetic transcript polish, or large architecture rewrites unrelated to the above. The next round should make disagreement handling more socially plausible and reduce cost.
