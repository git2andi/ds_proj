# Proposed Dialogue Realism Upgrades

## Upgrade 1: Add a Scenario Context Block

### Problem

The current simulator gives participants options with attributes, upsides, and trade-offs, but the group often lacks shared situational context. As a result, the dialogue can become a pure preference debate: one speaker prefers cost, another prefers novelty, another prefers feasibility. This produces coherent discussion, but it can still feel artificial because real groups usually rely on shared background assumptions, social context, and practical unknowns.

For example, in a birthday gift discussion, people would naturally ask whether the recipient already owns something, whether the gift needs to be ready by a certain date, whether the budget is shared by the group, or whether the recipient would need to schedule an activity. Without this kind of context, the speakers either stay abstract or risk hallucinating missing facts.

### Proposal

Extend the scenario setup with a small `context` block. This block should contain stable, non-option-specific information about the decision situation.

Example fields:

```json
"context": {
  "decision_scope": "group gift from three friends for Steve's birthday party",
  "known_constraints": [
    "the gift should be ready by Saturday",
    "the group wants to stay around $100 if possible"
  ],
  "group_facts": [
    "the group knows Steve likes gaming",
    "the group wants the gift to feel personal rather than purely practical"
  ],
  "unknowns_allowed_as_next_steps": [
    "whether Steve already owns a console",
    "whether Steve would enjoy a scheduled activity",
    "whether Steve is available for an experience on a specific date"
  ],
  "do_not_invent": [
    "Steve's exact availability",
    "current shop stock",
    "live booking availability",
    "unstated private preferences"
  ]
}
```

### Expected Benefit

This gives the dialogue more social grounding without encouraging hallucination. Participants can mention unknowns as unknowns instead of inventing answers.

For example:

* “Before buying the console, we should check whether Steve already has one.”
* “Let’s not assume he is free that day.”
* “A voucher would avoid the availability problem because he can choose the date himself.”
* “If the budget is around $100, the console is probably too much unless everyone contributes more.”

This should make the dialogue feel more like a real group decision and less like isolated option comparison.

### Implementation Direction

The scenario generator should create the context block once during setup. The dialogue generator can then use it as background information. The context should stay compact and only include stable facts or explicitly marked unknowns. Participants may reason about unknowns, but they must not resolve them unless the information is provided.

The controller can also use the context block to route practical questions and conditional agreement more naturally.

---

## Upgrade 2: Add Practical Constraint Acts to Routing

### Problem

The current dialogue already supports preference expression, objections, comparisons, and concessions. However, many conversations still remain too abstract. Speakers debate values such as cost, novelty, comfort, memorability, or quality, but they do not consistently test whether an option is practically workable.

In real decisions, people usually move from preference talk into constraint checking. They ask about budget, timing, availability, difficulty, group size, workload, access, reservations, preparation effort, or who will take responsibility. Without these practical checks, agreement can feel too easy or forced.

### Proposal

Add explicit practical constraint acts to the router. These acts should occasionally force the conversation to test an option against concrete feasibility questions.

Possible act types:

```python
PRACTICAL_CHECK_BUDGET
PRACTICAL_CHECK_TIME
PRACTICAL_CHECK_AVAILABILITY
PRACTICAL_CHECK_WORKLOAD
PRACTICAL_CHECK_GROUP_FIT
PRACTICAL_CHECK_ACCESS
PRACTICAL_CHECK_DIFFICULTY
PRACTICAL_CHECK_RESPONSIBILITY
```

These acts should not invent new facts. They should either use known scenario context, use option attributes, or identify something that must be checked later.

### Example Behavior

Instead of only saying:

> “The experience gift is memorable, so I think it is worth it.”

A participant could say:

> “I like the experience idea, but who would organize it?”

Or:

> “If it is a flexible voucher, I can accept it. If we have to pick a fixed date now, that feels risky.”

Instead of closing a route decision immediately:

> “Haute Route it is.”

The dialogue could produce:

> “Haute Route works for me if we check hut availability first and plan shorter stages.”

Instead of accepting a seminar topic abruptly:

> “Let’s plan Synthetic Biology and Biofuels next.”

A more realistic turn would be:

> “I can work with the biofuels topic if we narrow it to one case study and split the technical background.”

### Expected Benefit

This should make concessions feel earned. Participants would not simply abandon their original preference. They would accept an option under a condition, after a practical concern was addressed, or with a clear next step.

This also helps distinguish different outcome types:

* full consensus,
* conditional consensus,
* fallback decision,
* unresolved decision.

For example:

```text
Status: conditional_consensus
Final option: Experience Gift
Condition: choose a flexible voucher instead of a fixed-date activity
Next step: check suitable vouchers under $100
```

This is more realistic than treating every agreement as full consensus.

### Implementation Direction

The router should insert practical constraint acts when:

1. an option is becoming the leading candidate,
2. a participant has an unresolved concern,
3. a speaker is about to accept a non-preferred option,
4. the moderator is preparing to close the discussion,
5. the conversation has repeated preference arguments without progress.

Before final closure, the controller should check whether the main objections have been handled by one of:

* a known fact,
* a practical condition,
* an explicit tolerance statement,
* a next step.

If not, the dialogue should not close as full consensus. It should either ask one more practical question or end as conditional consensus / unresolved.

### Summary

The goal is not to add more random detail. The goal is to make the dialogue move from preference comparison toward realistic decision-making. A small context block gives the group shared social grounding. Practical constraint acts make the discussion test whether a preferred option can actually work.
