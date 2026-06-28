---
name: project-five-trait-routing
description: Routing and behavioral controls now derive only from the five OCEAN traits
metadata:
  node_type: memory
  type: project
---

`TraitProfile` stores only openness, conscientiousness, extraversion, agreeableness, and neuroticism. Response length is derived from extraversion, conscientiousness, and openness. Compromise willingness is derived primarily from agreeableness, with openness and calmness as smaller factors. Normal profiles use agreeableness 3–5; `hard_blocker_probability` can select one profile from configured stubborn five-trait ranges with agreeableness 1 and low openness.

Speaker selection uses extraversion and participation state. The participant who calls for a decision is chosen from extraversion, conscientiousness, openness, and derived compromise willingness. Pacing uses openness/conscientiousness as deliberation. Prompt behavior infers directness and proactivity from combinations of the five traits. No generated initiative, directness, detail, response-length, or compromise fields remain.

The next independent setup redesign is KF23: controller-owned preference overlap, one or a small ordered set of preferred options, at most one rare grounded rejection, no acceptable-option list, no numeric scores, and no forced common compromise.
