# Choose a data labeling strategy for a computer vision study

A group of six researchers must select a data labeling strategy for a computer vision study requiring high accuracy and consistency. The options vary in cost, time, and required expertise, and all must support labeling at least 10,000 images.

## Options

- A) In-house Expert Labeling — cost: high; time per 1000 images: 5 days; expertise required: high; quality consistency: high; scalability: medium (+ Ensures high-quality and consistent labels with domain experts directly involved; − Requires significant time and resource commitment from skilled staff)
- B) Crowdsourced Labeling Platform — cost: medium; time per 1000 images: 2 days; expertise required: low; quality consistency: variable; scalability: high (+ Faster labeling with flexible workforce and moderate cost; − Label quality may vary, requiring additional validation steps)
- C) Automated Pre-labeling with Human Verification — cost: medium; time per 1000 images: 3 days; expertise required: medium; quality consistency: medium; scalability: high (+ Reduces human effort by combining automation and manual checks; − Initial automated labels may introduce errors needing correction)
- D) Outsourced Professional Labeling Service — cost: high; time per 1000 images: 4 days; expertise required: low; quality consistency: high; scalability: medium (+ Delivers consistent, expert-level labeling without internal resource strain; − Higher cost and less direct control over labeling process)

## Participants

- **Maya**: preferred B; engagement 5, verbosity 3, directness 4, stubbornness 2
- **Liam**: preferred A; engagement 4, verbosity 2, directness 3, stubbornness 4
- **Nina**: preferred C; engagement 4, verbosity 2, directness 5, stubbornness 2
- **Eli**: preferred D; engagement 2, verbosity 5, directness 5, stubbornness 4
- **Zara**: preferred B; engagement 4, verbosity 2, directness 1, stubbornness 1
- **Omar**: preferred C; engagement 4, verbosity 4, directness 2, stubbornness 3

## Dialogue

**Moderator:** Let’s work through the options for: Choose a data labeling strategy for a computer vision study.
**Liam:** In-house Expert Labeling ensures consistent, high-quality labels from domain experts.
**Omar:** Automated Pre-labeling with Human Verification seems like a good middle ground, combining automation and manual checks to balance quality and cost.
**Zara:** Crowdsourced Labeling feels fast and scalable, with moderate cost.
**Maya:** Crowdsourced Labeling offers fast labeling with moderate cost and good scalability.
**Nina:** Automated Pre-labeling cuts human effort by mixing automation with verification.
**Eli:** Outsourced Professional labeling offers expert-level consistency and reduces internal load, which is crucial for maintaining project quality without disrupting our teams.
**Liam:** Variable quality risks model performance, so extra validation is needed.
**Nina:** Liam’s point stands, but Crowdsourced Labeling scales fast with moderate cost.
**Omar:** Both scale well, but mixing automation with human checks leans better toward balanced quality without extra cost.
**Nina:** The expertise needed is medium, which fits well with reducing manual checks.
**Maya:** Medium consistency means extra verification slows us down and raises costs.
**Maya:** Automated Pre-labeling with Human Verification cuts effort and cost, so I can accept it now.
**Liam:** Quality and consistency are only medium, so errors still impact reliability.
**Omar:** Medium consistency means we should watch for error patterns carefully, or costs might creep up unexpectedly.
**Omar:** Both scale high, but the automated pre-labeling with human checks fits my balance priority better since it can cut some manual effort.
**Liam:** Eli, do you think higher cost and less control with D justify its quality gains?
**Eli:** Higher cost and less control outweigh the quality gains for our scale and disruption tolerance.
**Maya:** Medium cost fits our deadline and budget better than the higher cost of professional labeling.
**Nina:** The expertise required is low, which means less training time.
**Maya:** Does relying on skilled staff for labeling fit within our current resource limits?
**Liam:** It demands too much time from skilled staff to be feasible.
**Eli:** High-quality, consistent labeling requires expert involvement; sacrificing that risks project integrity beyond acceptable limits.
**Zara:** The labeling platform costs medium, expert labeling is high—medium fits my fast, low-cost need better.
**Liam:** The quality consistency is variable, which could risk model reliability.
**Maya:** Quality dips, but since expertise required is low, I can manage quick fixes myself.
**Zara:** Two days per 1000 images feels a bit slow for my quick cycles.
**Zara:** Variable quality means more time fixing, which slows me down.
**Nina:** Automated pre-labeling takes 3 days per 1000 images, faster than 4 days with outsourcing. That speed fits my priority better.
**Liam:** Medium expertise still risks inconsistent labels and affects accuracy.
**Omar:** I see how Option B’s crowdsourced labeling could work here—fast and scalable with moderate cost, which helps balance quality and budget.
**Maya:** Two days per 1000 images is okay, but it still feels tight for our deadlines.
**Zara:** Medium cost feels more manageable than high for quick turns.
**Zara:** Automation with Human Verification might reduce manual work and speed us up.
**Omar:** Three days per 1000 images isn’t bad, but we should watch that it doesn’t slow down other parts of the pipeline too much.
**Zara:** That cost might be okay, but faster labeling would help me more.
**Moderator:** The leading options remain tied. Let’s use Automated Pre-labeling as the bounded tie-break target. Maya, Liam, Eli, and Zara, could it meet your main requirements?
**Maya:** Automated Pre-labeling with Human Verification cuts effort and cost by mixing automation and checks.
**Zara:** Automated Pre-labeling with Human Verification could work since it speeds things up and lowers manual effort, which fits my need to iterate fast.
**Moderator:** Let’s confirm the final choices.
**Maya:** I choose Automated Pre-labeling.
**Liam:** My final vote is In-house Expert.
**Nina:** My final vote is Automated Pre-labeling.
**Eli:** I’m voting for Outsourced Professional.
**Zara:** I choose Automated Pre-labeling.
**Omar:** Automated Pre-labeling is my final choice.
**Moderator:** The majority chose Automated Pre-labeling.

## Outcome

- Status: majority
- Final option: C
- Votes: {'p1': 'C', 'p2': 'A', 'p3': 'C', 'p4': 'D', 'p5': 'C', 'p6': 'C'}
- Reason: 4 of 6 participants selected the option

## Core metrics

- participant_count: 6
- participant_turns: 43
- voluntary_turns: 30
- moderator_turns: 4
- moderator_ratio: 0.0851
- avg_words_per_participant_turn: 12.84
- visible_preference_changes: 2
- repair_turns: 0
- dropped_turns: 0
- fallback_turns: 0
- response_failures: 0
- protocol_errors: 0
- vote_outcome_consistent: True
- input_tokens: 22771
- output_tokens: 2937
- llm_calls: 40
- voluntary_turns_by_persona: {'p2': 6, 'p3': 4, 'p6': 5, 'p1': 7, 'p4': 1, 'p5': 7}