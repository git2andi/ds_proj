# Topic examples

Suitable topics present a bounded group decision with several plausible public options. Any phrasing works — imperative, question, or noun phrase — as long as one fixed option board can represent the choice:

- Book a flight from Miami to Stockholm for a conference trip;
- Pick a movie for Friday night at home;
- Which espresso machine should the startup buy for its kitchen;
- Decide where to celebrate New Year's Eve this year;
- Choose a storage setup for a shared research dataset;
- Agree on quiet hours for the shared student house.

`eval2/scenarios.txt` contains 102 such topics across many domains (travel, food, entertainment, work, community, sports, family, hobbies) with balanced participant counts 2–7, used by the batch runner `eval2/run_scenarios.py`.

Two constraints:

- Avoid open-ended prompts without enumerable options ("how should we improve morale"), topics requiring unrestricted web knowledge or hidden factual research, and safety-critical professional decisions. The simulator assumes that objective option facts are fully represented by the public board.
- Do not name a group size in the topic ("five friends pick …"): setup fails fast when the stated count contradicts the configured participant count, rather than generating a contradicted world.
