# Routing and turn taking

## Simulator-owned actions

Each participant is represented by an independent `UserSimulator`. It constructs a complete `UserAction` rather than generating text directly. Actions may include:

- `OPENING`;
- `SUPPORT`, `REACT`, `OBJECT`, or `COMPARE`;
- `ASK` or `ANSWER`;
- `ACCEPT`;
- `VOTE`.

A bid records the speaker, priority, act, option focus, optional addressee, grounded reason source, and any stance update or vote. Later components may select or reject the bid but do not replace these decisions.

During ordinary discussion, the simulator derives currently valid candidates from its private stance and the visible state. Candidate construction uses public preferences, recent turns, the active thread, and point-use records. Comparisons use the same named public attribute from both options and receive a low selection weight.

After selecting one candidate, engagement determines whether the simulator submits it. Openings, directly required answers, and votes bypass voluntary willingness. An active thread increases voluntary willingness by `0.15`, capped at `1.0`.

## Floor allocation

Bids use `NORMAL`, `THREAD`, or `REQUIRED` priority. The floor keeps only the highest available priority and selects probabilistically among the remaining intact actions. Participants with fewer voluntary turns receive additional weight, while the previous speaker and simple alternating-speaker patterns are penalized. A participant cannot exceed the configured consecutive-turn limit when another bidder is available.

## Threads and response obligations

A direct question creates a response obligation for its named addressee. A group question allows eligible participants to self-select; when all decline, the runtime may force one related simulator-owned response for liveness. Objections may open concern threads. Only one thread is active at a time, and a participant normally contributes to it once. A thread closes at its configured cap, when no related action remains, or when discussion ends.

Structured point keys suppress repeated questions and repeated use of the same grounded reason. After an initial answer, later thread contributions must add another point, compare options, or visibly move.

## Movement

`ACCEPT` may either mark another option as publicly acceptable or switch the current preference. A switch during ordinary discussion requires the target to be recent in the public exchange, to have more public preference support than the participant's current option, and to pass the participant's stubbornness-based movement draw. Hard blockers never move.
