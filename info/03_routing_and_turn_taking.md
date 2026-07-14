# Simulator policy and floor arbitration

There is no global participant-act router. On every open-floor opportunity, each eligible `UserSimulator` independently evaluates a small seeded Python policy and returns either silence or one complete `UserAction`.

Candidate actions can come from:

- answering a direct question;
- responding to the active issue;
- raising an unspoken concern;
- supporting the current preference;
- comparing relevant options;
- asking a relevant question;
- acknowledging a recent contribution;
- defending a challenged option;
- accepting or switching to a plausible finalist;
- voting.

The action contains the speaker, willingness, urgency, act, option focus, optional addressee, grounded reason, optional personal context, issue effect, stance update, and vote. The LLM is not involved in bidding.

When a direct question is accepted, the addressed participant has the next response obligation. Its simulator still constructs the answer action. A pending obligation is completed, or explicitly exhausted through the existing bounded generation/repair path, before discussion can transition to narrowing; the required answer is not counted as voluntary engagement.

Without an obligation, `FloorManager` removes silent bids and makes a seeded urgency-weighted selection. It applies only light coordination: a recent-speaker penalty and a maximum of two consecutive participant turns when alternatives exist. It does not use expected shares, deficits, quotas, minimum turns, or controller-selected content. The selected action object is passed onward unchanged.

An empty floor is treated as progression evidence. The first empty round closes or stales an exhausted issue; a second may emit the one available structured moderator stimulus; only a later empty round below the minimum budget may invoke the final liveness mechanism. Liveness still asks a simulator for a policy-generated action, is logged separately, and is excluded from voluntary engagement metrics.

When the moderator emits its single coverage or stall question, the runtime stores a compact `GroupStimulus`. Simulators may voluntarily answer, support, reject, compare, or ignore that stimulus. The moderator never chooses the respondent or response act.

Question bids encode a specific information need: rationale, impact of a concern, acceptability, comparison, or clarification of a recent visible claim. Addressees are selected from public relevance rather than random choice. Compact question keys suppress only an equivalent `(intent, focus, addressee)` question; they do not block later questions about the same option for a different reason.
