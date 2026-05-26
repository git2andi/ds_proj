# state/ — structured dialogue state (Stage 5+)
# Populated in parallel with the legacy string-based logic; drives decisions from Stage 6+.
from state.turn import DialogueAct, StanceUpdate, TurnRecord
from state.discourse import DiscourseGraph
from state.stance import OptionState, StanceTable
from state.participant import ParticipantState
from state.dialogue_state import PhaseEvidence, StructuredState

__all__ = [
    "DialogueAct", "StanceUpdate", "TurnRecord",
    "DiscourseGraph",
    "OptionState", "StanceTable",
    "ParticipantState",
    "PhaseEvidence", "StructuredState",
]
