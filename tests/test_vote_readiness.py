"""I5: early vote readiness and vote candidate come from visible evidence,
never from latent preference concentration. No LLM calls."""

from __future__ import annotations

from dialogue import DialogueRunner, initialise_state
from models import (
    ActType,
    DialogueAct,
    OpenQuestion,
    OptionCard,
    Persona,
    Phase,
    Scenario,
    SimulatorParameters,
    TraitProfile,
    TurnRecord,
)
from parsing import OptionResolver


def _persona(pid: str, pref: str) -> Persona:
    return Persona(
        id=pid,
        name=pid.upper(),
        traits=TraitProfile(3, 3, 3, 3, 3),
        sim_params=SimulatorParameters(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        background="b",
        private_goal="g",
        preferred_options=[pref],
    )


def _state(n_turns: int = 12, prefs=("A", "B", "C")):
    options = [OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]]
    scenario = Scenario(topic="t", decision_kind="generic_decision", opening_question="q", options=options)
    personas = [_persona(f"p{i+1}", p) for i, p in enumerate(prefs)]
    state = initialise_state(scenario, personas)
    state.min_discussion_turns = 6
    state.force_narrow_turns = 40
    state.hard_max_turns = 60
    for coverage in state.coverage.values():
        coverage.mentions = 1  # option coverage satisfied; not under test here
    for i in range(n_turns):
        pid = f"p{i % len(personas) + 1}"
        state.turn_index += 1
        state.turns.append(TurnRecord(
            index=state.turn_index,
            speaker_id=pid,
            speaker_name=pid.upper(),
            text="…",
            phase=Phase.DISCUSSION,
            act=DialogueAct(speaker_id=pid, text="…", act_type=ActType.BUILD),
        ))
    return state


def _runner() -> DialogueRunner:
    runner = DialogueRunner.__new__(DialogueRunner)
    runner._resolver = OptionResolver([OptionCard(id=x, name=f"Option {x} Name") for x in ["A", "B", "C", "D"]])
    return runner


def test_latent_convergence_alone_is_not_ready():
    state = _state()
    for rt in state.runtimes.values():
        rt.current_preference = "A"  # everyone latently converged
    assert _runner()._ready_for_vote(state) is False


def test_visible_support_cluster_is_ready():
    state = _state()
    state.runtimes["p1"].explicit_vote = "A"
    state.runtimes["p2"].accepted_options.add("A")
    assert _runner()._ready_for_vote(state) is True


def test_open_question_about_candidate_delays_vote():
    state = _state()
    state.runtimes["p1"].explicit_vote = "A"
    state.runtimes["p2"].accepted_options.add("A")
    state.open_questions.append(OpenQuestion(turn_id=1, asked_by="p3", target_id="p1", text="?", option_focus=["A"]))
    assert _runner()._ready_for_vote(state) is False


def test_active_blocker_on_candidate_delays_vote():
    state = _state()
    state.runtimes["p1"].explicit_vote = "A"
    state.runtimes["p2"].accepted_options.add("A")
    state.runtimes["p3"].hard_rejections["A"] = "dealbreaker"
    assert _runner()._ready_for_vote(state) is False


def test_force_narrow_still_triggers_without_visible_evidence():
    state = _state(n_turns=40)
    assert _runner()._ready_for_vote(state) is True


def test_candidate_prefers_visible_support_over_latent():
    state = _state()
    for rt in state.runtimes.values():
        rt.current_preference = "A"
    state.runtimes["p2"].explicit_vote = "C"
    assert _runner()._candidate_for_vote(state) == "C"


def test_candidate_tie_broken_by_latent():
    state = _state()
    for rt in state.runtimes.values():
        rt.current_preference = "D"
    state.runtimes["p1"].explicit_vote = "C"
    state.runtimes["p2"].explicit_vote = "D"
    assert _runner()._candidate_for_vote(state) == "D"


def test_candidate_falls_back_to_latent_without_evidence():
    state = _state()
    for rt in state.runtimes.values():
        rt.current_preference = "B"
    assert _runner()._candidate_for_vote(state) == "B"
