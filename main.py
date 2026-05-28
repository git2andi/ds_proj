"""
main.py
-------
Entry point for the dialogue simulator.

Modes:
  python main.py                   interactive — prompted for a topic
  python main.py scenarios.txt     batch — one topic per non-comment line

Scenario file: one topic per line; lines starting with # are ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make all src/ modules importable without package prefix.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import cfg
from llm_client import get_llm_client
from orchestrator import Orchestrator
from persona import PersonaBuilder
from simulator import Simulator


# ---------------------------------------------------------------------------
# Single dialogue
# ---------------------------------------------------------------------------

def run_dialogue(topic: str) -> None:
    n = cfg.simulation.num_participants

    print(f"\n{'='*60}")
    print(f"Topic    : {topic}")
    print(f"Sims     : {n}")
    print(f"{'='*60}")

    orch = Orchestrator(topic)
    builder = PersonaBuilder(topic=topic)

    name_role_entries = builder.generate_names_and_roles(n)
    personas = builder.build_all(name_role_entries)

    print("\nGenerating belief states...")
    builder.assign_beliefs(personas, orch.options)

    _llm = get_llm_client()
    setup_tokens_in = _llm.session_tokens_in
    setup_tokens_out = _llm.session_tokens_out
    _llm.reset_session()

    print("\nParticipants:")
    for persona in personas:
        primary_tag = " [PRIMARY]" if persona.is_primary else ""
        trait_str = (
            f"open={persona.openness} consc={persona.conscientiousness} "
            f"extra={persona.extraversion} agree={persona.agreeableness} "
            f"neuro={persona.neuroticism} length={persona.response_length}"
        )
        print(f"  {persona.name}{primary_tag} | role: {persona.role} | {trait_str}")
        print(f"    goal: {persona.goal}")
        if persona.beliefs:
            b = persona.beliefs
            accept_str = (
                f", accepts {[x for x in b.acceptable if x != b.preferred]}"
                if len(b.acceptable) > 1 else ""
            )
            print(f"    beliefs: prefers {b.preferred}{accept_str} | {b.key_concern}")
            if b.reasons:
                print(f"      reasons: {' | '.join(b.reasons)}")
            if b.reservation:
                print(f"      reservation: {b.reservation}")
            if b.would_reconsider_if:
                print(f"      would reconsider if: {b.would_reconsider_if}")
    print()

    for persona in personas:
        sim = Simulator(persona=persona, topic=topic, options=orch.options)
        orch.add_sim(sim)

    orch.run_simulation(setup_tokens_in=setup_tokens_in, setup_tokens_out=setup_tokens_out)


# ---------------------------------------------------------------------------
# Batch + interactive
# ---------------------------------------------------------------------------

def run_batch(path: str) -> None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    scenarios = [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not scenarios:
        print(f"No scenarios found in {path}.")
        return

    print(f"Batch mode: {len(scenarios)} dialogue(s) from '{path}'")
    for i, topic in enumerate(scenarios, start=1):
        print(f"\n[{i}/{len(scenarios)}]")
        try:
            run_dialogue(topic)
        except Exception as exc:
            print(f"!! Dialogue failed for '{topic}': {exc}")


def run_interactive() -> None:
    topic = input("Enter the dialogue topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return
    run_dialogue(topic)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_batch(sys.argv[1])
    else:
        run_interactive()
