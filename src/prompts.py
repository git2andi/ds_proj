"""LLM-facing prompts for setup, moderator text, and utterance realization.

The environment controls facts, routing, addressees, and dialogue acts. The LLM
only realizes one natural message at a time. Generated messages must not contain
hidden metadata; public outcomes are parsed from visible text only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from aliases import short_alias_map
from config_loader import cfg
from models import DialogueState, MoveIntent, OptionCard, Persona, RunOutcome, Scenario
from utils import compact_words


def _schema(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _option_brief(option: OptionCard) -> str:
    return option.public_line(
        max_attrs=int(cfg.scenario.option_board_attr_max),
        note_words=int(cfg.scenario.option_board_note_max_words),
    )


def _option_cards(options: Iterable[OptionCard]) -> str:
    limit = int(cfg.scenario.option_prompt_max_words)
    return "\n".join(f"- {compact_words(option.prompt_card(), limit)}" for option in options)


def setup_scenario(topic: str, n: int) -> str:
    labels = list(cfg.scenario.option_labels)
    schema = {
        "scenario": {
            "decision_kind": "restaurant_choice | travel_destination | tool_choice | activity_choice | schedule_choice | purchase_choice | generic_decision",
            "opening_question": "one casual question about priorities and trade-offs",
            "shared_context": [
                "2-3 facts everyone knows before the discussion: timing, budget, group size, goal, or constraints"
            ],
            "options": [
                {
                    "id": label,
                    "name": "specific realistic option name",
                    "short_name": "1-2 recognizable words copied from name",
                    "attrs": {
                        "cost/time/effort/etc": "stable value",
                        "other_relevant_attribute": "stable value",
                    },
                    "upside": "specific benefit",
                    "tradeoff": "specific downside or cost",
                    "concern": "stable objection people could raise",
                    "best_for": "priority this option serves",
                }
                for label in labels
            ],
        }
    }
    return f"""Create one fictional but realistic option-grounded group-decision setup.

Topic: {topic}
Decision group: exactly {n} participants.
Option ids: {labels}

Rules:
- Create exactly {len(labels)} options.
- Each option name must be specific, realistic, and not a generic category.
- Keep option names concise, ideally 3-10 words.
- Option names must be complete noun phrases; do not end with words like "with", "in", "and", "on", "departing", "including", or a comma.
- Every option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} concrete attributes with stable values.
- Do not use unknown, TBD, live availability, current weather, or facts that need internet lookup.
- Options should expose real trade-offs, not one obvious winner.
- shared_context must be general context known to everyone, not hidden persona facts.
- If shared_context mentions the group size, it must say exactly {n}.

Return JSON only:
{_schema(schema)}"""


def setup_personas(
    topic: str,
    n: int,
    trait_rows: list[dict],
    required_preferences: dict[str, str],
    options_json: list[dict],
) -> str:
    names_by_id = {row["id"]: row.get("name", row["id"]) for row in trait_rows}
    preference_lines = "\n".join(
        f"- {pid} ({names_by_id.get(pid, pid)}): {option_id}"
        for pid, option_id in sorted(required_preferences.items(), key=lambda item: int(item[0][1:]))
    )
    schema = {
        "participants": [
            {
                "id": "p1",
                "name": "exact name from trait row",
                "background": "one sentence explaining the person's angle on this decision",
                "private_goal": "what they personally want from the decision",
                "preferred_options": ["A"],
                "rejection": None,
                "rejection_reason": "",
            }
        ]
    }
    return f"""Create {n} simulated users for an option-grounded group decision.

Topic: {topic}
Options:
{json.dumps(options_json, ensure_ascii=False, indent=2)}

Use these trait rows exactly. Traits are 1-5 OCEAN scores.
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Initial primary preference assignment. preferred_options[0] MUST match this exactly:
{preference_lines}

Rules:
- Use the exact id and name from each trait row.
- preferred_options is the person's initial private preference, not a final vote. Add at most one secondary acceptable option if it fits.
- A participant with agreeableness=1 must have exactly one preferred option (no secondary).
- Participants want a workable group decision. High openness/agreeableness means easier compromise; low agreeableness means more resistance.
- For agreeableness=1 only, you may set one grounded rejection if an option conflicts with their background/goal. That rejection is a hard blocker.
- For all other participants, rejection must be null.
- background and private_goal must be one sentence each, specific to this topic, and grounded in the option cards/shared context.

Return JSON only:
{_schema(schema)}"""


def moderator_opening(scenario: Scenario) -> str:
    lines = [
        f"Today we're deciding: {scenario.topic}.",
        "For this simulated decision, I’ll treat the following setup as the shared facts.",
        "Options:",
    ]
    for option in scenario.options:
        lines.append(_option_brief(option))
    context_items = list(scenario.shared_context)[: int(cfg.scenario.shared_context_max_items)]
    if context_items:
        context = "; ".join(
            compact_words(item, int(cfg.scenario.shared_context_max_words))
            for item in context_items
        )
        lines.append("Context: " + context.rstrip(".") + ".")
    lines.append(compact_words(scenario.opening_question, 20))
    return "\n".join(lines)


def moderator_nudge_prompt(
    state: DialogueState,
    reason: str,
    candidate_name: str | None,
    *,
    target_name: str | None = None,
    requested_action: str | None = None,
    focus_options: list[str] | None = None,
) -> str:
    recent = _recent_chat(state, limit=5)
    candidate = candidate_name or "no single option yet"
    focus = _option_names(state, focus_options or []) or "not specified"
    target = target_name or "the group"
    action = requested_action or "move the decision forward with one concrete next step"
    public = _public_state_summary(state)
    return f"""You are the neutral moderator of a casual group decision chat.

Use MUCA-style control: decide what to ask, when to intervene, and who to address.
Write one short progress nudge, under {cfg.utterances.word_budgets.moderator} words.
Do not vote, do not decide, do not add facts, and do not repeat the option board.
Reason to intervene: {reason}
Address this target if useful: {target}
Requested action: {action}
Current likely common ground: {candidate}
Focus options: {focus}
Visible state: {public}

Recent chat:
{recent}

No speaker prefix. One sentence only."""


def moderator_closure_prompt(outcome: RunOutcome, scenario: Scenario, state: DialogueState) -> str:
    final = scenario.option(outcome.final_option).name if outcome.final_option else "no option"
    return f"""You are the neutral moderator closing a casual group decision chat.

Outcome status: {outcome.status}
Final option: {final}
Reason: {outcome.reason}

Write one short closing line under {cfg.utterances.word_budgets.closure} words.
Do not add new reasons or facts. No farewell. No speaker prefix."""


def sim_utterance(
    *,
    persona: Persona,
    state: DialogueState,
    intent: MoveIntent,
    recent_lines: list[str],
    focus_options: list[OptionCard],
    addressee_name: str | None,
    max_words: int,
) -> str:
    aliases = short_alias_map(state.scenario.options)
    cards = _option_cards(focus_options or state.scenario.options)
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    current = state.runtimes[persona.id].current_preference or persona.preferred_option
    current_name = aliases.get(current, state.scenario.option(current).name if current in state.scenario.option_ids else "undecided")
    initial_name = aliases.get(persona.preferred_option, persona.preferred_option)
    blocked = ""
    if persona.rejection:
        blocked = f"\nHard blocker: they strongly reject {aliases.get(persona.rejection, persona.rejection)} because {persona.rejection_reason}. Do not accept or vote for that option."
    target = _target_line(state, intent)
    target_block = f"\nRespond to this recent point: {target}" if target else ""
    address = f"\nAddress {addressee_name} if it sounds natural." if addressee_name else ""
    context = "; ".join(compact_words(item, 14) for item in state.scenario.shared_context) if state.scenario.shared_context else "none"
    params = persona.sim_params
    voice = _voice_guidance(persona)
    decision_instruction = ""
    if intent.act.value in {"vote", "accept"}:
        decision_instruction = "\nFor this decision turn, visibly commit with an unambiguous final vote, preferably 'I vote for ...'. Do not add a new question after the vote."
    elif intent.act.value == "reject":
        decision_instruction = "\nFor this decision turn, visibly reject the blocked option and name the acceptable alternative if there is one."
    agenda = ""
    if intent.agenda_index is not None and 0 <= intent.agenda_index < len(persona.agenda):
        item = persona.agenda[intent.agenda_index]
        agenda = f"\nPending simulator agenda item: {item.act.value} about {item.option or 'the decision'} — {item.reason}"
    style_notes = ""
    if intent.suppress_name_prefix:
        style_notes += "\n- Recent turns over-used names; do NOT open with another participant's name, just reply."
    if intent.avoid_pattern in {"concede_but", "worry_but", "tradeoff_but"}:
        style_notes += "\n- Avoid the 'fair point, but…' / 'X is good but I worry…' concession-objection shape used just now; make a different move (a plain claim, a direct question, a concrete comparison, or a firm stance)."

    return f"""Write {persona.name}'s next message in a natural group decision chat.

Environment: option-grounded multi-user decision simulation.
Topic: {state.scenario.topic}
Shared context: {context}
Available options: {', '.join(f'{o.id}={aliases[o.id]}' for o in state.scenario.options)}

Simulated user:
- background: {persona.background}
- private goal: {persona.private_goal}
- OCEAN 1-5: openness={persona.traits.openness}, conscientiousness={persona.traits.conscientiousness}, extraversion={persona.traits.extraversion}, agreeableness={persona.traits.agreeableness}, neuroticism={persona.traits.neuroticism}
- simulator parameters 0-1: engagement={params.engagement:.2f}, verbosity={params.verbosity:.2f}, initiative={params.initiative:.2f}, responsiveness={params.responsiveness:.2f}, stubbornness={params.stubbornness:.2f}, directness={params.directness:.2f}, compromise_threshold={params.compromise_threshold:.2f}
- voice guidance: {voice}
- initial preference: {initial_name}; current internal lean: {current_name}{blocked}

Move to render: {intent.act.value}
Purpose: {intent.reason}{agenda}{target_block}{address}{decision_instruction}

Use only these option facts:
{cards}

Recent chat:
{recent}

Style:
- One message only, no name prefix, no quotes, no bullet list.
- Aim for {8 if max_words > 16 else 5}-{max_words} words. Full casual sentence, not a meeting speech.
- Follow the voice guidance; make this person sound distinct from the others without becoming a caricature.
- Vary sentence shape. Do not keep starting with "I'm leaning", "feels", or the option name.
- Use names, "you", "we", "us", short option names, or no option name when that fits.
- Add one new point, concern, answer, or stance shift. Avoid repeating the same reason from recent chat.
- Ask a question only when the move is ask or invite; otherwise usually make a statement.
- Never invent facts outside the option cards/shared context.
- Do not append metadata, tags, JSON, or bracketed labels.{style_notes}"""


def repair_utterance(
    *,
    original_text: str,
    issue_codes: list[str],
    persona: Persona,
    state: DialogueState,
    recent_lines: list[str],
    intent: MoveIntent,
    max_words: int,
) -> str:
    cards = _option_cards(state.scenario.options)
    recent = "\n".join(recent_lines[-3:]) if recent_lines else "(no recent turns)"
    clear_commit = ""
    if intent.act.value in {"vote", "accept"}:
        clear_commit = " Include exactly one clear final vote to one option, e.g. 'I vote for Option B'. Do not ask a question after it."
    required_focus = ""
    if intent.option_focus and "MISSING_REQUIRED_OPTION_FOCUS" in issue_codes:
        required_focus = f" Mention and discuss Option {intent.option_focus[0]} explicitly."
    grounding = ""
    if "UNSUPPORTED_FACT" in issue_codes:
        grounding = " The line invented a fact not in the option cards/context; remove any invented service, fee, policy, location, time, or number and keep only what the cards state (uncertainty like 'we don't know if…' is fine)."
    return f"""Repair this generated chat line.

Speaker: {persona.name}
Original line: {original_text}
Problems: {', '.join(issue_codes)}
Allowed option facts:
{cards}
Recent chat:
{recent}

Write one natural chat line under {max_words} words. No speaker prefix. Do not invent facts. Avoid generic filler.{clear_commit}{required_focus}{grounding} Do not append metadata, tags, JSON, or bracketed labels."""


def grounding_check(*, utterance: str, state: DialogueState, focus_options: list[OptionCard]) -> str:
    """Prompt a strict fact-checker: does the line invent facts beyond the board?"""
    cards = _option_cards(focus_options or state.scenario.options)
    context = "; ".join(compact_words(item, 14) for item in state.scenario.shared_context) or "none"
    return f"""You are a strict fact-checker for a simulated group decision.
The ONLY facts that exist in this world are in the option cards and shared context.

Option cards:
{cards}
Shared context: {context}

Message to check:
"{utterance}"

A message is UNSUPPORTED only if it states a NEW concrete fact that is not in, and
not directly implied by, the cards/context: e.g. an invented service, included or
excluded feature, fee, policy, location, exact time/number, or operational detail.
Opinions, priorities, trade-off reasoning, questions, and uncertainty are ALWAYS allowed.
Reasoning that follows from a listed attribute is allowed.

Reply with JSON only: {{"unsupported": true or false, "snippet": "the offending phrase, or empty"}}"""


def _option_names(state: DialogueState, ids: list[str]) -> str:
    aliases = short_alias_map(state.scenario.options)
    names = [aliases.get(option_id, option_id) for option_id in ids if option_id in state.scenario.option_ids]
    return ", ".join(names)


def _public_state_summary(state: DialogueState) -> str:
    votes = []
    for persona in state.personas:
        vote = state.runtimes[persona.id].explicit_vote
        if vote:
            votes.append(f"{persona.name}->{vote}")
    untouched = [oid for oid, cov in state.coverage.items() if cov.mentions == 0]
    open_q = [f"{state.name_for(q.target_id)} owes answer" for q in state.open_questions[-2:]]
    parts = []
    parts.append("votes: " + (", ".join(votes) if votes else "none"))
    if untouched:
        parts.append("untouched options: " + ", ".join(untouched))
    if open_q:
        parts.append("open questions: " + ", ".join(open_q))
    return "; ".join(parts)


def _voice_guidance(persona: Persona) -> str:
    p = persona.sim_params
    parts: list[str] = []
    if p.directness >= 0.70:
        parts.append("direct, concrete, low hedging")
    elif p.directness <= 0.35:
        parts.append("careful, softer wording")
    if p.stubbornness >= 0.70:
        parts.append("pushes their concern instead of agreeing too quickly")
    elif p.compromise_threshold <= 0.35:
        parts.append("actively looks for workable compromise")
    if p.engagement >= 0.70:
        parts.append("proactive and opinionated")
    elif p.engagement <= 0.35:
        parts.append("brief, selective, only speaks when useful")
    if p.responsiveness >= 0.70:
        parts.append("reacts to the previous speaker by name when natural")
    if p.verbosity <= 0.35:
        parts.append("short and plain")
    elif p.verbosity >= 0.70:
        parts.append("slightly more explanatory")
    return "; ".join(parts) if parts else "balanced, natural, no assistant-like phrasing"


def _recent_chat(state: DialogueState, limit: int) -> str:
    rows = [f"{t.speaker_name}: {t.text}" for t in state.turns[-limit:]]
    return "\n".join(rows) if rows else "(none yet)"


def _target_line(state: DialogueState, intent: MoveIntent) -> str:
    if intent.respond_to_turn is None:
        return ""
    for turn in state.turns:
        if turn.index == intent.respond_to_turn:
            text = compact_words(turn.text, int(cfg.utterances.response_target_max_words))
            return f"{turn.speaker_name}: {text}"
    return ""
