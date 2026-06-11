"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, Phase, RunOutcome, Scenario
from utils import compact_words


def _schema(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _option_cards(options: Iterable[OptionCard]) -> str:
    limit = int(cfg.scenario.option_prompt_max_words)
    return "\n".join(f"- {compact_words(option.prompt_card(), limit)}" for option in options)


def _option_brief(option: OptionCard) -> str:
    # Name + the comparable attributes only (no upside) so the moderator board stays
    # a clean, scannable list of options.
    attrs = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in option.attrs.items())
    return f"{option.name}: {attrs}" if attrs else option.name


# ---------------------------------------------------------------------------
# Setup prompt
# ---------------------------------------------------------------------------


def setup_world(topic: str, n: int, trait_rows: list[dict]) -> str:
    labels = list(cfg.scenario.option_labels)
    schema = {
        "scenario": {
            "decision_kind": "restaurant_choice | travel_destination | hotel_booking | flight_booking | study_plan | presentation_topic | tool_choice | activity_choice | generic_decision",
            "opening_question": "one casual question asking what matters most before choosing",
            "options": [
                {
                    "id": label,
                    "name": "short concrete option name",
                    "attrs": {"cost/time/effort/etc": "stable value", "other_relevant_attribute": "stable value"},
                    "upside": "specific benefit",
                    "tradeoff": "specific downside or cost",
                    "concern": "stable objection, not missing info",
                    "best_for": "the priority this option serves",
                }
                for label in labels
            ],
        },
        "participants": [
            {
                "id": "p1",
                "name": "short first name",
                "role": "1-4 word conversational role",
                "speech_style": "plain style description",
                "private_goal": "what they want from the decision",
                "backstory": "one short prior experience or habit",
                "main_concern": "budget | comfort | fairness | speed | feasibility | novelty | safety | ...",
                "preferred_option": "A",
                "acceptable_options": ["A", "C"],
                "soft_rejections": ["D"],
                "hard_rejections": [],
                "scores": {"A": 5, "B": 3, "C": 4, "D": 2},
                "reasons": {"A": ["grounded reason"], "C": ["grounded reason"]},
                "reservation": "one addressable concern about another option",
                "reconsider_if": "condition based on group priorities, not changed facts",
            }
        ],
    }
    return f"""Create one complete fictional group-decision scenario and participant state.

Topic: {topic}
Participants needed: {n}
Option ids: {labels}
Trait/control profiles to use exactly by id:
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Scenario requirements:
- Create exactly {len(labels)} options.
- Each option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} stable, topic-specific attributes.
- Use concrete stable facts for comparison: cost, time, effort, comfort, risk, flexibility, distance, difficulty, score, or equivalent topic-specific dimensions.
- Every attribute must be a fixed value known now. No placeholders, "unknown"/"TBD", or facts that require a live lookup (availability, current weather, booking status).
- Options must differ meaningfully and expose real trade-offs.
- The opening question must ask about priorities and trade-offs, not ask for votes.

Participant requirements:
- Everyone should try to reach a workable group decision.
- Normal participants need at least {cfg.personas.non_blocker_min_acceptable} acceptable options including their preferred option.
- "scores" rates every option {cfg.scenario.score_min}-{cfg.scenario.score_max} for that person ({cfg.scenario.score_max}=loves it, {cfg.scenario.score_min}=cannot accept). Make scores consistent with the labels: preferred highest, acceptable options {cfg.scenario.acceptance_score} or above, rejected options below {cfg.scenario.acceptance_score}.
- Hard blockers are allowed only when the supplied trait profile says hard_blocker=true.
- Preferences should be diverse enough that the group has something to discuss.
- At least one option should be acceptable to all non-hard-blockers so compromise is possible.
- Reasons must be grounded only in the option cards.
- Reconsider conditions must be about group priorities, not changed facts. Bad: "if the price drops". Good: "if everyone values comfort over price".
- Names should be plausible short first names. Avoid stereotypes and demographic labels.
- Conversation should later sound like friends/classmates deciding together, not a business meeting.

Return JSON only in this shape:
{_schema(schema)}"""


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------


def moderator_opening(scenario: Scenario) -> str:
    lines = [f"Today we're deciding: {scenario.topic}. Here are the options:"]
    for option in scenario.options:
        lines.append(f"Option {option.id} - {_option_brief(option)}")
    lines.append(scenario.opening_question)
    return "\n".join(lines)


def moderator_closure(outcome: RunOutcome, scenario: Scenario) -> str:
    if outcome.final_option:
        option = scenario.option(outcome.final_option)
        if outcome.status == "consensus":
            return f"Alright, then we’ll go with Option {option.id}, {option.name}. Sounds like everyone can live with that."
        if outcome.status == "fallback":
            return f"We did not get full agreement, but the strongest workable choice is Option {option.id}, {option.name}."
    return "Looks like we do not have a clean agreement, so we’ll stop here rather than fake a consensus."


def _lean_name(state: DialogueState, persona: Persona) -> str:
    rt = state.runtimes[persona.id]
    lean = rt.explicit_vote or rt.current_preference or persona.preferred_option
    return state.scenario.option(lean).name if lean in state.scenario.option_ids else "no clear pick"


def moderator_stall_nudge(state: DialogueState) -> str:
    summary = "; ".join(f"{p.name} likes {_lean_name(state, p)}" for p in state.personas)
    return (
        f"We're going round in circles a bit. Where we stand: {summary}. "
        "Is anyone's view actually shifting, or should we start narrowing it down?"
    )


def moderator_holdout_nudge(state: DialogueState, candidate_id: str, holdout_id: str) -> str:
    option = state.scenario.option(candidate_id)
    name = state.name_for(holdout_id)
    return (
        f"{name}, sounds like most of us could live with {option.name}. "
        "What would make it work for you — or is there another one you'd all be fine with?"
    )


# ---------------------------------------------------------------------------
# Turn-generation prompt
# ---------------------------------------------------------------------------


def speaker_card(persona: Persona, focus_ids: Iterable[str]) -> str:
    t = persona.traits
    smin, smax, thr = int(cfg.scenario.score_min), int(cfg.scenario.score_max), int(cfg.scenario.acceptance_score)
    ratings = ", ".join(f"{opt}={persona.score_for(opt)}" for opt in sorted(persona.option_scores))
    # Only the reasons for options in play this turn, to keep the prompt small.
    keep = {persona.preferred_option, *focus_ids}
    reasons = {opt: vals for opt, vals in persona.reasons.items() if opt in keep}
    return f"""{persona.name} ({persona.role})
style: {persona.speech_style}
goal: {persona.private_goal}; priority: {persona.main_concern}
stance: prefers {persona.preferred_option}; can live with {persona.acceptable_options}; concerns {persona.soft_rejections}; hard rejects {persona.hard_rejections}
private ratings (hidden, {smin}-{smax}, you can accept {thr}+): {ratings}
traits: extra={t.extraversion}, agree={t.agreeableness}, direct={t.directness:.2f}, compromise={t.compromise_willingness:.2f}, detail={t.detail:.2f}
reasons: {json.dumps(reasons, ensure_ascii=False)}
reservation: {persona.reservation}; reconsider if: {persona.reconsider_if}"""


def intent_line(intent: MoveIntent, state: DialogueState, addressee_name: Optional[str]) -> str:
    focus = ", ".join(intent.option_focus) if intent.option_focus else "none"
    to_part = f" to {addressee_name}" if addressee_name else ""
    return f"move={intent.act.value}{to_part}; focus={focus}; reason={intent.reason}; length={intent.length_hint}"


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    if intent.act == ActType.OPENING:
        # Vary the opening framing by trait so the first round isn't three identical lines.
        if t.extraversion >= 4:
            return "Open warmly: react to the topic, then say what you care about and which way you lean. Don't lock in."
        if t.directness >= 0.55:
            return "Open bluntly: your top priority and your lean in one line. Don't lock in."
        if t.detail >= 0.6:
            return "Open by pointing at one concrete attribute that matters to you, then your lean. Don't lock in."
        return "Open in your own words: what matters to you and which option you lean to. Don't lock in."
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
        if high_compromise:
            return "Enough discussion. Commit to a workable choice; if a good case was made for an option you rate well enough, move to it instead of restating your first pick."
        return "Enough discussion. State where you stand; only hold out if you are genuinely not convinced."
    if high_compromise:
        return "It's fine to be persuaded: if a point shifts you, say so. Otherwise add a new angle, don't repeat yourself."
    return "Make your case with a concrete trade-off the others haven't covered yet. Don't repeat earlier points."


def sim_utterance(
    *,
    persona: Persona,
    state: DialogueState,
    recent_lines: list[str],
    public_board: str,
    intent: MoveIntent,
    focus_options: list[OptionCard],
    addressee_name: Optional[str],
    max_words: int,
    own_recent: Optional[list[str]] = None,
) -> str:
    option_names = ", ".join(f"{o.id}={o.name}" for o in state.scenario.options)
    focused = _option_cards(focus_options) if focus_options else "- none"
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    already = ""
    if own_recent:
        bullets = "\n".join(f"- {compact_words(line, 16)}" for line in own_recent[-3:])
        already = f"\nYou already said this (do NOT repeat it; say something new or move toward deciding):\n{bullets}\n"
    return f"""Write exactly one natural chat message for the next speaker.

Topic: {state.scenario.topic}
Available options: {option_names}
Speaker:
{speaker_card(persona, intent.option_focus)}

Public state:
{public_board}

Relevant option cards for this move:
{focused}

Recent chat:
{recent}
{already}
Next local move:
{intent_line(intent, state, addressee_name)}
Guidance: {_move_guidance(state, persona, intent)}

Rules:
- Write only {persona.name}'s message: one line, no name prefix, no quotes, under {max_words} words.
- Casual group-chat tone: human, not corporate, not slang-heavy. Reply to the addressed person if one is given.
- Do the local move only; don't re-evaluate every option.
- Add something new: never restate a point already in the chat or copy the previous speaker's phrasing.
- Use only facts from the option cards; invent nothing (no prices, ratings, availability, policies, weather).
- Name options by name or "Option X". If you move to an option you didn't prefer, say so briefly in your own words.

End with a hidden status tag on its own line: [act={intent.act.value}; opt=<{state.scenario.option_ids} or ->; stance=<vote|accept|object|reject|propose|neutral>] (vote=final pick, accept=agree to a compromise, object=mild concern, reject=dealbreaker, propose=offer a compromise, neutral=otherwise)."""


def repair_utterance(
    *,
    original_text: str,
    issue_codes: list[str],
    persona: Persona,
    state: DialogueState,
    recent_lines: list[str],
    public_board: str,
    intent: MoveIntent,
    focus_options: list[OptionCard],
    addressee_name: Optional[str],
    max_words: int,
) -> str:
    issue_text = ", ".join(issue_codes[: int(cfg.utterances.repair_issue_limit)])
    return sim_utterance(
        persona=persona,
        state=state,
        recent_lines=recent_lines,
        public_board=public_board,
        intent=intent,
        focus_options=focus_options,
        addressee_name=addressee_name,
        max_words=max_words,
    ) + f"\n\nRewrite the message because the previous version had these issues: {issue_text}. Previous version: {original_text}"
