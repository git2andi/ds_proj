"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, Phase, RunOutcome, Scenario
from scoring import best_overlap_option, current_lean, leading_option
from utils import compact_words
from validation import classify_discourse_frames, covered_slots_hint, recent_frame_hint


def _schema(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _option_cards(options: Iterable[OptionCard]) -> str:
    limit = int(cfg.scenario.option_prompt_max_words)
    return "\n".join(f"- {compact_words(option.prompt_card(), limit)}" for option in options)


def _option_brief(option: OptionCard) -> str:
    attrs = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in option.attrs.items())
    return f"{option.name}: {attrs}" if attrs else option.name


def _option_name_only(option: OptionCard) -> str:
    return option.name


# ---------------------------------------------------------------------------
# Setup prompt
# ---------------------------------------------------------------------------


def setup_scenario(topic: str) -> str:
    labels = list(cfg.scenario.option_labels)
    schema = {
        "scenario": {
            "decision_kind": "restaurant_choice | travel_destination | hotel_booking | flight_booking | study_plan | presentation_topic | tool_choice | activity_choice | generic_decision",
            "opening_question": "one casual question asking what matters most before choosing",
            "shared_context": ["2-3 stable background facts about the decision situation that all participants would know (budget, group size, timing, who it's for, key constraints)"],
            "options": [
                {
                    "id": label,
                    "name": "specific realistic name, not a generic category",
                    "short_name": "1-2 word casual nickname friends would use; must stand alone (never end with 'and', 'of', 'to', 'the', 'a', 'an')",
                    "attrs": {"cost/time/effort/etc": "stable value", "other_relevant_attribute": "stable value"},
                    "upside": "specific benefit",
                    "tradeoff": "specific downside or cost",
                    "concern": "stable objection, not missing info",
                    "best_for": "the priority this option serves",
                }
                for label in labels
            ],
        },
    }
    return f"""Create a fictional group-decision scenario.

Topic: {topic}
Option ids: {labels}

Requirements:
- Create exactly {len(labels)} options.
- Option names must be specific and realistic, not generic categories.
- Each option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} stable, topic-specific attributes with concrete values people can compare and discuss.
- Every attribute must be a fixed value known now. No placeholders, "unknown"/"TBD", or facts that require a live lookup (availability, current weather, booking status).
- Options must differ meaningfully and expose real trade-offs.
- The opening question must ask about priorities and trade-offs, not ask for votes.
- shared_context: 2-3 stable facts about the decision situation (not about specific options). These are things all participants would know going in.

Return JSON only in this shape:
{_schema(schema)}"""


def setup_personas(topic: str, n: int, trait_rows: list[dict], pref_groups: list[list[str]], options_json: list[dict]) -> str:
    labels = list(cfg.scenario.option_labels)
    group_lines = "\n".join(f"- group {i + 1}: {', '.join(g)}" for i, g in enumerate(pref_groups))
    schema = {
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
    return f"""Create {n} participants for this group decision.

Topic: {topic}
Options (already decided):
{json.dumps(options_json, ensure_ascii=False, indent=2)}

Trait/control profiles to use exactly by id:
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Requirements:
- Everyone should try to reach a workable group decision.
- Normal participants need at least {cfg.personas.non_blocker_min_acceptable} acceptable options including their preferred option.
- "scores" rates every option {cfg.scenario.score_min}-{cfg.scenario.score_max} for that person ({cfg.scenario.score_max}=loves it, {cfg.scenario.score_min}=cannot accept). Make scores consistent with the labels: preferred highest, acceptable options {cfg.scenario.acceptance_score} or above, rejected options below {cfg.scenario.acceptance_score}.
- Hard blockers are allowed only when the supplied trait profile says hard_blocker=true.
- Preferred-option groups (participants in the SAME group share ONE preferred option; different groups must prefer DIFFERENT options):
{group_lines}
- Even when participants share a preferred option, give them distinct roles, reasons, and concerns so they don't sound identical.
- At least one option should be acceptable to all non-hard-blockers so compromise is possible.
- Reasons must be grounded only in the option cards above.
- Reconsider conditions must be about group priorities, not changed facts. Bad: "if the price drops". Good: "if everyone values comfort over price".
- Use the exact name given in each trait profile row. Do not change or substitute names.
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
    if scenario.shared_context:
        lines.append("Some things we know: " + "; ".join(scenario.shared_context) + ".")
    lines.append(scenario.opening_question)
    return "\n".join(lines)


# Moderator facilitation lines are written by the LLM (so they vary run to run) from the
# situation data below. Only the opening option board (moderator_opening) stays fixed.
_MODERATOR_VOICE = (
    "You are a relaxed, neutral facilitator in a casual group chat (friends/colleagues), "
    "not a corporate host. Write ONE short spoken line"
)
_MODERATOR_RULES = (
    "Use only the facts and names given — don't invent names, options, votes, or reasons. "
    "No quotes, no name prefix, no lists, no emoji."
)


def _camp_split(state: DialogueState) -> str:
    """A compact tally of how the room is divided right now, e.g.
    '3 for City Break, 3 for Mountain Resort, 1 undecided' — so the moderator can name the
    actual split instead of a vague 'we're going in circles'."""
    from collections import Counter
    counts = Counter(_lean_name(state, p) for p in state.personas)
    parts = [f"{n} for {name}" if name != "no clear pick" else f"{n} undecided" for name, n in counts.most_common()]
    return ", ".join(parts)


def _conflict_dimension(state: DialogueState) -> str:
    """Name the key concern driving the split (e.g. 'cost vs comfort') so the moderator
    can diagnose the real trade-off instead of a generic 'we're going in circles'."""
    from collections import Counter
    concerns = Counter(p.main_concern for p in state.personas)
    top = [c for c, _ in concerns.most_common(2)]
    if len(top) >= 2:
        return f"The real tension is {top[0]} vs {top[1]}."
    return ""


def moderator_stall_prompt(state: DialogueState) -> str:
    overlap = best_overlap_option(state)
    conflict = _conflict_dimension(state)
    split = _camp_split(state)
    if overlap is not None:
        push = (f"Point out that {state.scenario.option(overlap).name} may be the common ground — "
                "people on different sides can live with it — and ask if everyone can get behind it, "
                "or ask a specific holdout what would make it work.")
    elif _has_clear_holdout(state):
        holdout_name, concern = _holdout_info(state)
        push = (f"Ask {holdout_name} directly what would need to change about the leading option "
                f"to make it work for them (their concern is {concern}). Don't propose a solution yet.")
    else:
        push = ("The group is genuinely split. Ask one specific question to surface the real blocker: "
                "what's the one thing that would change someone's mind? Don't propose a compromise yet.")
    names = ", ".join(p.name for p in state.personas)
    return f"""{_MODERATOR_VOICE} (max 32 words) for this moment.
Topic: {state.scenario.topic}. Participants: {names}.
The split: {split}. {conflict}
Name the actual disagreement, then move things forward. {push} Don't just ask 'is anyone's view changing?' again. {_MODERATOR_RULES}"""


def _has_clear_holdout(state: DialogueState) -> bool:
    lead = leading_option(state)
    if not lead:
        return False
    holdouts = [p for p in state.personas
                if state.runtimes[p.id].current_preference != lead
                and state.runtimes[p.id].explicit_vote != lead
                and lead not in state.runtimes[p.id].accepted_options]
    return 1 <= len(holdouts) <= 2


def _holdout_info(state: DialogueState) -> tuple[str, str]:
    lead = leading_option(state)
    for p in state.personas:
        rt = state.runtimes[p.id]
        if rt.current_preference != lead and rt.explicit_vote != lead and lead not in rt.accepted_options:
            return p.name, p.main_concern
    return "someone", "their main concern"


def moderator_agreement_prompt(state: DialogueState, candidate_id: str) -> str:
    option = state.scenario.option(candidate_id)
    # "everyone"/"the group" reads oddly for two people; address the pair directly instead.
    whole = "you both" if len(state.personas) == 2 else "the group"
    return f"""{_MODERATOR_VOICE} (max 28 words) for this moment.
Topic: {state.scenario.topic}
It looks like {whole} are converging on {option.name}.
Note that {whole} seem agreed on {option.name} and ask if anyone objects or if you should lock it in. {_MODERATOR_RULES}"""


def moderator_holdout_prompt(state: DialogueState, candidate_id: str, holdout_ids: list[str]) -> str:
    option = state.scenario.option(candidate_id)
    names = " and ".join(state.name_for(h) for h in holdout_ids)
    if len(state.personas) == 2:
        standing = f"You're okay with {option.name}, but {names} isn't sure yet."
    else:
        standing = f"Most of the group can live with {option.name}; the holdout(s): {names}."
    concerns = ", ".join(f"{state.name_for(h)} cares about {state.persona_by_id(h).main_concern}" for h in holdout_ids)
    return f"""{_MODERATOR_VOICE} (max 32 words) addressed to {names}.
Topic: {state.scenario.topic}
{standing} ({concerns}.)
Address their specific concern about {option.name} — or ask if there's another option everyone could accept. {_MODERATOR_RULES}"""


def _remaining_concerns(state: DialogueState, option_id: str) -> str:
    concerns = []
    for p in state.personas:
        rt = state.runtimes[p.id]
        if rt.explicit_vote != option_id and option_id not in rt.accepted_options:
            concerns.append(f"{p.name} cares about {p.main_concern}")
    return "; ".join(concerns[:2]) if concerns else ""


def moderator_closure_prompt(outcome: RunOutcome, scenario: Scenario, state: DialogueState) -> str:
    if outcome.final_option and outcome.status == "consensus":
        chosen = scenario.option(outcome.final_option).name
        situation = f"The group agreed on {chosen}."
        instruction = (f"Name {chosen} as the decision and suggest one practical thing to do next "
                       "that fits the topic. Warm and brief.")
    elif outcome.final_option and outcome.status == "fallback":
        chosen = scenario.option(outcome.final_option).name
        holdout_concerns = _remaining_concerns(state, outcome.final_option)
        situation = f"No full agreement, but most lean toward {chosen}."
        concern_note = f" Remaining concern: {holdout_concerns}." if holdout_concerns else ""
        instruction = (f"Name {chosen} as the working pick.{concern_note} "
                       "Acknowledge it wasn't everyone's first choice. One practical next step that fits the topic.")
    else:
        blocker = _identify_blocker(state)
        situation = f"The group is split: {_camp_split(state)}."
        instruction = (f"Name the specific thing that blocked agreement: {blocker}. "
                       "Suggest one concrete action to break the deadlock. "
                       "Don't say 'we couldn't decide' or 'we'll figure it out'.")
    return f"""{_MODERATOR_VOICE} (max 28 words) to close the conversation.
Topic: {scenario.topic}
Situation: {situation}
{instruction} {_MODERATOR_RULES}"""


def _identify_blocker(state: DialogueState) -> str:
    lead = leading_option(state)
    if not lead:
        return "no clear front-runner emerged"
    lead_name = state.scenario.option(lead).name
    holdouts = []
    for p in state.personas:
        rt = state.runtimes[p.id]
        if rt.current_preference != lead and rt.explicit_vote != lead and lead not in rt.accepted_options:
            holdouts.append((p.name, p.main_concern))
    if holdouts:
        names = " and ".join(h[0] for h in holdouts)
        concerns = ", ".join(h[1] for h in holdouts[:2])
        return f"{names} couldn't get behind {lead_name} ({concerns})"
    return f"the group couldn't fully commit to {lead_name}"


def _audience_clause(others: list[str]) -> str:
    # Keep the phrasing honest about group size: "hey all" reads wrong in a two-person chat.
    if len(others) == 1:
        return f" It's just you and {others[0]} here — address {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
    return " Greet the group casually (e.g. 'hey all', 'hi everyone')."


def _distinct_from_prior(prior: list[str], what: str) -> str:
    # Greetings/farewells are generated one at a time and otherwise can't see each other,
    # so big groups produce several near-identical "hey all" / "sounds good, talk soon"
    # lines. Showing each speaker what's already been said keeps them varied.
    if not prior:
        return ""
    bullets = "; ".join(compact_words(line, 10) for line in prior[-4:])
    return f" Others already {what}: {bullets}. Say yours differently — don't reuse their words, structure, or opener."


def greeting_line(persona: Persona, topic: str, others: list[str], max_words: int, prior: list[str]) -> str:
    return f"""The group is gathering to chat about a decision (casual group chat, friends/colleagues).
Write a quick, natural hello from {persona.name} — at most {max_words} words, in their voice ({persona.speech_style}).
Just a plain casual greeting, like a real person dropping into a chat ('hey', 'hi all', 'morning').{_audience_clause(others)}{_distinct_from_prior(prior, 'said hi')}
Keep it short and unforced. Do NOT use seminar/meeting openers ('looking forward to', 'excited to dive in', 'great to be here', "can't wait", 'eager to'), do NOT mention the topic or any option, state no opinion, ask nothing. No name prefix, no quotes, no emoji."""


def farewell_line(persona: Persona, scenario: Scenario, outcome: RunOutcome, others: list[str], max_words: int, prior: list[str]) -> str:
    if outcome.final_option and outcome.status in {"consensus", "fallback"}:
        chosen = scenario.option(outcome.final_option).name
        result = f"the group is going with {chosen}"
        if outcome.final_option == persona.preferred_option:
            tone = "You got your top pick — be pleased but brief."
        elif outcome.status == "fallback":
            tone = f"This wasn't your first choice — you can show mild acceptance ('fair enough', 'I can live with it') and keep your personality."
        else:
            tone = "You came around to this — brief and genuine."
    else:
        result = "the group couldn't land on a choice this time"
        tone = "No decision yet — react naturally (mild frustration, acceptance, or suggest checking one thing before next time). Don't just say 'we'll figure it out'."
    audience = (f" It's just you and {others[0]} — sign off to {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
                if len(others) == 1 else "")
    return f"""The discussion just wrapped: {result}.
Write a short, casual sign-off from {persona.name} — at most {max_words} words, in their voice ({persona.speech_style}).
{tone}{audience}{_distinct_from_prior(prior, 'signed off')}
You may name the chosen option plainly, but do NOT describe it or invent any detail about it (no genre, author, plot, attributes, dates, or timeframe like 'next week/next month') — you'd risk getting it wrong. Do NOT re-argue, raise new points, or name other options.
Do NOT use stiff/formal closers like 'looking forward to', 'confirmed and set', 'satisfied with', 'have a great day', 'talk to you soon'. Just a quick casual bye like a real person leaving a chat. No name prefix, no quotes, no emoji."""


def _lean_name(state: DialogueState, persona: Persona) -> str:
    lean = current_lean(state, persona)
    return state.scenario.option(lean).name if lean in state.scenario.option_ids else "no clear pick"


# ---------------------------------------------------------------------------
# Compact runtime persona (replaces full speaker_card in generation prompt)
# ---------------------------------------------------------------------------


def _speaking_habit(persona: Persona) -> str:
    t = persona.traits
    # Primary: the most dominant voice character
    if t.neuroticism >= 3 and t.agreeableness <= 3:
        primary = "worrier — raises what could go wrong before agreeing"
    elif t.directness >= 0.55 and t.response_length <= 2:
        primary = "blunt — short and direct, no preamble"
    elif t.conscientiousness >= 4 and t.openness >= 4:
        primary = "methodical — wants one concrete fact before committing"
    elif t.agreeableness >= 4 and t.compromise_willingness >= 0.6:
        primary = "collaborator — looks for what everyone can live with"
    elif t.conscientiousness >= 4:
        primary = "careful — asks for specifics, checks before agreeing"
    elif t.extraversion >= 4 and t.openness >= 4:
        primary = "energetic — jumps in, builds on what others say"
    elif t.extraversion <= 2:
        primary = "measured — listens, then gives one clear take"
    elif t.compromise_willingness < 0.45:
        primary = "firm — holds position unless shown a concrete fix"
    elif t.openness >= 4:
        primary = "curious — explores before deciding"
    elif t.agreeableness >= 4:
        primary = "agreeable — finds the shared angle quickly"
    elif t.neuroticism >= 3:
        primary = "cautious — flags risks, hedges before committing"
    else:
        primary = "practical — goes with what works, moves things along"
    # Secondary: a distinct behavioral tell from a different trait
    used = primary.split(" — ")[0]
    secondary = ""
    if t.conscientiousness >= 4 and "methodical" not in used and "careful" not in used:
        secondary = "asks for one concrete detail"
    elif t.detail >= 0.6 and "methodical" not in used and "careful" not in used:
        secondary = "pins down a specific number or constraint"
    elif t.neuroticism >= 3 and "worrier" not in used and "cautious" not in used:
        secondary = "names the risk before saying yes"
    elif t.response_length <= 2 and "blunt" not in used:
        secondary = "keeps it to one short line"
    elif t.agreeableness >= 4 and "collaborator" not in used and "agreeable" not in used:
        secondary = "softens objections before landing"
    elif t.extraversion >= 4 and "energetic" not in used:
        secondary = "thinks out loud"
    elif t.openness >= 4 and "curious" not in used and "energetic" not in used:
        secondary = "asks what they don't know yet"
    elif t.compromise_willingness < 0.45 and "firm" not in used:
        secondary = "pushes back if their concern isn't addressed"
    return f"{primary}; {secondary}" if secondary else primary


def _responding_to_line(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    if intent.act in {ActType.OPENING, ActType.VOTE}:
        return ""
    if intent.respond_to_turn is not None:
        for turn in state.turns:
            if turn.index == intent.respond_to_turn:
                return f"Responding to {turn.speaker_name}: \"{compact_words(turn.text, 20)}\""
    target_id = intent.addressee_id
    for turn in reversed(state.turns):
        if turn.speaker_id == persona.id:
            continue
        if target_id and turn.speaker_id != target_id:
            continue
        return f"Responding to {turn.speaker_name}: \"{compact_words(turn.text, 20)}\""
    return ""


def _group_leans(state: DialogueState) -> str:
    parts: list[str] = []
    for persona in state.personas:
        rt = state.runtimes[persona.id]
        lean = rt.current_preference or persona.preferred_option
        if lean in state.scenario.option_ids:
            parts.append(f"{persona.name}→{state.scenario.option(lean).name}")
    return "Group: " + ", ".join(parts)


def runtime_speaker_card(persona: Persona, state: DialogueState, intent: MoveIntent) -> str:
    rt = state.runtimes[persona.id]
    lean = rt.current_preference or persona.preferred_option
    lean_name = state.scenario.option(lean).name if lean in state.scenario.option_ids else "undecided"

    concern = persona.main_concern
    if intent.option_focus:
        focus = intent.option_focus[0]
        if focus != persona.preferred_option and persona.reservation:
            concern = persona.reservation

    lines = [
        f"Speaker: {persona.name}, {persona.role}. Lean: {lean_name}. Voice: {persona.speech_style}.",
        f"Concern: {concern}. Style: {_speaking_habit(persona)}.",
    ]
    if rt.already_said:
        recent = rt.already_said[-2:]
        bullets = "; ".join(f'"{compact_words(s, 10)}"' for s in recent)
        lines.append(f"You already said: {bullets} — don't repeat these points.")
    if lean != persona.preferred_option:
        old_name = state.scenario.option(persona.preferred_option).name
        lines.append(f"Started with {old_name}, now leaning {lean_name}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Turn-generation prompt
# ---------------------------------------------------------------------------


def _others_back(state: DialogueState, persona: Persona, option_focus: list[str]) -> bool:
    """True if another participant already votes/accepts/leans the focused option. Used to
    steer the speaker toward a brief affirmation or a genuinely new angle instead of
    repeating the same headline reason — the convergence 'chorus' problem."""
    if not option_focus:
        return False
    opt = option_focus[0]
    for other in state.personas:
        if other.id == persona.id:
            continue
        rt = state.runtimes[other.id]
        if rt.explicit_vote == opt or opt in rt.accepted_options or rt.current_preference == opt:
            return True
    return False


def _concession_bridge(persona: Persona, option_name: str = "", reservation: str = "") -> str:
    import random
    worry = reservation or persona.main_concern
    opt = option_name or "this"
    bridges = [
        f"Take {opt} — name the one condition about {worry} that would make it fully work.",
        f"Back {opt}: be honest that {worry} is the trade-off you're accepting.",
        f"Go along with {opt} — leave one specific concern about {worry} on the table.",
        f"Commit to {opt} and flag the one thing about {worry} you'd want handled.",
    ]
    if persona.traits.compromise_willingness < 0.5:
        bridges = [
            f"Give {opt} a reluctant yes — only if there's a concrete plan for {worry}.",
            f"You'll live with {opt} but not without a specific fix for {worry} named first.",
        ]
    return random.choice(bridges)


def _face_work(persona: Persona, intent: MoveIntent) -> str:
    import random
    t = persona.traits
    if intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.OPENING}:
        return ""
    if intent.act in {ActType.OBJECT, ActType.PUSH_BACK, ActType.REJECT}:
        if t.agreeableness >= 4:
            return random.choice([
                " Soften it — 'I might be wrong but...' or 'not trying to be difficult, just...'.",
                " Show you get their side first, then say your worry.",
            ])
        if t.neuroticism >= 3:
            return " You're genuinely anxious about this — let that show ('this kind of worries me')."
        if t.directness >= 0.55:
            return " Be direct but not rude — name the problem, skip the diplomacy."
    if intent.act == ActType.PROPOSE_COMPROMISE:
        if t.agreeableness >= 4:
            return " Frame it as a gentle condition or suggestion, not a direct push."
    if intent.act == ActType.REACT:
        if t.agreeableness >= 4:
            return " Find what you agree with first, then add your angle."
        if t.agreeableness <= 2:
            return " Don't just nod — show your own take."
    if intent.act == ActType.SUPPORT:
        if t.conscientiousness >= 4:
            return " Point to one specific attribute from the card that convinced you."
        if t.neuroticism >= 3:
            return " Back it, but name the one risk you're accepting."
    if intent.act == ActType.COMPARE:
        if t.directness >= 0.55:
            return " State your preference plainly — no hedging."
        if t.neuroticism >= 3:
            return " Acknowledge a downside on both sides before landing."
    return ""


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    face = _face_work(persona, intent)
    if intent.act == ActType.OPENING:
        common = " Name your pick casually and say why it fits you personally — don't announce a criterion ('X matters most to me')."
        spoken_ids = {turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"}
        if any(p.id in spoken_ids and p.preferred_option == persona.preferred_option
               for p in state.personas if p.id != persona.id):
            common += " Someone already backed this — give your own distinct angle."
        if t.extraversion >= 4:
            return "Jump in — say what grabbed you." + common
        if t.directness >= 0.55:
            return "Cut to the chase." + common
        return "Say what you're leaning toward." + common
    if intent.act == ActType.ACCEPT:
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name, persona.reservation)
            return bridge
        if _others_back(state, persona, intent.option_focus):
            return "Others agreed already — confirm briefly without echoing their phrasing."
        return "Say you're in — one reason why it's okay for you."
    if intent.act == ActType.VOTE:
        chorus = (" Others already voted this way — brief, don't echo them."
                  if _others_back(state, persona, intent.option_focus) else "")
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name, persona.reservation)
            return f"Name your pick. {bridge}" + chorus
        return "State your pick without 'I'm voting for X because' — vary it. One brief reason." + chorus
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act == ActType.REJECT:
        chorus = (" Others already backed this — brief or add a new angle."
                  if _others_back(state, persona, intent.option_focus) else "")
        if high_compromise:
            return "Back what the group is landing on." + chorus
        return "Say where you stand — hold out only if you mean it." + chorus + face
    if intent.act == ActType.SUPPORT and intent.option_focus:
        focus = intent.option_focus[0]
        lean = state.runtimes[persona.id].current_preference or persona.preferred_option
        if focus != lean and focus in set(persona.acceptable_options):
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name, persona.reservation)
            return f"You're warming up to this. {bridge}"
    if intent.act == ActType.ANSWER:
        return "Answer if the option cards cover it. If they don't, say you're not sure and move on — don't repeat the question back." + face
    by_act = {
        ActType.REACT: "React — a fragment is fine ('yeah fair', 'hmm not sure', 'huh interesting'). Don't re-pitch your option." + face,
        ActType.ASK: "Ask one real question you'd want answered before deciding. Casual, end with '?'." + face,
        ActType.COMPARE: "One genuine strength of theirs; one concrete reason yours fits you better. No attribute lists, no templates." + face,
        ActType.SUPPORT: "Back this from your angle — a personal reason or past experience, not the spec sheet." + face,
        ActType.OBJECT: "Name your specific worry. One concrete thing, not a general critique." + face,
        ActType.PUSH_BACK: "Push back on the exact claim just made." + face,
        ActType.PROPOSE_COMPROMISE: "Name the fix directly — skip 'What if we' and 'How about we'." + face,
    }
    return by_act.get(intent.act, "Respond to the last point directly.")


def _alias_rule(state: DialogueState, intent: MoveIntent) -> str:
    if intent.act in {ActType.OPENING, ActType.VOTE}:
        return "\n- Name options naturally, not 'Option B'."
    mentioned = {opt for t in state.turns for opt in (t.act.option_refs if hasattr(t.act, 'option_refs') else [])}
    if mentioned:
        shorts = ", ".join(f'"{_short_alias(o)}"' for o in state.scenario.options if o.id in mentioned)
        if shorts:
            return f"\n- Use short names for options: {shorts}. Don't repeat full names — shorten like friends would."
        return "\n- Use short names for options already discussed. Don't repeat full names."
    return "\n- Name options naturally, not 'Option B'."


_ALIAS_STOPWORDS = frozenset({"and", "or", "of", "to", "a", "an", "in", "on", "at", "for", "by", "with", "the"})


def _short_alias(option: OptionCard) -> str:
    if option.short_name:
        return option.short_name
    words = option.name.split()
    if len(words) <= 3:
        return option.name
    # 4+ words: take first 2, but swap in word[2] if word[1] is a dangling stopword
    alias = words[:2]
    if alias[-1].lower() in _ALIAS_STOPWORDS and len(words) >= 3:
        alias = [words[0], words[2]]
    return " ".join(alias)


def _verbosity_note(persona: Persona, max_words: int) -> str:
    t = persona.traits
    if t.response_length >= 4 or t.detail >= 0.66:
        return f"Aim for {max_words} words. You like to explain — make your point and add a quick why or detail."
    if t.response_length <= 2:
        target = max(6, max_words // 2)
        return f"Aim for ~{target} words max. You keep it short — one crisp line, no preamble."
    return f"Aim for ~{max_words} words. Natural length — a sentence or two."


def _collect_recent_frames(state: DialogueState, window: int = 4) -> list[str]:
    frames: list[str] = []
    recent = [t for t in state.turns if t.speaker_id != "moderator"][-window:]
    for turn in recent:
        frames.extend(classify_discourse_frames(turn.text))
    return frames


def sim_utterance(
    *,
    persona: Persona,
    state: DialogueState,
    recent_lines: list[str],
    intent: MoveIntent,
    focus_options: list[OptionCard],
    addressee_name: str | None,
    max_words: int,
) -> str:
    option_names = ", ".join(f"{o.id}={o.name}" for o in state.scenario.options)
    opt_choices = "|".join(state.scenario.option_ids)
    card = runtime_speaker_card(persona, state, intent)
    leans = _group_leans(state)
    responding = _responding_to_line(state, persona, intent)
    if intent.act in {ActType.COMPARE, ActType.VOTE, ActType.OPENING}:
        focused = _option_cards(focus_options) if focus_options else "- none"
    else:
        focused = "\n".join(f"- {_option_name_only(o)}" for o in focus_options) if focus_options else "- none"
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    focus_str = ", ".join(intent.option_focus) if intent.option_focus else "-"
    guidance = _move_guidance(state, persona, intent)
    responding_block = f"\n{responding}\n" if responding else "\n"
    address_rule = (f"\n- Responding to {addressee_name}: you can use their name once ('{addressee_name}, ...' or '...{addressee_name}'), then 'you' for their point."
                    if addressee_name else "")
    rt = state.runtimes[persona.id]
    frame_hint = recent_frame_hint(_collect_recent_frames(state), rt.discourse_frames)
    frame_line = f"\n{frame_hint}" if frame_hint else ""
    slot_hint = ""
    if intent.option_focus and intent.act not in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
        focus_opt = intent.option_focus[0]
        if focus_opt in state.coverage:
            covered = state.coverage[focus_opt].covered_slots
            slot_hint_text = covered_slots_hint(focus_opt, covered, [])
            if slot_hint_text:
                frame_line += f"\n{slot_hint_text}"
    alias_rule = _alias_rule(state, intent)
    ctx_line = ""
    if state.scenario.shared_context:
        ctx_line = "\nContext: " + "; ".join(state.scenario.shared_context) + "."
    return f"""Write one natural chat message for the next speaker.

Topic: {state.scenario.topic}
Options: {option_names}{ctx_line}

{card}
{leans}
{responding_block}
Option facts:
{focused}

Recent chat:
{recent}

Act: {intent.act.value}; focus: {focus_str}
Guidance: {guidance}
{_verbosity_note(persona, max_words)}{frame_line}

Rules:
- One line, no name prefix, no quotes. Talk like friends chatting — fragments, shortcuts, reactions all fine. Voice: {persona.speech_style}.
- Vary your opener each turn — fragments, reactions, names, direct points. Never start with an option name in possessive ('X's ...').{address_rule}
- No stock phrases ('outweighs', 'valid point', 'Considering...', 'wins me over', 'seems like a good fit', 'edges ahead', 'still pick'). You know only what's in the option cards — anything else is unknown: say 'I'm not sure' or 'we'd need to check', never a confident claim.{alias_rule}

End with: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER={opt_choices} or -; STANCE=vote|accept|object|reject|propose|neutral."""


# Short, human fix-instructions per validation code. Used to build a focused repair prompt
# instead of re-sending the whole generation prompt (which roughly doubled the tokens of every
# repaired turn). Naming the concrete problem also makes the rewrite land more reliably.
_REPAIR_HINTS = {
    "SPEAKER_PREFIX": "drop the 'Name:' prefix",
    "MULTI_TURN_OUTPUT": "write only one single line",
    "INVALID_OPTION_REFERENCE": "only mention the real options listed",
    "UNGROUNDED_NUMERIC_FACT": "don't state numbers that aren't in the option cards — hedge as 'I think ~X' if unsure",
    "INVENTED_OPTION_ATTRIBUTE": "don't state facts that aren't in the option cards — frame as 'I think they have...' or 'do they...?' instead",
    "DUPLICATE_TURN": "don't repeat another speaker's line — say it in your own words",
    "ECHOED_PHRASE": "don't reuse another speaker's phrasing — reword it your way",
    "GROUP_REPETITION": "don't echo a recent turn — add your own angle",
    "QUESTION_ECHO": "don't re-ask what was just asked — if the cards don't say, hedge ('not sure, we'd need to check') and move on",
    "REPETITIVE_OPENER": "don't open with 'I' (and not 'we'/'Considering'/'Given' either) — start with the other person's point, a reaction ('fair,', 'true,', 'okay but'), the option/detail itself, or a short question",
    "REPEATED_START": "open with different words than the recent turns — don't start the same way again",
    "SELF_REPETITION": "you already made this point — add something new or move toward deciding",
    "UNCLEAR_VOTE": "clearly name your final pick",
    "UNCLEAR_ACCEPT": "clearly say you're agreeing to this option",
    "UNCLEAR_REJECT": "clearly state your objection",
    "QUESTION_IN_CONFIRMATION": "make it a statement, not a question",
    "UNWANTED_QUESTION": "make it a statement, not a question",
    "QUESTION_CHAIN": "don't ask another question — react or state instead",
    "INCOMPLETE_TURN": "finish the thought; don't trail off",
    "ROBOTIC_TEMPLATE": "drop the formulaic phrasing ('outweighs', 'point is valid', 'makes me think', 'seems like a/the best fit', 'still beats') and DON'T open with a participial frame ('Considering...', 'Given the discussion...') — just say it plainly in your own voice",
    "POSSESSIVE_SUBJECT": "don't open with an option name in possessive form ('X's ...') — use a fragment, reaction, or your own take instead",
    "COLLECTIVE_VOICE": "this is your own view, not the committee's — speak for yourself, not 'we'/'our'",
    "CARD_READING": "don't parrot the option card's description — put it in your own words",
    "SELF_NARRATION": "don't narrate your thinking ('I should consider', 'I need to prioritize') — just say the thing directly",
}


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
    """A focused rewrite prompt: just the speaker's voice, the recent chat for context, the bad
    line, and concrete fixes — not the full generation prompt. Keeps the meaning and stance,
    fixes the flagged problems."""
    fixes = "; ".join(dict.fromkeys(_REPAIR_HINTS.get(c, c.lower().replace("_", " ")) for c in issue_codes[: int(cfg.utterances.repair_issue_limit)]))
    option_names = ", ".join(f"{o.id}={o.name}" for o in state.scenario.options)
    opt_choices = "|".join(state.scenario.option_ids)
    recent = "\n".join(recent_lines[-4:]) if recent_lines else "(no recent turns)"
    return f"""Rewrite this one chat line from {persona.name} ({persona.role}). Keep their meaning and stance, but fix the problems. Style: {persona.speech_style}.
Options: {option_names}
Recent chat:
{recent}
Original line: {original_text}
Fix: {fixes}.
Write one natural line, under {max_words} words, no name prefix or quotes; name options in words (not "Option B"); use only facts from the option cards — anything else must be hedged ('I think...', 'do they...?'); don't copy others' wording.
End with a status tag on its own line, exactly: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER is one of {opt_choices} (or - if none); STANCE is one of vote|accept|object|reject|propose|neutral."""
