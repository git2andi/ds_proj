"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from aliases import short_alias_map
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


def setup_scenario(topic: str, n: int, common_option: str | None = None) -> str:
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
                    "short_name": f"1-2 recognizable words copied from the option name; at least {cfg.scenario.short_alias_min_chars} characters",
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
    compromise_rule = (
        f"\n- Option {common_option} must be a credible broad compromise: meaningful, but not obviously dominant."
        if common_option else ""
    )
    return f"""Create a fictional group-decision scenario.

Topic: {topic}
Option ids: {labels}
Decision group: exactly {n} participants.

Requirements:
- Create exactly {len(labels)} options.
- Option names must be specific and realistic, not generic categories.
- Each option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} stable, topic-specific attributes with concrete values people can compare and discuss.
- Every attribute must be a fixed value known now. No placeholders, "unknown"/"TBD", or facts that require a live lookup (availability, current weather, booking status).
- Options must differ meaningfully and expose real trade-offs.{compromise_rule}
- The opening question must ask about priorities and trade-offs, not ask for votes.
- shared_context: 2-3 stable facts about the decision situation (not specific options). If it mentions the decision-makers' group size, it must be exactly {n}.

Return JSON only in this shape:
{_schema(schema)}"""


def setup_personas(topic: str, n: int, trait_rows: list[dict], pref_groups: list[list[str]], options_json: list[dict], common_option: str | None = None) -> str:
    labels = list(cfg.scenario.option_labels)
    names_by_id = {row["id"]: row.get("name", row["id"]) for row in trait_rows}
    group_lines = "\n".join(
        f"  - camp {i + 1}: {', '.join(f'{pid} ({names_by_id.get(pid, pid)})' for pid in g)}"
        for i, g in enumerate(pref_groups)
    )
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
                "scores": {"A": 5, "B": 1, "C": 3, "D": 1},
                "reasons": {"A": ["grounded reason"], "C": ["grounded reason"]},
                "reservation": "one addressable concern about another option",
                "reconsider_if": "condition based on group priorities, not changed facts",
            }
        ],
    }
    compromise_rule = (
        f"- Every participant with agreeableness >= 2 MUST include Option {common_option} in acceptable_options, "
        f"score it at least {cfg.scenario.acceptance_score}, and provide a grounded reason for it."
        if common_option else
        "- Non-stubborn participants must share at least one acceptable option."
    )
    return f"""Create {n} participants for this group decision.

Topic: {topic}
Options (already decided):
{json.dumps(options_json, ensure_ascii=False, indent=2)}

Trait/control profiles to use exactly by id:
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Requirements:
- Everyone should try to reach a workable group decision.
- Participants with agreeableness ≥ 2 need at least {cfg.personas.non_blocker_min_acceptable} acceptable options including their preferred option.
- A participant with agreeableness=1 is deeply resistant to compromise: they hold a strong personal conviction and their acceptable_options should contain only their preferred option. Their backstory must reflect this conviction.
- "scores" rates every option {cfg.scenario.score_min}-{cfg.scenario.score_max} for that person ({cfg.scenario.score_max}=loves it, {cfg.scenario.score_min}=cannot accept). Make scores consistent with the labels: preferred highest, acceptable options {cfg.scenario.acceptance_score} or above, rejected options below {cfg.scenario.acceptance_score}.
- Preference camps — CRITICAL: same-camp participants MUST share exactly one preferred_option (they want the same thing for different personal reasons); different camps MUST choose DIFFERENT preferred_options (this disagreement is what creates the conversation). If two camps end up with the same preferred_option the scenario cannot produce meaningful conflict:
{group_lines}
- Even when participants share a preferred option, give them distinct roles, reasons, and concerns so they don't sound identical.
{compromise_rule}
- Reasons must be grounded only in the option cards above.
- Reconsider conditions must depend on how the group weighs known trade-offs, never on changing an option fact.
- Persona consistency: role, main concern, and preferred option must fit the option's core attributes. If the combination is surprising, the backstory must explain it.
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
        if _has_clear_holdout(state):
            holdout_name, concern = _holdout_info(state)
            overlap_name = state.scenario.option(overlap).name
            push = (f"Most people can live with {overlap_name}. "
                    f"Ask {holdout_name} directly what specifically blocks them from {overlap_name} "
                    f"(their concern is {concern}) and what would need to change. Don't ask the whole group.")
        else:
            push = (f"Point out that {state.scenario.option(overlap).name} may be common ground. "
                    "Ask what one thing would need to change for anyone still doubtful.")
    elif _has_clear_holdout(state):
        holdout_name, concern = _holdout_info(state)
        push = (f"Ask {holdout_name} directly what would need to change about the leading option "
                f"to make it work for them (concern: {concern}). Don't propose a solution yet.")
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
    if outcome.final_option and outcome.status == "successful":
        chosen = scenario.option(outcome.final_option).name
        situation = f"The group agreed on {chosen}."
        instruction = (f"Name {chosen} as the decision and suggest one practical thing to do next "
                       "that fits the topic. Warm and brief.")
    elif outcome.final_option and outcome.status == "majority":
        chosen = scenario.option(outcome.final_option).name
        holdout_concerns = _remaining_concerns(state, outcome.final_option)
        situation = f"No full consensus — majority working pick is {chosen}, but not everyone fully agreed."
        concern_note = f" The remaining concern: {holdout_concerns}." if holdout_concerns else ""
        instruction = (f"Open by being upfront that agreement wasn't unanimous.{concern_note} "
                       f"Then name {chosen} as what the group is going with. "
                       "One practical next step. Don't phrase it as a shared win.")
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
    if outcome.final_option and outcome.status in {"successful", "majority"}:
        chosen = scenario.option(outcome.final_option).name
        result = f"the group is going with {chosen}"
        if outcome.final_option == persona.preferred_option:
            tone = "You got your top pick — be pleased but brief."
        elif outcome.status == "majority":
            tone = f"This wasn't your first choice — you can show mild acceptance ('fair enough', 'I can live with it') and keep your personality."
        else:
            tone = "You came around to this — brief and genuine."
    else:
        result = "the group couldn't land on a choice this time"
        tone = "No decision yet — name the one thing you'd need resolved before the group reconvenes, or how you feel about it. Don't just restate the impasse: not 'tabling this', 'still up in the air', 'sleep on it', 'revisit this later', 'we'll figure it out'."
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


def _responding_to_line(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    if intent.act in {ActType.OPENING, ActType.VOTE}:
        return ""
    if intent.respond_to_turn is not None:
        for turn in state.turns:
            if turn.index == intent.respond_to_turn:
                snippet = compact_words(turn.text, 20)
                return f"Responding to {turn.speaker_name}: \"{snippet}\""
    target_id = intent.addressee_id
    if not target_id:
        return ""
    for turn in reversed(state.turns):
        if turn.speaker_id == persona.id:
            continue
        if target_id and turn.speaker_id != target_id:
            continue
        return f"Responding to {turn.speaker_name}: \"{compact_words(turn.text, 20)}\""
    return ""


def _speaking_behavior(persona: Persona) -> str:
    """Translate traits into observable turn behavior without persona labels."""
    t = persona.traits
    cues: list[str] = []
    if t.conscientiousness >= 4:
        cues.append("check one concrete constraint before agreeing")
    elif t.conscientiousness <= 2:
        cues.append("react to the practical gist without auditing every detail")
    if t.neuroticism >= 3:
        cues.append("state the risk that still needs resolving")
    if t.openness >= 4:
        cues.append("probe a useful alternative or missing trade-off")
    if t.agreeableness >= 4:
        cues.append("acknowledge the other side before disagreeing")
    elif t.agreeableness <= 2:
        cues.append("challenge weak reasoning plainly")
    if _is_direct(t):
        cues.append("put the bottom line early")
    if _is_proactive(t):
        cues.append("move the group toward a next step")
    if t.compromise_willingness < 0.45:
        cues.append("hold the concern until someone addresses it")
    if not cues:
        cues.append("give one practical take and build on the conversation")
    return "; ".join(cues[:3])


def _is_direct(traits) -> bool:
    return traits.extraversion >= 4 and traits.agreeableness <= 3


def _is_proactive(traits) -> bool:
    return traits.extraversion >= 4 or (
        traits.conscientiousness >= 4 and traits.openness >= 4
    )


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
        f"Speaker: {persona.name}; lean: {lean_name}; concern: {concern}; "
        f"behavior: {_speaking_behavior(persona)}.",
    ]
    if rt.already_said:
        lines.append(f"Avoid repeating your last point: \"{compact_words(rt.already_said[-1], 10)}\".")
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
    worry = reservation or persona.main_concern
    opt = option_name or "this"
    if persona.traits.compromise_willingness < 0.5:
        return f"Give {opt} a reluctant yes and name the unresolved {worry} condition."
    return f"Accept {opt}, acknowledge it was not your first choice, and name the {worry} trade-off you can live with."


def _face_work(persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    if intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.OPENING}:
        return ""
    if intent.act in {ActType.OBJECT, ActType.PUSH_BACK, ActType.REJECT}:
        if t.agreeableness >= 4:
            return " Acknowledge their side before naming your worry."
        if t.neuroticism >= 3:
            return " You're genuinely anxious about this — let that unease show in your phrasing."
        if _is_direct(t):
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
        if _is_direct(t):
            return " State your preference plainly — no hedging."
        if t.neuroticism >= 3:
            return " Acknowledge a downside on both sides before landing."
    return ""


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    face = _face_work(persona, intent)
    if intent.act == ActType.OPENING:
        common = " Name your pick casually and give one personal reason."
        spoken_ids = {turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"}
        if any(p.id in spoken_ids and p.preferred_option == persona.preferred_option
               for p in state.personas if p.id != persona.id):
            common += " Someone already backed this — give your own distinct angle."
        if t.extraversion >= 4:
            return "Jump in — say what grabbed you." + common
        if _is_direct(t):
            return "Cut to the chase." + common
        return "Say what you're leaning toward." + common
    if intent.act == ActType.ACCEPT:
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name, persona.reservation)
            return f"Name {opt_name} and use an unhedged first-person acceptance verb. {bridge}"
        if _others_back(state, persona, intent.option_focus):
            return "Others agreed already — confirm briefly without echoing their phrasing."
        return "Name the option and use an unhedged first-person acceptance verb. Give one reason it works for you."
    if intent.act == ActType.VOTE:
        chorus = (" Others already voted this way — brief, don't echo them."
                  if _others_back(state, persona, intent.option_focus) else "")
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name, persona.reservation)
            return f"Name {opt_name} and use a first-person commitment verb without hedging. {bridge}" + chorus
        return "Name the option and use a first-person commitment verb without hedging. Give one brief personal reason." + chorus
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
        return "Answer the question first from the cards; if they do not cover it, say that plainly and move on. Never invent facts or re-ask." + face
    by_act = {
        ActType.REACT: "Acknowledge, challenge, or build on the last useful point. Do not re-pitch your option." + face,
        ActType.ASK: "Ask one answerable question that would help the decision. Keep it casual." + face,
        ActType.COMPARE: "One genuine strength of theirs; one concrete reason yours fits you better. No attribute lists, no templates." + face,
        ActType.SUPPORT: "Back this from your angle — a personal reason or past experience, not the spec sheet." + face,
        ActType.OBJECT: "Name your specific worry. One concrete thing from the option card — don't invent flaws not mentioned there." + face,
        ActType.PUSH_BACK: "Push back on the exact claim just made." + face,
        ActType.PROPOSE_COMPROMISE: (
            "The group has covered the ground. Suggest directly that it's time to each pick one option — briefly explain your lean, then ask everyone to commit. No new trade-offs."
            if state.narrowing_called else
            "Name one workable fix directly without inventing details beyond the cards."
        ) + face,
    }
    return by_act.get(intent.act, "Respond to the last point directly.")


def _alias_rule(state: DialogueState, intent: MoveIntent) -> str:
    if intent.act == ActType.VOTE:
        return "\n- Use the shortest recognizable option name from the Options line."
    if intent.act == ActType.OPENING:
        return "\n- Name options naturally rather than using lettered labels."
    mentioned = {opt for t in state.turns for opt in (t.act.option_refs if hasattr(t.act, 'option_refs') else [])}
    if mentioned:
        aliases = short_alias_map(state.scenario.options)
        shorts = ", ".join(f'"{aliases[o.id]}"' for o in state.scenario.options if o.id in mentioned)
        if shorts:
            return f"\n- Use short names for options: {shorts}. Don't repeat full names — shorten like friends would."
        return "\n- Use short names for options already discussed. Don't repeat full names."
    return "\n- Name options naturally rather than using lettered labels."


def _verbosity_note(persona: Persona, max_words: int) -> str:
    t = persona.traits
    if t.response_length >= 4:
        return f"Use up to {max_words} words and at most two short sentences: one point and its reason, without a mini-speech."
    if t.response_length == 3:
        target = max(10, 2 * max_words // 3)
        return f"Aim for ~{target} words. Direct — one point, no padding."
    if t.response_length == 2:
        target = max(8, max_words // 2)
        return f"Aim for ~{target} words. Brief — one clear line."
    target = max(6, max_words // 3)
    return f"Aim for ~{target} words. Very short — a fragment or a single clause."


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
    aliases = short_alias_map(state.scenario.options)
    option_names = ", ".join(f"{o.id}={aliases[o.id]}" for o in state.scenario.options)
    opt_choices = "|".join(state.scenario.option_ids)
    card = runtime_speaker_card(persona, state, intent)
    responding = _responding_to_line(state, persona, intent)
    if intent.act in {ActType.COMPARE, ActType.VOTE, ActType.OPENING}:
        focused = _option_cards(focus_options) if focus_options else "- none"
    else:
        focused = "\n".join(f"- {_option_name_only(o)}" for o in focus_options) if focus_options else "- none"
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    focus_str = ", ".join(intent.option_focus) if intent.option_focus else "-"
    guidance = _move_guidance(state, persona, intent)
    responding_block = f"\nExact message: {responding}\n" if responding else "\n"
    # Derive the name to address: explicit addressee first, then extract from the
    # responding line for ANSWER/REACT so the model always has a cue to name-drop.
    effective_address = addressee_name
    derived_address = None
    if not effective_address and intent.act in {ActType.ANSWER, ActType.REACT}:
        for _pfx in ("Responding to ",):
            if responding.startswith(_pfx):
                derived_address = responding[len(_pfx):].split(":")[0]
                break
    if effective_address:
        address_rule = f"\n- Address {effective_address} only if it sounds natural."
    elif derived_address:
        address_rule = f"\n- You can use {derived_address}'s name if it flows naturally — don't force it."
    else:
        address_rule = ""
    rt = state.runtimes[persona.id]
    frame_hint = recent_frame_hint(_collect_recent_frames(state), rt.discourse_frames)
    frame_line = f"\n{frame_hint}" if frame_hint else ""
    slot_hint = ""
    if intent.option_focus and intent.act not in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
        focus_opt = intent.option_focus[0]
        if focus_opt in state.coverage:
            covered = state.coverage[focus_opt].covered_slots
            slot_hint_text = covered_slots_hint(focus_opt, covered)
            if slot_hint_text:
                frame_line += f"\n{slot_hint_text}"
    alias_rule = _alias_rule(state, intent)
    ctx_line = ""
    if state.scenario.shared_context:
        ctx_line = "\nContext: " + "; ".join(compact_words(item, 12) for item in state.scenario.shared_context)
    verbosity = _verbosity_note(persona, max_words)
    local_job = (
        f"respond to this exact message first; {guidance[0].lower() + guidance[1:]}"
        if responding else guidance
    )
    return f"""Write one natural chat message for the next speaker.

Topic: {state.scenario.topic}
Options: {option_names}{ctx_line}

{card}
{responding_block}
Option facts:
{focused}

Recent chat:
{recent}

Act: {intent.act.value}; focus: {focus_str}
Local job: {local_job}
{verbosity}{frame_line}

Rules:
- Do only the local job. One line; no name prefix or quotes. Use plain spoken language, not a complete option pitch.
- Vary the opening; don't default to I/we or a bare option name.{address_rule}
- No formulaic filler. Use only card/context facts; otherwise say it's unknown or hedge it.{alias_rule}

End with: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER={opt_choices} or -; STANCE=vote|accept|object|reject|propose|neutral."""


# Short, human fix-instructions per validation code. Used to build a focused repair prompt
# instead of re-sending the whole generation prompt (which roughly doubled the tokens of every
# repaired turn). Naming the concrete problem also makes the rewrite land more reliably.
_REPAIR_HINTS = {
    "SPEAKER_PREFIX": "drop the 'Name:' prefix",
    "MULTI_TURN_OUTPUT": "write only one single line",
    "INVALID_OPTION_REFERENCE": "only mention the real options listed",
    "UNGROUNDED_NUMERIC_FACT": "remove numbers that are not in the option cards and state uncertainty plainly",
    "INVENTED_OPTION_ATTRIBUTE": "remove facts that are not in the option cards; state uncertainty plainly and never turn it into a question",
    "DUPLICATE_TURN": "don't repeat another speaker's line — say it in your own words",
    "ECHOED_PHRASE": "don't reuse another speaker's phrasing — reword it your way",
    "GROUP_REPETITION": "don't echo a recent turn — add your own angle",
    "QUESTION_ECHO": "don't re-ask what was just asked; if the cards do not answer it, state that and move on",
    "REPETITIVE_OPENER": "change the grammatical opening and lead from the local point rather than the speaker",
    "REPEATED_START": "open with different words than the recent turns — don't start the same way again",
    "SELF_REPETITION": "you already made this point — add something new or move toward deciding",
    "UNCLEAR_VOTE": "say the option name explicitly and commit without hedging",
    "UNCLEAR_ACCEPT": "say the option name explicitly and confirm agreement",
    "UNCLEAR_REJECT": "clearly state your objection",
    "QUESTION_IN_CONFIRMATION": "make it a statement, not a question",
    "UNWANTED_QUESTION": "don't respond with a question — if answering, give a direct answer or hedge; otherwise make a statement",
    "QUESTION_CHAIN": "don't ask another question — react or state instead",
    "INCOMPLETE_TURN": "finish the thought; don't trail off",
    "ROBOTIC_TEMPLATE": "replace the formulaic sentence structure with one plain local response in the speaker's own voice",
    "POSSESSIVE_SUBJECT": "don't open with an option name in possessive form; lead from the local point instead",
    "COLLECTIVE_VOICE": "state the speaker's own view rather than speaking for the group",
    "CARD_READING": "don't parrot the option card's description — put it in your own words",
    "SELF_NARRATION": "don't narrate the thinking process; state the point directly",
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
    hints = dict(_REPAIR_HINTS)
    if "REPEATED_START" in issue_codes:
        words = original_text.split()[:3]
        if words:
            hints["REPEATED_START"] = f"don't start with '{' '.join(words)}' — use a completely different first word or phrase"
    fixes = "; ".join(dict.fromkeys(hints.get(c, c.lower().replace("_", " ")) for c in issue_codes[: int(cfg.utterances.repair_issue_limit)]))
    ids = list(dict.fromkeys(intent.option_focus))
    if state.candidate_option and state.candidate_option not in ids:
        ids.append(state.candidate_option)
    if not ids:
        ids.append(persona.preferred_option)
    if "INVALID_OPTION_REFERENCE" in issue_codes:
        ids = state.scenario.option_ids
    repair_options = [state.scenario.option(option_id) for option_id in ids if option_id in state.scenario.option_ids]
    grounding_codes = {"UNGROUNDED_NUMERIC_FACT", "INVENTED_OPTION_ATTRIBUTE", "CARD_READING"}
    if grounding_codes.intersection(issue_codes):
        option_context = _option_cards(repair_options)
    else:
        option_context = ", ".join(f"{o.id}={o.name}" for o in repair_options)
    opt_choices = "|".join(state.scenario.option_ids)
    context_codes = {
        "DUPLICATE_TURN", "ECHOED_PHRASE", "GROUP_REPETITION", "QUESTION_ECHO",
        "REPEATED_START", "SELF_REPETITION",
    }
    recent_block = ""
    if context_codes.intersection(issue_codes) and recent_lines:
        limit = int(cfg.utterances.repair_recent_turns)
        recent_block = "\nRecent chat:\n" + "\n".join(recent_lines[-limit:])
    decision_block = ""
    if intent.act in {ActType.VOTE, ActType.ACCEPT} and intent.option_focus:
        target_id = intent.option_focus[0]
        if target_id in state.scenario.option_ids:
            target = short_alias_map(state.scenario.options)[target_id]
            decision_block = (
                f"\nDecision requirement: name {target} and visibly say you are selecting it now. "
                "Praising it, describing it, or promising to present it is not a choice. The trailer is metadata only."
            )
    return f"""Rewrite {persona.name}'s line. Keep its grounded point and fulfill the routed act.
Behavior: {_speaking_behavior(persona)}.
Relevant options: {option_context}{recent_block}
Line: {original_text}
Fix: {fixes}.{decision_block}
One natural line under {max_words} words; no prefix, quotes, invented facts, questions, or copied wording.
Then: [act={intent.act.value}; opt=LETTER; stance=STANCE] where LETTER={opt_choices} or - and STANCE=vote|accept|object|reject|propose|neutral."""
