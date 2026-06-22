"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, Phase, RunOutcome, Scenario
from scoring import best_overlap_option, current_lean
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


def setup_world(topic: str, n: int, trait_rows: list[dict], pref_groups: list[list[str]]) -> str:
    labels = list(cfg.scenario.option_labels)
    group_lines = "\n".join(f"- group {i + 1}: {', '.join(g)}" for i, g in enumerate(pref_groups))
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
- Preferred-option groups (participants in the SAME group share ONE preferred option; different groups must prefer DIFFERENT options):
{group_lines}
- Even when participants share a preferred option, give them distinct roles, reasons, and concerns so they don't sound identical.
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


# Moderator facilitation lines are written by the LLM (so they vary run to run) from the
# situation data below. Only the opening option board (moderator_opening) stays fixed.
_MODERATOR_VOICE = (
    "You are a relaxed, neutral facilitator in a casual group chat (friends/colleagues), "
    "not a corporate host. Write ONE short spoken line"
)
_MODERATOR_RULES = (
    "Use only the facts given; don't invent options, votes, or reasons. "
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


def moderator_stall_prompt(state: DialogueState) -> str:
    # If a single option actually bridges the split (people in different camps can all live
    # with it), steer the moderator to surface THAT as the common ground rather than just
    # pitting the loudest front-runners against each other (ANALYSIS #4: the mod missed the
    # real overlap candidate). Otherwise fall back to a plain "pick between the front-runners".
    overlap = best_overlap_option(state)
    if overlap is not None:
        push = (f"Point out that {state.scenario.option(overlap).name} may be the common ground — "
                "people on different sides can live with it — and ask if everyone can get behind it, "
                "or ask a specific holdout what would make it work.")
    else:
        push = ("Push toward a decision: propose setting aside any option nobody backs and choosing "
                "between the front-runners, or ask a specific holdout what would actually change their mind.")
    return f"""{_MODERATOR_VOICE} (max 32 words) for this moment.
Topic: {state.scenario.topic}
The discussion is going in circles. The split right now: {_camp_split(state)}.
Name that split plainly (who's where), note we're repeating the same points, then move things forward. {push} Don't just ask 'is anyone's view changing?' again. {_MODERATOR_RULES}"""


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
    # Avoid "most of us" when it's really one-on-one.
    if len(state.personas) == 2:
        standing = f"You're okay with {option.name}, but {names} isn't sure yet."
    else:
        standing = f"Most of the group can live with {option.name}; the holdout(s): {names}."
    return f"""{_MODERATOR_VOICE} (max 32 words) addressed to {names}.
Topic: {state.scenario.topic}
{standing}
Acknowledge where things stand, then ask {names} what would make {option.name} work for them — or whether there's another option everyone could accept. {_MODERATOR_RULES}"""


def moderator_closure_prompt(outcome: RunOutcome, scenario: Scenario, state: DialogueState) -> str:
    if outcome.final_option and outcome.status == "consensus":
        situation = f"The group agreed on {scenario.option(outcome.final_option).name}."
        instruction = "Wrap it up warmly and plainly, naming the chosen option."
    elif outcome.final_option and outcome.status == "fallback":
        situation = f"There was no full agreement, but the strongest workable choice is {scenario.option(outcome.final_option).name}."
        instruction = "Close by naming that as the group's pick, acknowledging it wasn't unanimous."
    else:
        situation = f"The group stayed split and couldn't agree. The split: {_camp_split(state)}."
        instruction = ("Close by naming the actual split (who wants what) and a concrete next step — "
                       "e.g. revisit it later, take a quick vote next time, or settle one open question first. "
                       "Don't just say 'we couldn't decide'.")
    return f"""{_MODERATOR_VOICE} (max 28 words) to close the conversation.
Topic: {scenario.topic}
Situation: {situation}
{instruction} {_MODERATOR_RULES}"""


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
        result = f"the group is going with {scenario.option(outcome.final_option).name}"
    else:
        result = "the group couldn't land on a choice this time"
    audience = (f" It's just you and {others[0]} — sign off to {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
                if len(others) == 1 else "")
    return f"""The discussion just wrapped: {result}.
Write a short, casual sign-off from {persona.name} — at most {max_words} words, in their voice ({persona.speech_style}).
A quick goodbye with a touch of feeling about how it ended (pleased, relieved, or fine with it; mild disappointment is okay if there was no decision).{audience}{_distinct_from_prior(prior, 'signed off')}
You may name the chosen option plainly, but do NOT describe it or invent any detail about it (no genre, author, plot, attributes, dates, or timeframe like 'next week/next month') — you'd risk getting it wrong. Do NOT re-argue, raise new points, or name other options. No name prefix, no quotes, no emoji."""


def _lean_name(state: DialogueState, persona: Persona) -> str:
    lean = current_lean(state, persona)
    return state.scenario.option(lean).name if lean in state.scenario.option_ids else "no clear pick"


# ---------------------------------------------------------------------------
# Turn-generation prompt
# ---------------------------------------------------------------------------


def speaker_card(persona: Persona, focus_ids: Iterable[str], current_lean: Optional[str] = None) -> str:
    t = persona.traits
    smin, smax, thr = int(cfg.scenario.score_min), int(cfg.scenario.score_max), int(cfg.scenario.acceptance_score)
    ratings = ", ".join(f"{opt}={persona.score_for(opt)}" for opt in sorted(persona.option_scores))
    # Only the reasons for options in play this turn, to keep the prompt small.
    keep = {persona.preferred_option, *focus_ids}
    reasons = {opt: vals for opt, vals in persona.reasons.items() if opt in keep}
    # If the persona has already moved off their first pick, lead with where they stand
    # NOW so the model doesn't keep dragging them back to their original favourite.
    if current_lean and current_lean != persona.preferred_option:
        stance = (f"started out preferring {persona.preferred_option} but has moved to {current_lean}; "
                  f"can live with {persona.acceptable_options}; concerns {persona.soft_rejections}; hard rejects {persona.hard_rejections}")
    else:
        stance = (f"prefers {persona.preferred_option}; can live with {persona.acceptable_options}; "
                  f"concerns {persona.soft_rejections}; hard rejects {persona.hard_rejections}")
    return f"""{persona.name} ({persona.role})
style: {persona.speech_style}
who you are: {persona.backstory}
what you want: {persona.private_goal}; what you care about most: {persona.main_concern}
stance: {stance}
private ratings (hidden, {smin}-{smax}, you can accept {thr}+): {ratings}
traits: extra={t.extraversion}, agree={t.agreeableness}, direct={t.directness:.2f}, compromise={t.compromise_willingness:.2f}, detail={t.detail:.2f}
reasons: {json.dumps(reasons, ensure_ascii=False)}
reservation: {persona.reservation}; you'd reconsider if: {persona.reconsider_if}"""


def intent_line(intent: MoveIntent, state: DialogueState, addressee_name: Optional[str]) -> str:
    focus = ", ".join(intent.option_focus) if intent.option_focus else "none"
    to_part = f" to {addressee_name}" if addressee_name else ""
    return f"move={intent.act.value}{to_part}; focus={focus}; reason={intent.reason}; length={intent.length_hint}"


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


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    if intent.act == ActType.OPENING:
        common = " Say which one caught your eye and why — one quick take, no vote yet. Lead with the detail, not 'I prefer'."
        spoken_ids = {turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"}
        if any(p.id in spoken_ids and p.preferred_option == persona.preferred_option
               for p in state.personas if p.id != persona.id):
            common += " Someone already backed this option — give your own distinct reason."
        if t.extraversion >= 4:
            return "Open with energy." + common
        if t.directness >= 0.55:
            return "Get to the point." + common
        if t.detail >= 0.6:
            return "Point at one concrete detail that catches your eye." + common
        return "Say casually which one you're leaning toward." + common
    if intent.act == ActType.ACCEPT:
        if _others_back(state, persona, intent.option_focus):
            return "Others already agreed — a quick 'yeah works for me' is fine, don't repeat their reason."
        return "Agree to this option — name the one thing that makes it okay for you."
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act in {ActType.VOTE, ActType.REJECT}:
        consistent = " Back what you already said works — don't revert or re-air old worries."
        chorus = (" Others already backed this — brief '+1' or a new reason, don't reuse their words."
                  if _others_back(state, persona, intent.option_focus) else "")
        if high_compromise:
            return "Commit to a workable choice; move to what the group built a case for." + consistent + chorus
        return "State where you stand; only hold out if genuinely not convinced." + consistent + chorus
    if intent.act == ActType.SUPPORT and intent.option_focus:
        focus = intent.option_focus[0]
        lean = state.runtimes[persona.id].current_preference or persona.preferred_option
        if focus != lean and focus in set(persona.acceptable_options):
            return "You're being won over — say what specifically changed your mind. Don't relitigate your earlier pick."
    persuadable = (
        " If their point lands, say so and shift."
        if high_compromise else
        " Acknowledge a fair point before pushing back."
    )
    by_act = {
        ActType.REACT: "React to what was just said — a short fragment is fine ('yeah that's fair', 'eh, not for me'). Don't re-pitch your option." + persuadable,
        ActType.ASK: "Ask one real question you'd actually want answered before deciding. Keep it casual.",
        ActType.COMPARE: "Weigh their option against yours — name a real strength of theirs and where yours still wins." + persuadable,
        ActType.SUPPORT: "Back this option from your own angle — your concern or experience. Try a dimension others haven't dwelt on.",
        ActType.OBJECT: "Raise a genuine worry about this option, grounded in what you care about. Stay friendly.",
        ActType.PUSH_BACK: "Push back on the exact point just made and say why." + persuadable,
        ActType.PROPOSE_COMPROMISE: "Offer a compromise that answers others' concerns, not a relabel of your pick.",
    }
    return by_act.get(intent.act, "Respond to the last point; don't restate what's been said.")


def _verbosity_note(persona: Persona, max_words: int) -> str:
    t = persona.traits
    if t.response_length >= 4 or t.detail >= 0.66:
        return f"Aim for {max_words} words. You like to explain — make your point and add a quick why or detail."
    if t.response_length <= 2:
        target = max(6, max_words // 2)
        return f"Aim for ~{target} words max. You keep it short — one crisp line, no preamble."
    return f"Aim for ~{max_words} words. Natural length — a sentence or two."


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
    opt_choices = "|".join(state.scenario.option_ids)
    focused = _option_cards(focus_options) if focus_options else "- none"
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    already = ""
    if own_recent:
        bullets = "\n".join(f"- {compact_words(line, 16)}" for line in own_recent[-3:])
        already = f"\nYou already said this (do NOT repeat it; say something new or move toward deciding):\n{bullets}\n"
    # When the turn is aimed at one person, talk TO them ("you") rather than narrating them
    # in the third person ("Liam's point about ..."), which reads stilted between two people.
    address_rule = (
        f"\n- You're reacting to {addressee_name}: you can use 'you' once for your reaction to their point, "
        f"but don't start the line with 'Your...' and don't narrate them in the third person "
        f"('{addressee_name}'s point'). Lead with your own thought, not with them."
        if addressee_name else ""
    )
    return f"""Write exactly one natural chat message for the next speaker.

Topic: {state.scenario.topic}
Available options: {option_names}
Speaker:
{speaker_card(persona, intent.option_focus, state.runtimes[persona.id].current_preference)}

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
Length & voice: {_verbosity_note(persona, max_words)}

Rules:
- One line only, no name prefix, no quotes. Talk like friends deciding something — short sentences, fragments OK, casual ('honestly', 'okay but', 'hmm', 'right?'). Your voice is {persona.speech_style}.
- React first: pick up the specific thing just said before adding your own view.{address_rule}
- Don't start with 'I' most of the time. Open with a reaction, the detail itself, or their point. Don't swap 'I' for committee 'we' either — say 'we' only for genuine joint suggestions.
- Put things in your own words. Never copy the option card's description/upside/tradeoff text. Never narrate your thinking ('I should consider', 'I need to prioritize').
- Tie reasons to concrete details from the cards and what they'd mean, not bare value words. Name options naturally, not "Option B". Use only facts from the cards.
- Don't fold easily — hold your ground in the first few turns. Don't repeat points already made. No stock frames ('outweighs', 'point is valid', 'Given the discussion', 'Considering', 'seems like the best fit').

End with a status tag on its own line, exactly: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER is one of {opt_choices} (or - if none); STANCE is one of vote|accept|object|reject|propose|neutral (vote=final pick, accept=agree to a compromise, object=mild concern, reject=dealbreaker, propose=offer a compromise, neutral=otherwise)."""


# Short, human fix-instructions per validation code. Used to build a focused repair prompt
# instead of re-sending the whole generation prompt (which roughly doubled the tokens of every
# repaired turn). Naming the concrete problem also makes the rewrite land more reliably.
_REPAIR_HINTS = {
    "SPEAKER_PREFIX": "drop the 'Name:' prefix",
    "MULTI_TURN_OUTPUT": "write only one single line",
    "INVALID_OPTION_REFERENCE": "only mention the real options listed",
    "UNGROUNDED_NUMERIC_FACT": "don't state numbers that aren't in the option cards",
    "INVENTED_OPTION_ATTRIBUTE": "don't state numbers/facts that aren't in the option cards",
    "DUPLICATE_TURN": "don't repeat another speaker's line — say it in your own words",
    "ECHOED_PHRASE": "don't reuse another speaker's phrasing — reword it your way",
    "GROUP_REPETITION": "don't echo a recent turn — add your own angle",
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
    "ROBOTIC_TEMPLATE": "drop the formulaic phrasing ('outweighs', 'point is valid', 'makes me think', 'seems like the best fit') and DON'T open with a participial frame ('Considering...', 'Given the discussion...') — just say it plainly in your own voice",
    "POSSESSIVE_SUBJECT": "don't start with an option's possessive ('X's ...') — lead with the team or yourself",
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
Write one natural line, under {max_words} words, no name prefix or quotes; name options in words (not "Option B"); use only facts from the options; don't copy others' wording.
End with a status tag on its own line, exactly: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER is one of {opt_choices} (or - if none); STANCE is one of vote|accept|object|reject|propose|neutral."""
