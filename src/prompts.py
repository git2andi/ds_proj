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


def _standings(state: DialogueState) -> str:
    return "; ".join(f"{p.name} leans {_lean_name(state, p)}" for p in state.personas)


def moderator_stall_prompt(state: DialogueState) -> str:
    return f"""{_MODERATOR_VOICE} (max 30 words) for this moment.
Topic: {state.scenario.topic}
The discussion is going in circles. Where people stand: {_standings(state)}.
Gently point out we're repeating ourselves, reflect roughly where things stand, and ask whether anyone's view is shifting or if it's time to start narrowing down. {_MODERATOR_RULES}"""


def moderator_agreement_prompt(state: DialogueState, candidate_id: str) -> str:
    option = state.scenario.option(candidate_id)
    return f"""{_MODERATOR_VOICE} (max 28 words) for this moment.
Topic: {state.scenario.topic}
It looks like everyone is converging on {option.name}.
Note that the group seems agreed on {option.name} and ask if anyone objects or if you should lock it in. {_MODERATOR_RULES}"""


def moderator_holdout_prompt(state: DialogueState, candidate_id: str, holdout_ids: list[str]) -> str:
    option = state.scenario.option(candidate_id)
    names = " and ".join(state.name_for(h) for h in holdout_ids)
    return f"""{_MODERATOR_VOICE} (max 32 words) addressed to {names}.
Topic: {state.scenario.topic}
Most of the group can live with {option.name}; the holdout(s): {names}.
Acknowledge most are fine with {option.name}, then ask {names} what would make it work for them — or whether there's another option everyone could accept. {_MODERATOR_RULES}"""


def moderator_closure_prompt(outcome: RunOutcome, scenario: Scenario) -> str:
    if outcome.final_option and outcome.status == "consensus":
        situation = f"The group agreed on {scenario.option(outcome.final_option).name}."
    elif outcome.final_option and outcome.status == "fallback":
        situation = f"There was no full agreement, but the strongest workable choice is {scenario.option(outcome.final_option).name}."
    else:
        situation = "The group could not agree on any option."
    return f"""{_MODERATOR_VOICE} (max 26 words) to close the conversation.
Topic: {scenario.topic}
Situation: {situation}
Wrap it up warmly and plainly, naming the chosen option if there is one. {_MODERATOR_RULES}"""


def _audience_clause(others: list[str]) -> str:
    # Keep the phrasing honest about group size: "hey all" reads wrong in a two-person chat.
    if len(others) == 1:
        return f" It's just you and {others[0]} here — address {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
    return " Greet the group casually (e.g. 'hey all', 'hi everyone')."


def greeting_line(persona: Persona, topic: str, others: list[str], max_words: int) -> str:
    return f"""The options have just been laid out and the group is about to discuss: {topic}.
Write a quick, natural hello from {persona.name} ({persona.role}) — at most {max_words} words, in their voice ({persona.speech_style}).
A simple greeting, optionally a few words of light anticipation about deciding this.{_audience_clause(others)}
Do NOT name or favour any option, state any opinion, or ask about the decision yet. No name prefix, no quotes, no emoji."""


def farewell_line(persona: Persona, scenario: Scenario, outcome: RunOutcome, others: list[str], max_words: int) -> str:
    if outcome.final_option and outcome.status in {"consensus", "fallback"}:
        result = f"the group is going with {scenario.option(outcome.final_option).name}"
    else:
        result = "the group couldn't land on a choice this time"
    audience = (f" It's just you and {others[0]} — sign off to {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
                if len(others) == 1 else "")
    return f"""The discussion just wrapped: {result}.
Write a short, casual sign-off from {persona.name} ({persona.role}) — at most {max_words} words, in their voice ({persona.speech_style}).
A quick goodbye that briefly acknowledges the outcome (pleased, relieved, or fine with it; mild disappointment is okay if there was no decision).{audience}
Do NOT re-argue, raise new points, or name other options. No name prefix, no quotes, no emoji."""


def _lean_name(state: DialogueState, persona: Persona) -> str:
    rt = state.runtimes[persona.id]
    lean = rt.explicit_vote or rt.current_preference or persona.preferred_option
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


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    if intent.act == ActType.OPENING:
        # Vary the opening framing by trait so the first round isn't three identical lines.
        # The shared rider kills the "I prioritize X and lean toward Option Y" template that
        # otherwise makes every opener sound the same.
        common = " Talk like a person to friends, not like you're filling in a form. Don't start with 'I prioritize' or 'I care about', and don't lock in a final vote yet."
        # Coalition echo guard: if someone who already opened favours this same option,
        # nudge toward a distinct, personal angle so coalition members don't paraphrase
        # each other (e.g. both calling the same option "engaging").
        spoken_ids = {turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"}
        if any(p.id in spoken_ids and p.preferred_option == persona.preferred_option
               for p in state.personas if p.id != persona.id):
            common += (" Someone already backed this same option — give your own distinct"
                       " reason (your goal, your experience), don't echo their wording.")
        if t.extraversion >= 4:
            return "Open with a bit of energy: react to the idea and say which option you're drawn to and why." + common
        if t.directness >= 0.55:
            return "Get to the point: which option you lean to and the single thing that matters most to you." + common
        if t.detail >= 0.6:
            return "Point at one concrete detail of the option you lean to that catches your eye, and why." + common
        return "Say casually which option you're leaning toward and roughly why." + common
    if intent.act == ActType.ACCEPT:
        return "Agree to this option as the group's pick, and name the one concrete thing about it you can genuinely live with — don't just say 'works for me'."
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act in {ActType.VOTE, ActType.REJECT}:
        # Stop the vote round from collapsing into a stock template — several
        # "Given the discussion, I think <option>" lines in a row read formulaic.
        no_template = " Don't preface it with 'Given the discussion' or 'I think' — just name your pick in your own voice."
        # A commit turn states a choice cleanly; it must not re-open a worry the speaker
        # already set aside or swing back to a pick they've moved off.
        consistent = " If you already said an option works for you, back that one now — don't revert to your first pick or re-air worries you'd let go."
        if high_compromise:
            return "Enough discussion. Commit to a workable choice; if a good case was made for an option you rate well enough, move to it instead of restating your first pick." + consistent + no_template
        return "Enough discussion. State where you stand; only hold out if you are genuinely not convinced." + consistent + no_template
    # Discussion phase: every line should engage the conversation, not re-pitch your option.
    # A SUPPORT move aimed at an option the speaker can live with but isn't currently
    # leaning toward is a genuine change of mind — frame it as being won over, not as a
    # fresh pitch for a new favourite.
    if intent.act == ActType.SUPPORT and intent.option_focus:
        focus = intent.option_focus[0]
        lean = state.runtimes[persona.id].current_preference or persona.preferred_option
        if focus != lean and focus in set(persona.acceptable_options):
            return (
                "A real case was just made for this option and you can genuinely live with it — "
                "say what specifically won you over and that you're switching to it. "
                "Don't relitigate your earlier pick."
            )
    persuadable = (
        " If their point genuinely eases your concern or fits what you want, say it changed your mind and move toward their option."
        if high_compromise else
        " You don't shift easily, but acknowledge a fair point before you push back."
    )
    by_act = {
        ActType.REACT: "React to what was just said specifically — agree with part of it or say why it doesn't move you. Don't change the subject to your own pick." + persuadable,
        ActType.ASK: "Ask one real question about the other person's reasoning or what matters to them — not a rhetorical one.",
        ActType.COMPARE: "Genuinely weigh their option against yours — name a real strength of theirs and where yours still wins for you, in your own words; don't fall into a 'nice, but I prefer mine' shape." + persuadable,
        ActType.SUPPORT: "Back an option with a reason rooted in who you are (your goal or your experience), not just its spec sheet.",
        ActType.OBJECT: "Raise a genuine worry about the specific point or option just discussed, staying friendly; don't dismiss the person.",
        ActType.PUSH_BACK: "Push back on the exact point just made — quote or paraphrase it — and say why, while staying cooperative." + persuadable,
        ActType.PROPOSE_COMPROMISE: "Offer a compromise that actually answers others' concerns, not just a relabel of your favourite. Say why it could work for them.",
    }
    return by_act.get(intent.act, "Respond to the last point and add one genuine thought; don't restate what's already been said.")


def _verbosity_note(persona: Persona) -> str:
    # Turn the persona's verbosity traits into a concrete instruction so length/detail
    # actually shows in the text, rather than every speaker landing on the same one-liner.
    t = persona.traits
    if t.response_length >= 4 or t.detail >= 0.66:
        return "You tend to explain yourself — make your point and add a brief why or one concrete detail (a second short sentence is fine), not just a one-liner."
    if t.response_length <= 2:
        return "You keep it short — one crisp sentence, no preamble."
    return "Keep it a natural length — a sentence or two."


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
        f"\n- You're speaking directly to {addressee_name}: refer to them as 'you'/'your', not in the "
        f"third person ('{addressee_name}'s point') — but do NOT start your line with 'Your...'; lead with "
        f"your own reaction or thought and weave the 'you' in naturally."
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
Length & voice: {_verbosity_note(persona)}

Rules:
- Write only {persona.name}'s message: one line, no name prefix, no quotes, under {max_words} words.
- Casual group-chat tone: human, not corporate, not slang-heavy. Sound like {persona.name}, speaking from what you want and who you are — not like a brochure listing features.
- Build on the chat: engage with the most recent point (agree, build on it, or push back on that specific thing). Don't ignore others and re-pitch your option.{address_rule}
- It's good to be persuaded: if someone's point genuinely lands, say so and shift — people change their minds in real discussions.
- Refer to options by their name (e.g. "the contest", "the valley walk"), not "Option B" — only use a letter if you truly need it to disambiguate.
- Don't open with an option's name, and vary your sentence shape from the previous speakers — don't reuse their opener or rhythm.
- Avoid stock templates: "X's <feature> beats/outweighs Y's", "Considering <X>, <Y> could be...", "<option> is appealing/nice, but I still prefer mine", "Given the discussion, I think...". Speak like a person, not a spec comparison or a fill-in-the-blank.
- Don't restate a point already made (yours or anyone's) or copy the previous speaker's phrasing.
- Default to statements, but an occasional open or rhetorical question that invites the others in is fine — just don't turn every turn into a question or end on one out of habit.
- Use only facts from the option cards; invent nothing (no prices, ratings, availability, policies, weather).

End your message with a status tag on its own line, copied in this exact shape — keep the square brackets, and put a single option letter in opt (never a list): [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER is one of {opt_choices} (or - if none); STANCE is one of vote|accept|object|reject|propose|neutral (vote=final pick, accept=agree to a compromise, object=mild concern, reject=dealbreaker, propose=offer a compromise, neutral=otherwise)."""


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
