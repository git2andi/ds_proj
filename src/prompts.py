"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, Phase, RunOutcome, Scenario
from scoring import current_lean
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
    return f"""{_MODERATOR_VOICE} (max 32 words) for this moment.
Topic: {state.scenario.topic}
The discussion is going in circles. The split right now: {_camp_split(state)}.
Name that split plainly (who's where), note we're repeating the same points, then push toward a decision — propose setting aside any option nobody backs and choosing between the front-runners, or ask a specific holdout what would actually change their mind. Don't just ask 'is anyone's view changing?' again. {_MODERATOR_RULES}"""


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
    return f"""The options have just been laid out and the group is about to discuss: {topic}.
Write a quick, natural hello from {persona.name} ({persona.role}) — at most {max_words} words, in their voice ({persona.speech_style}).
A simple greeting, optionally a few words of light anticipation about deciding this.{_audience_clause(others)}{_distinct_from_prior(prior, 'said hi')}
Do NOT name or favour any option, state any opinion, or ask about the decision yet. No name prefix, no quotes, no emoji."""


def farewell_line(persona: Persona, scenario: Scenario, outcome: RunOutcome, others: list[str], max_words: int, prior: list[str]) -> str:
    if outcome.final_option and outcome.status in {"consensus", "fallback"}:
        result = f"the group is going with {scenario.option(outcome.final_option).name}"
    else:
        result = "the group couldn't land on a choice this time"
    audience = (f" It's just you and {others[0]} — sign off to {others[0]} directly; don't say 'all', 'everyone', 'team', or 'you all'."
                if len(others) == 1 else "")
    return f"""The discussion just wrapped: {result}.
Write a short, casual sign-off from {persona.name} ({persona.role}) — at most {max_words} words, in their voice ({persona.speech_style}).
A quick goodbye that briefly acknowledges the outcome (pleased, relieved, or fine with it; mild disappointment is okay if there was no decision).{audience}{_distinct_from_prior(prior, 'signed off')}
Do NOT re-argue, raise new points, or name other options. No name prefix, no quotes, no emoji."""


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
        if _others_back(state, persona, intent.option_focus):
            return ("Others already agreed to this — keep it short. Either just affirm ('yeah, that works for me') "
                    "or add one genuinely new angle from what you care about; don't repeat the reason already given or reuse their wording.")
        return "Agree to this option as the group's pick, and name the one concrete thing about it you can genuinely live with — don't just say 'works for me'."
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act in {ActType.VOTE, ActType.REJECT}:
        # Stop the vote round from collapsing into a stock template — several
        # "Given the discussion, I think <option>" lines in a row read formulaic.
        no_template = " Don't preface it with 'Given the discussion' or 'I think' — just name your pick in your own voice."
        # A commit turn states a choice cleanly; it must not re-open a worry the speaker
        # already set aside or swing back to a pick they've moved off.
        consistent = " If you already said an option works for you, back that one now — don't revert to your first pick or re-air worries you'd let go."
        # Anti-chorus: when others already backed this same option, don't pile on the same
        # headline reason; affirm in a few words or bring a fresh angle. Name the worst stock
        # closers explicitly — three "X works for me because..." / "seems like the best fit"
        # lines in a row is the clearest tell that the ending was generated.
        chorus = (" Others already backed this exact option — don't reuse their wording or sentence stem, and DON'T use 'works for me because...', 'seems/feels like the best fit', or 'X is key'. Either keep it to a brief, plain '+1' ('yeah, I'm good with that', 'okay, count me in') or add a reason nobody has raised yet."
                  if _others_back(state, persona, intent.option_focus) else "")
        if high_compromise:
            return "Enough discussion. Commit to a workable choice; if a good case was made for an option you rate well enough, move to it instead of restating your first pick." + consistent + chorus + no_template
        return "Enough discussion. State where you stand; only hold out if you are genuinely not convinced." + consistent + chorus + no_template
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
        ActType.REACT: "React to what was just said specifically — agree with part of it or say why it doesn't move you. A short, casual reaction or sentence fragment is fine here ('fair, but it's pricey', 'eh, not sold', 'true'), you don't need a full argument. Don't change the subject to your own pick." + persuadable,
        ActType.ASK: "Ask one real question about the other person's reasoning or what matters to them — not a rhetorical one.",
        ActType.COMPARE: "Genuinely weigh their option against yours — name a real strength of theirs and where yours still wins on what matters most to you, in your own words; don't fall into a 'nice, but I prefer mine' shape." + persuadable,
        ActType.SUPPORT: "Back an option from your own angle — your main concern or a quick bit of your experience — and try a dimension others haven't dwelt on, not the headline everyone's repeating.",
        ActType.OBJECT: "Raise a genuine worry — grounded in what you care about most — about the specific point or option just discussed, staying friendly; don't dismiss the person.",
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
Length & voice: {_verbosity_note(persona)}

Rules:
- One line only, as {persona.name}: no name prefix, no quotes, under {max_words} words. Casual, human group-chat tone (not corporate or slang-heavy); sound like {persona.name}, not a brochure.
- Build on the latest point (agree, extend, or push back on that specific thing); don't ignore it to re-pitch your option.{address_rule}
- It's ONE shared group decision, not advice to one person — never frame an option as good "for you/them" or benefiting one individual; weigh it for the group or say what it means to you.
- Voice what you care about naturally; don't shoehorn your one-word concern in as a literal noun if it reads oddly (no "comfort for our community" about a charity).
- Be persuaded only when a point genuinely lands (then say so and shift) — holding your ground is just as valid; don't fold in your first couple of turns.
- Name options naturally ("the valley walk"), not "Option B" unless needed to disambiguate. Don't open with an option's name, especially its possessive ("Beach Town's cost..."); lead with the team or yourself ("the cafe has way more variety", "I'd get more done in Notion"). Vary your sentence shape from prior speakers.
- Don't restate a point already made or copy another speaker's phrasing. Avoid stock/acknowledge-then-pivot frames: "X outweighs Y", "X's point is valid, but...", "makes me think/reconsider", "Given the discussion...", "seems like the best fit", opening with "Considering...".
- Default to statements; an occasional question that invites others in is fine, but don't end every turn on one. Use only facts from the option cards — invent no prices, ratings, availability, or weather.

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
    "SELF_REPETITION": "you already made this point — add something new or move toward deciding",
    "UNCLEAR_VOTE": "clearly name your final pick",
    "UNCLEAR_ACCEPT": "clearly say you're agreeing to this option",
    "UNCLEAR_REJECT": "clearly state your objection",
    "QUESTION_IN_CONFIRMATION": "make it a statement, not a question",
    "UNWANTED_QUESTION": "make it a statement, not a question",
    "QUESTION_CHAIN": "don't ask another question — react or state instead",
    "INCOMPLETE_TURN": "finish the thought; don't trail off",
    "ROBOTIC_TEMPLATE": "drop the formulaic phrasing ('outweighs', 'point is valid', 'makes me think', 'Given the discussion', 'seems like the best fit') — say it plainly",
    "POSSESSIVE_SUBJECT": "don't start with an option's possessive ('X's ...') — lead with the team or yourself",
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
