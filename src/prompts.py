"""All LLM-facing prompts and all chat text templates.

No other module should contain prose that is sent to an LLM or printed as a
moderator/chat message.  Other modules pass structured data into these functions.
"""

from __future__ import annotations

import json
import re
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


def setup_scenario(topic: str, n: int) -> str:
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
    return f"""Create a fictional group-decision scenario.

Topic: {topic}
Option ids: {labels}
Decision group: exactly {n} participants.

Requirements:
- Create exactly {len(labels)} options.
- Option names must be specific and realistic, not generic categories.
- Each option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} stable, topic-specific attributes with concrete values people can compare and discuss.
- Every attribute must be a fixed value known now. No placeholders, "unknown"/"TBD", or facts that require a live lookup (availability, current weather, booking status).
- Options must differ meaningfully and expose real trade-offs.
- The opening question must ask about priorities and trade-offs, not ask for votes.
- shared_context: 2-3 stable facts about the decision situation (not specific options). If it mentions the decision-makers' group size, it must be exactly {n}.

Return JSON only in this shape:
{_schema(schema)}"""


def setup_personas(topic: str, n: int, trait_rows: list[dict], required_preferences: dict[str, str], options_json: list[dict]) -> str:
    names_by_id = {row["id"]: row.get("name", row["id"]) for row in trait_rows}
    preference_lines = "\n".join(
        f"  - {pid} ({names_by_id.get(pid, pid)}): {option_id}"
        for pid, option_id in sorted(required_preferences.items(), key=lambda item: int(item[0][1:]))
    )
    schema = {
        "participants": [
            {
                "id": "p1",
                "name": "short first name",
                "background": "one sentence: a personal detail that shapes how they approach this specific decision",
                "private_goal": "what they personally want from this decision",
                "preferred_options": ["A"],
                "rejection": None,
                "rejection_reason": "",
            }
        ],
    }
    return f"""Create {n} participants for this group decision.

Topic: {topic}
Options (already decided):
{json.dumps(options_json, ensure_ascii=False, indent=2)}

Trait/control profiles to use exactly by id:
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Required primary option for each participant — use exactly:
{preference_lines}

Requirements:
- Use the exact name from each trait profile row. Do not change or substitute names.
- preferred_options: ordered list; preferred_options[0] MUST equal that participant's required primary option above. Optionally add one more if it genuinely fits the same background and goal. Maximum 2.
- rejection: set only for participants with agreeableness=1 (hard blockers) when grounded in the option cards. Leave null for everyone else.
- background: one sentence of private personal detail (a habit, past experience, or constraint) that explains why their required primary option fits them. Must stay inside the decision context described in shared_context — do not invent a separate event, relationship, or situation not mentioned there.
- private_goal: what they personally want from this group decision; it must directly support their required primary option and be achievable within the constraints described in shared_context.
- Everyone should try to reach a workable group decision; even a hard blocker remains civil.

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
    "Use only the names and facts given here — do not invent names, votes, reasons, or any quality, "
    "attribute, or description of any option beyond its name. "
    "Say 'leaning toward' or 'seems to prefer', never 'voted for', unless someone explicitly cast a visible vote. "
    "Do not suggest combining, blending, or merging options — the options are fixed. "
    "No quotes, no name prefix, no lists, no emoji."
)


def _camp_split(state: DialogueState) -> str:
    """Compact tally of current leanings — explicitly labelled as leans, not votes.
    '3 leaning toward City Break, 1 undecided' — so the moderator uses correct state language."""
    from collections import Counter
    counts = Counter(_lean_name(state, p) for p in state.personas)
    parts = [
        f"{n} leaning toward {name}" if name != "no clear pick" else f"{n} undecided"
        for name, n in counts.most_common()
    ]
    return ", ".join(parts)


def _vote_summary(state: DialogueState) -> str:
    """List of personas who have cast explicit visible votes, for use in moderator prompts
    that must distinguish leanings from binding commitments."""
    voted = [
        f"{p.name} voted for {state.scenario.option(rt.explicit_vote).name}"
        for p in state.personas
        if (rt := state.runtimes[p.id]).explicit_vote
        and rt.explicit_vote in state.scenario.option_ids
    ]
    if not voted:
        return "No explicit votes yet — only leanings."
    return "; ".join(voted)


def moderator_stall_prompt(state: DialogueState) -> str:
    overlap = best_overlap_option(state)
    split = _camp_split(state)
    if overlap is not None:
        if _has_clear_holdout(state):
            holdout_name, concern = _holdout_info(state)
            overlap_name = state.scenario.option(overlap).name
            push = (f"Most people can live with {overlap_name}. "
                    f"Ask {holdout_name} directly what specifically blocks them from {overlap_name} "
                    f"({concern}) and what would need to change. Don't ask the whole group.")
        else:
            push = (f"Point out that {state.scenario.option(overlap).name} may be common ground. "
                    "Ask what one thing would need to change for anyone still doubtful.")
    elif _has_clear_holdout(state):
        holdout_name, concern = _holdout_info(state)
        push = (f"Ask {holdout_name} directly what would need to change about the leading option "
                f"to make it work for them ({concern}). Don't propose a solution yet.")
    else:
        push = ("The group is genuinely split. Ask one specific question to surface the real blocker: "
                "what's the one thing that would change someone's mind? Don't propose a compromise yet.")
    names = ", ".join(p.name for p in state.personas)
    return f"""{_MODERATOR_VOICE} (max 32 words) for this moment.
Topic: {state.scenario.topic}. Participants: {names}.
The split: {split}.
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
            pref = rt.current_preference or p.preferred_option
            pref_name = state.scenario.option(pref).name if pref in state.scenario.option_ids else "their option"
            return p.name, f"they prefer {pref_name}"
    return "someone", "their concern"


def moderator_agreement_prompt(state: DialogueState, candidate_id: str) -> str:
    option = state.scenario.option(candidate_id)
    whole = "you both" if len(state.personas) == 2 else "the group"
    votes = _vote_summary(state)
    return f"""{_MODERATOR_VOICE} (max 28 words) for this moment.
Topic: {state.scenario.topic}
Candidate: {_option_brief(option)}
Votes so far: {votes}
State as a fact that {whole} seem to be leaning toward or have chosen {option.name}. Use "leaning toward" if no one has explicitly voted yet. Declarative sentence — no question, no "should we", no "anyone object". {_MODERATOR_RULES}"""


def moderator_holdout_prompt(state: DialogueState, candidate_id: str, holdout_ids: list[str]) -> str:
    option = state.scenario.option(candidate_id)
    names = " and ".join(state.name_for(h) for h in holdout_ids)
    if len(state.personas) == 2:
        standing = f"You're okay with {option.name}, but {names} isn't sure yet."
    else:
        standing = f"Most of the group can live with {option.name}; the holdout(s): {names}."
    prefs = ", ".join(
        f"{state.name_for(h)} prefers "
        + (state.scenario.option(
            state.runtimes[h].current_preference or state.persona_by_id(h).preferred_option
        ).name if (
            state.runtimes[h].current_preference or state.persona_by_id(h).preferred_option
        ) in state.scenario.option_ids else "another option")
        for h in holdout_ids
    )
    return f"""{_MODERATOR_VOICE} (max 32 words) addressed to {names}.
Topic: {state.scenario.topic}
Candidate: {_option_brief(option)}
{standing} ({prefs}.)
Address what's blocking them from {option.name} — or ask if there's another option everyone could accept. {_MODERATOR_RULES}"""


def _remaining_concerns(state: DialogueState, option_id: str) -> str:
    concerns = []
    for p in state.personas:
        rt = state.runtimes[p.id]
        if rt.explicit_vote != option_id and option_id not in rt.accepted_options:
            pref = rt.current_preference or p.preferred_option
            pref_name = state.scenario.option(pref).name if pref in state.scenario.option_ids else "another option"
            concerns.append(f"{p.name} prefers {pref_name}")
    return "; ".join(concerns[:2]) if concerns else ""


def moderator_closure_prompt(outcome: RunOutcome, scenario: Scenario, state: DialogueState) -> str:
    if outcome.final_option and outcome.status == "successful":
        option = scenario.option(outcome.final_option)
        situation = f"The group agreed on {option.name}."
        instruction = (f"Name {option.name} as the decision. Add one practical next step based on one specific card fact "
                       f"(use this data: {_option_brief(option)}). "
                       "Declarative sentence — no question, no 'should we'. Don't list multiple facts.")
    elif outcome.final_option and outcome.status == "majority":
        option = scenario.option(outcome.final_option)
        holdout_concerns = _remaining_concerns(state, outcome.final_option)
        situation = f"No full consensus — majority working pick is {option.name}, but not everyone fully agreed."
        concern_note = f" The remaining concern: {holdout_concerns}." if holdout_concerns else ""
        instruction = (f"Open by being upfront that agreement wasn't unanimous.{concern_note} "
                       f"Then name {option.name} as what the group is going with. "
                       "One practical next step. Don't phrase it as a shared win.")
    else:
        blocker = _identify_blocker(state)
        situation = f"The group is split: {_camp_split(state)}."
        instruction = (f"Name the specific thing that blocked agreement: {blocker}. "
                       "Suggest one procedural next step (check a fact, meet again, narrow to two options) — "
                       "don't invent facts about any option. "
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
            pref = rt.current_preference or p.preferred_option
            pref_name = state.scenario.option(pref).name if pref in state.scenario.option_ids else "another option"
            holdouts.append((p.name, pref_name))
    if holdouts:
        names = " and ".join(h[0] for h in holdouts)
        blocked = ", ".join(f"{h[0]} preferred {h[1]}" for h in holdouts[:2])
        return f"{names} couldn't get behind {lead_name} ({blocked})"
    return f"the group couldn't fully commit to {lead_name}"


def _audience_clause(others: list[str]) -> str:
    if len(others) == 1:
        return f" Address {others[0]} directly — this is a 2-person chat, not a group."
    return ""  # group greeting: no special constraint; the prompt already says "casual"


def _distinct_from_prior(prior: list[str], what: str) -> str:
    # Greetings/farewells are generated one at a time and otherwise can't see each other,
    # so big groups produce several near-identical "hey all" / "sounds good, talk soon"
    # lines. Showing each speaker what's already been said keeps them varied.
    if not prior:
        return ""
    bullets = "; ".join(compact_words(line, 10) for line in prior[-4:])
    return f" Others already {what}: {bullets}. Say yours differently — don't reuse their words, structure, or opener."


def greeting_line(persona: Persona, topic: str, others: list[str], max_words: int, prior: list[str]) -> str:
    return f"""One short opening line from {persona.name} before a group decision chat — hard limit: {max_words} words.{_audience_clause(others)}
These people already know each other. Treat this as an ongoing thread, not a meeting people are arriving at. Plain and brief. No arrival announcement, no slang check-in, no topic commentary, no emoji, no name prefix."""


def farewell_line(persona: Persona, scenario: Scenario, outcome: RunOutcome, others: list[str], max_words: int, prior: list[str]) -> str:
    if outcome.final_option and outcome.status in {"successful", "majority"}:
        chosen = scenario.option(outcome.final_option).name
        result = f"the group is going with {chosen}"
        if outcome.final_option == persona.preferred_option:
            tone = "You got your pick — genuine, pleased, brief."
        elif outcome.status == "majority":
            tone = "This wasn't your first choice — show mild acceptance without enthusiasm."
        else:
            tone = "You came around — brief and genuine, a little resigned or relieved depending on your personality."
    else:
        result = "the group couldn't land on a choice this time"
        tone = "No decision — name one specific thing blocking you, or how you feel leaving it unresolved. Don't just restate the impasse."
    audience = (f" Sign off to {others[0]} directly — this is a 2-person chat, not a group."
                if len(others) == 1 else "")
    return f"""The discussion just wrapped: {result}.
Write a short, casual sign-off from {persona.name} — hard limit: {max_words} words.
These are colleagues who know each other well.
{tone}{audience}
Don't open with the option name. One brief reaction or a sentence about what's next. You can say the name mid-sentence or use "it" or a short form.
No invented details. Casual, like leaving a text chat. No formal closers, no name prefix, no emoji."""


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
                snippet = compact_words(turn.text, int(cfg.utterances.response_target_max_words))
                return f'{turn.speaker_name} said: "{snippet}"'
    target_id = intent.addressee_id
    if not target_id:
        return ""
    for turn in reversed(state.turns):
        if turn.speaker_id == persona.id:
            continue
        if target_id and turn.speaker_id != target_id:
            continue
        snippet = compact_words(turn.text, int(cfg.utterances.response_target_max_words))
        return f'{turn.speaker_name} said: "{snippet}"'
    return ""


def _turn_behavior(persona: Persona, intent: MoveIntent) -> str:
    """Select compact, act-specific behavior directly from OCEAN traits."""
    t = persona.traits
    cues: list[str] = []
    if intent.act in {ActType.ASK, ActType.COMPARE}:
        cues.append("explore a less obvious trade-off" if t.openness >= 4 else "stay with the most practical known point")
        if t.conscientiousness >= 4:
            cues.append("pin down one concrete constraint")
    elif intent.act in {ActType.OBJECT, ActType.PUSH_BACK, ActType.REJECT}:
        cues.append("acknowledge the useful part before disagreeing" if t.agreeableness >= 4 else "challenge the weak point plainly")
        if t.neuroticism >= 3:
            cues.append("make the unresolved risk clear")
    elif intent.act in {ActType.REACT, ActType.SUPPORT, ActType.ANSWER}:
        cues.append("build on their point" if t.agreeableness >= 4 else "test their point instead of simply agreeing")
        if t.conscientiousness >= 4:
            cues.append("keep the reply tied to a concrete constraint")
    elif intent.act == ActType.PROPOSE_COMPROMISE:
        cues.append("move the group toward a decision" if _is_proactive(t) else "offer one modest next step")
        if t.agreeableness >= 4:
            cues.append("leave room for the holdout")
    else:
        cues.append("jump in directly" if t.extraversion >= 4 else "keep it reserved and economical")
        if t.neuroticism >= 3:
            cues.append("surface the concern before committing")
    if t.compromise_willingness < 0.25 and intent.act in {ActType.ACCEPT, ActType.VOTE, ActType.REJECT}:
        cues.append("hold the established position")
    if not cues:
        cues.append("make one practical contribution")
    return "; ".join(cues[:2])


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
    lines = [
        f"Speaker: {persona.name}",
        f"Lean: {lean_name}",
        f"For this turn: {_turn_behavior(persona, intent)}.",
    ]
    if intent.act == ActType.OPENING:
        lines.insert(1, f"Background: {persona.background}")
        lines.insert(2, f"Goal: {persona.private_goal}")
    elif intent.respond_to_turn is None and intent.act != ActType.PROPOSE_COMPROMISE:
        lines.insert(1, f"Personal stake: {compact_words(persona.private_goal, 16)}")
    if rt.already_said:
        recent_pts = "; ".join(f'"{compact_words(p, 10)}"' for p in rt.already_said[-2:])
        lines.append(f"Don't repeat these points you already made: {recent_pts}.")
    if lean != persona.preferred_option:
        old_name = state.scenario.option(persona.preferred_option).name
        lines.append(f"Originally leaning {old_name}; shifted to {lean_name}.")
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


_SELF_OPENER = re.compile(r"^\s*(?:i\b|i['’]|we\b|we['’]|our\b)", re.I)


def _opener_variety_hint(state: DialogueState, intent: MoveIntent) -> str:
    """Return a prompt nudge when recent turns have all opened with 'I'/'we'.
    Exempt commitment turns — 'I'll go with X' is expected and correct there."""
    if intent.act in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
        return ""
    recent = [t for t in state.turns if t.speaker_id != "moderator"][-3:]
    if sum(1 for t in recent if _SELF_OPENER.match(t.text)) >= 2:
        return "Recent turns all started with 'I' — lead from the other person's point, a reaction, a question, or a verb."
    return ""


def _concession_bridge(persona: Persona, option_name: str = "") -> str:
    opt = option_name or "this"
    if persona.traits.compromise_willingness < 0.5:
        return f"Give {opt} a reluctant yes and name one unresolved concern."
    return f"Accept {opt}, acknowledge it was not your first choice, and name one trade-off you can live with."


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
            return " Tie it to one concrete detail already in the point you're answering."
        if t.neuroticism >= 3:
            return " Back it, but name the one risk you're accepting."
    if intent.act == ActType.COMPARE:
        if _is_direct(t):
            return " State your preference plainly — no hedging."
        if t.neuroticism >= 3:
            return " Name the one risk that decides it for you."
    return ""


def _move_guidance(state: DialogueState, persona: Persona, intent: MoveIntent) -> str:
    t = persona.traits
    high_compromise = t.compromise_willingness >= 0.6
    face = _face_work(persona, intent)
    if intent.act == ActType.OPENING:
        common = " Name your pick and one personal reason, as briefly as you'd text it. Not a formal position statement. No 'though/but/however' clause."
        spoken_ids = {turn.speaker_id for turn in state.turns if turn.speaker_id != "moderator"}
        if any(p.id in spoken_ids and p.preferred_option == persona.preferred_option
               for p in state.personas if p.id != persona.id):
            common += " Someone already backed this — give your own distinct angle."
        if t.extraversion >= 4:
            return "Jump in — say what grabbed you." + common
        if _is_direct(t):
            return "Cut to the chase." + common
        return "Say what you'd go for." + common
    if intent.act == ActType.ACCEPT:
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name)
            return f"Agree to {opt_name} in plain first-person chat language, not formal ballot wording. No 'now' or trailing conditions. {bridge}"
        if _others_back(state, persona, intent.option_focus):
            return "Clearly agree to the option in plain first-person chat language. Keep it brief and don't echo the others."
        return "Clearly agree to the option in plain first-person chat language with one brief reason. No formal ballot wording or trailing conditions."
    if intent.act == ActType.VOTE:
        if _others_back(state, persona, intent.option_focus):
            chorus = " Others already voted this way — brief, don't echo them."
        elif any(t.act == ActType.VOTE for t in state.turns if t.speaker_id not in {persona.id, "moderator"}):
            chorus = " Others already voted — one short phrase is enough."
        else:
            chorus = ""
        focus = intent.option_focus[0] if intent.option_focus else None
        if focus and focus != persona.preferred_option:
            opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
            bridge = _concession_bridge(persona, opt_name)
            return f"Clearly choose {opt_name} in plain first-person chat language. No formal ballot wording, process commentary, or trailing conditions. {bridge}" + chorus
        return "Clearly choose the option in plain first-person chat language with one brief reason. No formal ballot wording or process commentary." + chorus
    if state.phase in {Phase.NARROWING, Phase.CONFIRMATION} or intent.act == ActType.REJECT:
        chorus = (" Others already backed this — brief or add a new angle."
                  if _others_back(state, persona, intent.option_focus) else "")
        if high_compromise:
            return "Back what the group is landing on." + chorus
        return "Say where you stand — hold out only if you mean it." + chorus + face
    if intent.act == ActType.SUPPORT and intent.option_focus and intent.moves_lean:
        focus = intent.option_focus[0]
        opt_name = state.scenario.option(focus).name if focus in state.scenario.option_ids else ""
        bridge = _concession_bridge(persona, opt_name)
        return f"You're warming up to this. {bridge}"
    if intent.act == ActType.ANSWER:
        return (
            "The only valid facts are those already in the card or shared_context. "
            "If the question asks for something the card does not state, say 'the card doesn't say' and pivot to a point the card does support. "
            "Never estimate, guess, or turn an unknown into a confident new fact. Never re-ask." + face
        )
    if intent.act == ActType.ASK:
        # Extract the attribute keys from the focused option(s) so the model can only ask
        # about things that are actually on the card, not implied/adjacent attributes.
        card_keys: list[str] = []
        for opt_id in (intent.option_focus or []):
            if opt_id in state.scenario.option_ids:
                card_keys.extend(state.scenario.option(opt_id).attrs.keys())
        attr_list = ", ".join(dict.fromkeys(card_keys)) if card_keys else "the attributes above"
        return (
            f"Ask about ONE specific card attribute — one of: {attr_list}. "
            "Do not ask about things not listed on the card (no weather, no proximity guesses, no service quality). "
            "Keep it casual." + face
        )
    by_act = {
        ActType.REACT: "Deal with the exact point just made. Say what it changes, confirms, or misses. Don't restart your option case." + face,
        ActType.COMPARE: "Reply to the local trade-off and say which one consideration matters more. Don't retell your background or summarize both options." + face,
        ActType.SUPPORT: "Build on the local point with one new reason or consequence. Don't repeat your biography or give a fresh option pitch." + face,
        ActType.OBJECT: "Name the specific problem in the local point. Don't restart your own case or use a generic acknowledgment." + face,
        ActType.PUSH_BACK: "Push back on the exact claim just made — not a general counterpoint. Don't open with 'I get that' or 'I hear you'." + face,
        ActType.PROPOSE_COMPROMISE: (
            "The group has covered the ground. Suggest directly that it's time to each pick one option — briefly explain your lean, then ask everyone to commit. No new trade-offs."
            if state.narrowing_called else
            "Connect two priorities already voiced and suggest one existing option as common ground. Don't add facts or simply restate your own preference."
        ) + face,
    }
    return by_act.get(intent.act, "Respond to the last point directly.")


def _alias_rule(state: DialogueState, intent: MoveIntent) -> str:
    aliases = short_alias_map(state.scenario.options)
    if intent.act == ActType.VOTE:
        focus_ids = intent.option_focus
        if focus_ids and focus_ids[0] in aliases:
            opt_name = state.scenario.option(focus_ids[0]).name if focus_ids[0] in {o.id for o in state.scenario.options} else ""
            return f"\n- Name your option clearly: full name or \"{aliases[focus_ids[0]]}\" — not just a letter."
        return "\n- Name the option clearly — full name or a recognizable shortening, not just a letter."
    # General discussion: show safe shortenings as reference, but encourage natural variation.
    mentioned = {opt for t in state.turns for opt in (t.act.option_refs if hasattr(t.act, 'option_refs') else [])}
    if mentioned:
        pairs = "; ".join(
            f"{o.name} = {aliases[o.id]}"
            for o in state.scenario.options if o.id in mentioned
        )
        return f"\n- Option shortenings: {pairs}. Use full name or shortening — vary naturally, don't always repeat the same one."
    return "\n- Name options by their full name or a recognizable word from the name."


def _verbosity_note(persona: Persona, max_words: int) -> str:
    t = persona.traits
    style = "Contractions OK. No semicolons, no em-dash between thoughts, no 'we should', no 'we need to', no formal transitions. One thought — no 'though/but/while' clause, nothing after a dash."
    if t.response_length >= 4:
        return f"Hard limit: {max_words} words, 1-2 sentences. Make the point — don't build a case. {style}"
    if t.response_length == 3:
        target = max(10, 2 * max_words // 3)
        return f"Hard limit: {target} words. One direct line. {style}"
    if t.response_length == 2:
        target = max(8, max_words // 3)
        return f"Hard limit: {target} words. One short clause — really stick to it. {style}"
    target = max(6, max_words // 4)
    return f"Hard limit: {target} words. A fragment — stop there. {style}"


def _collect_recent_frames(state: DialogueState, window: int = 4) -> list[str]:
    frames: list[str] = []
    recent = [t for t in state.turns if t.speaker_id != "moderator"][-window:]
    for turn in recent:
        frames.extend(classify_discourse_frames(turn.text))
    return frames


def _trailer_stance_hint(intent: MoveIntent) -> str:
    if intent.act == ActType.VOTE:
        return " For this turn stance must be vote."
    if intent.act == ActType.ACCEPT:
        return " For this turn stance must be accept."
    return ""


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
    elif intent.act == ActType.ASK:
        # Show attribute facts so the model can ask about something actually in the card.
        focused = "\n".join(f"- {_option_brief(o)}" for o in focus_options) if focus_options else "- none"
    else:
        focused = "\n".join(f"- {_option_name_only(o)}" for o in focus_options) if focus_options else "- none"
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    focus_str = ", ".join(intent.option_focus) if intent.option_focus else "-"
    guidance = _move_guidance(state, persona, intent)
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
        address_note = f" Address {effective_address} only if natural."
    elif derived_address:
        address_note = f" Can use {derived_address}'s name if it flows naturally."
    else:
        address_note = ""
    rt = state.runtimes[persona.id]
    frame_hint = recent_frame_hint(_collect_recent_frames(state), rt.discourse_frames)
    frame_line = f"\n{frame_hint}" if frame_hint else ""
    opener_hint = _opener_variety_hint(state, intent)
    if opener_hint:
        frame_line += f"\n{opener_hint}"
    if intent.option_focus and intent.act not in {ActType.VOTE, ActType.ACCEPT, ActType.REJECT}:
        focus_opt = intent.option_focus[0]
        if focus_opt in state.coverage:
            slot_hint_text = covered_slots_hint(focus_opt, state.coverage[focus_opt].covered_slots)
            if slot_hint_text:
                frame_line += f"\n{slot_hint_text}"
    alias_rule = _alias_rule(state, intent)
    ctx_line = ""
    if state.scenario.shared_context:
        ctx_line = "\nContext: " + "; ".join(compact_words(item, 12) for item in state.scenario.shared_context)
    verbosity = _verbosity_note(persona, max_words)
    if responding:
        job_block = (
            f"Reply to this exact point: {responding}\n"
            f"Your one job: {guidance}{address_note}\n"
            "Make the connection obvious without a generic acknowledgment. Don't restart your case or repeat your background."
        )
    else:
        job_block = f"Job: {guidance}{address_note}"
    return f"""Next message in a casual group chat. One short message — one point. Write like you'd text a friend. Don't open with "I get that", "I hear you", or "True, but".

Topic: {state.scenario.topic}
Options: {option_names}{ctx_line}

{card}

{job_block}
{verbosity}{frame_line}

Option facts (reference only):
{focused}

Recent chat:
{recent}

No name prefix. No semicolons in the message. Never write an option letter (A/B/C/D) in the message — only in the trailer. Card/context facts only — if a detail isn't in the card or shared_context, skip it entirely; never estimate, guess, or invent a logistics detail (distance, quality, availability, weather, or service not listed). Don't end with a question unless your act is ASK.{alias_rule}
End with: [act={intent.act.value}; opt=LETTER; stance=STANCE]. LETTER={opt_choices} or -; STANCE=vote|accept|object|reject|propose|neutral.{_trailer_stance_hint(intent)}"""


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
    "MISSING_COMMITMENT_TRAILER": "append a machine trailer [act=...; opt=LETTER; stance=vote/accept] at the end",
    "UNCLEAR_VOTE": "say the option name explicitly and commit without hedging; use stance=vote in the trailer",
    "UNCLEAR_ACCEPT": "say plainly that the option works for you — a reluctant yes ('can live with it', 'it works') is fine; append [act=accept; opt=LETTER; stance=accept]",
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
                f"\nDecision requirement: name {target} and visibly commit to it. "
                "Praising it, describing it, or promising to present it is not a choice. The trailer is metadata only."
            )
    response_block = ""
    responding = _responding_to_line(state, persona, intent)
    if responding:
        response_block = (
            f"\nKeep this as a direct reply to: {responding}. "
            "Make that connection clear without a generic acknowledgment or a fresh option pitch."
        )
    return f"""Rewrite {persona.name}'s line. Keep its grounded point and fulfill the routed act.
Behavior: {_turn_behavior(persona, intent)}.
Relevant options: {option_context}{recent_block}{response_block}
Line: {original_text}
Fix: {fixes}.{decision_block}
One natural line under {max_words} words; no prefix, quotes, invented facts, questions, or copied wording.
Then: [act={intent.act.value}; opt=LETTER; stance=STANCE] where LETTER={opt_choices} or - and STANCE=vote|accept|object|reject|propose|neutral."""
