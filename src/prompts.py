"""LLM-facing prompts for setup, moderator text, and utterance realization.

The environment controls facts, routing, addressees, and dialogue acts. The LLM
only realizes one natural message at a time. Generated messages must not contain
hidden metadata; public outcomes are parsed from visible text only.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable

from aliases import short_alias_map
from config_loader import cfg
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, Phase, RunOutcome, Scenario, STANCE_ACCEPTABLE, STANCE_DISLIKED, STANCE_NEUTRAL, STANCE_PREFERRED, STANCE_REJECTED, ThreadStatus, ThreadType
from utils import compact_words


# Fixed neutral line that opens general discussion after the option board.
# Deliberately criteria-free: the setup must not steer the first turns toward
# predefined decision dimensions.
NEUTRAL_OPENING_LINE = "Let's discuss which option fits best overall."


def _schema(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _option_brief(option: OptionCard) -> str:
    return option.public_line()


def _option_cards(options: Iterable[OptionCard]) -> str:
    # The prompt receives exactly the complete facts printed on the public
    # board. No hidden fourth attribute or clipped concern may influence a sim.
    return "\n".join(f"- {option.prompt_card()}" for option in options)


def setup_scenario(topic: str, n: int) -> str:
    labels = list(cfg.scenario.option_labels)
    schema = {
        "scenario": {
            "shared_context": [
                "2-3 public facts every participant knows before the discussion"
            ],
            "options": [
                {
                    "id": label,
                    "name": "specific realistic option name",
                    "short_name": "1-2 recognizable words copied from name",
                    "attrs": {
                        "attribute_name": "stable value",
                        "another_attribute": "stable value",
                    },
                    "upside": "specific benefit",
                    "concern": "stable objection people could raise",
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
- Every option needs a short_name: a concise natural alias of 1-{cfg.scenario.short_alias_max_words} words copied from its name, unique across options.
- Every option must have {cfg.scenario.public_attr_min}-{cfg.scenario.public_attr_max} concrete attributes with stable values. Choose attributes that are natural for this topic.
- Do not use unknown, TBD, live availability, current weather, or facts that need internet lookup.
- Options should expose real trade-offs, not one obvious winner.
- shared_context is the public source of truth: facts known by ALL participants before the discussion. Never put private-only or single-person information there.
- If shared_context states a hard numeric limit, every option must satisfy it — never create an option that violates a stated cap.
- If shared_context mentions the group size, it must say exactly {n}.

Return JSON only:
{_schema(schema)}"""


def alias_repair(
    options_needing: list[tuple[str, str, str]],
    used_aliases: list[str],
) -> str:
    """Small alias-only repair call (todo_validation item 3).

    Receives only the affected options (id, full name, rejected alias) and the
    aliases already taken by other options. The original option board is never
    regenerated; only concise replacement aliases are requested.
    """
    rows = "\n".join(
        f"- {oid}: name \"{name}\"" + (f" (rejected alias: \"{rejected}\")" if rejected else "")
        for oid, name, rejected in options_needing
    )
    taken = ", ".join(f'"{a}"' for a in used_aliases) or "none"
    example = {"aliases": {oid: "words copied from that option's name" for oid, _n, _r in options_needing}}
    return f"""Provide a concise short alias for each option below.

Options needing an alias:
{rows}
Aliases already taken by other options: {taken}

Rules:
- Each alias uses 1-{cfg.scenario.short_alias_max_words} words COPIED from that option's own name (trivial singular/plural is fine). Never invent new words, never abbreviate.
- At least {cfg.scenario.short_alias_min_chars} characters; pick the most recognizable words, not simply the first ones.
- Aliases must be unique and must differ from the taken aliases above.
- Do not reuse a rejected alias.

Return JSON only:
{_schema(example)}"""


def setup_personas(
    topic: str,
    n: int,
    trait_rows: list[dict],
    required_preferences: dict[str, str],
    options_json: list[dict],
    shared_context: list[str],
    hard_blocker_id: str | None = None,
) -> str:
    names_by_id = {row["id"]: row.get("name", row["id"]) for row in trait_rows}
    # Manual participant profiles may fix persona fields; tell the LLM to keep
    # them verbatim so generated fields stay consistent with them.
    fixed_field_rule = (
        "\n- If a trait row already contains background, private_goal, or age, copy that field exactly "
        "and keep the other fields consistent with it, including age plausibility."
        if any(row.get("background") or row.get("private_goal") or row.get("age") for row in trait_rows)
        else ""
    )
    preference_lines = "\n".join(
        f"- {pid} ({names_by_id.get(pid, pid)}): {option_id}"
        for pid, option_id in sorted(required_preferences.items(), key=lambda item: int(item[0][1:]))
    )
    schema = {
        "participants": [
            {
                "id": "p1",
                "name": "exact name from trait row",
                "age": 28,
                "background": "one sentence explaining the person's angle on this decision",
                "private_goal": "what they personally want from the decision",
                "preferred_options": ["A"],
                "option_stances": [
                    {"option": "A", "rank": 5, "reason_for": "short grounded pro", "reason_against": ""},
                    {"option": "B", "rank": 3, "reason_for": "", "reason_against": ""}
                ],
                "rejection": None,
                "rejection_reason": "",
            }
        ]
    }
    if hard_blocker_id:
        blocker_name = names_by_id.get(hard_blocker_id, hard_blocker_id)
        blocker_pref = required_preferences.get(hard_blocker_id, "their assigned option")
        hard_blocker_rules = (
            f"\n- {hard_blocker_id} ({blocker_name}) is this group's ONE exclusive hard blocker: "
            f"give {blocker_pref} rank 5, and give EVERY other option rank 1 with a short grounded "
            "reason_against each (or one clear exclusive requirement that explains all of them). "
            "Their background and private_goal must state that one non-negotiable requirement that "
            f"only {blocker_pref} satisfies — one plausible story, politely held. "
            "You may also set rejection to the most central alternative with a matching rejection_reason."
            "\n- Every other participant must remain movable: rejection null, no rank-1 options."
        )
    else:
        hard_blocker_rules = (
            "\n- For agreeableness=1 only, you may set one grounded rejection if an option conflicts "
            "with their background/goal. That rejection is a hard blocker."
            "\n- For all other participants, rejection must be null."
        )
    context_lines = "\n".join(f"- {item}" for item in shared_context) or "- none"
    return f"""Create {n} simulated users for an option-grounded group decision.

Topic: {topic}
Shared context (public facts every participant knows):
{context_lines}
Options:
{json.dumps(options_json, ensure_ascii=False, indent=2)}

Use these trait rows exactly. Traits are 1-5 OCEAN scores.
{json.dumps(trait_rows, ensure_ascii=False, indent=2)}

Initial primary preference assignment. preferred_options[0] MUST match this exactly:
{preference_lines}

Rules:
- Use the exact id and name from each trait row.
- Assign a plausible age between 18 and 75 unless age is already fixed in the trait row.
- The background/private_goal must be plausible for that age. Use soft age bands:
  * 18-22: student, apprentice, trainee, early job, shared flat/parents, no spouse/kids/mortgage/senior role.
  * 23-35: student or early/mid career, partner possible, young family possible only from late 20s onward.
  * 36-55: established career/family/home routines are plausible.
  * 56-72: senior career, older children, retirement planning, formal habits are plausible; do not make them teenagers/apprentices.
- Do not create absurd biographies: no 19-year-old married parent with two kids, no 21-year-old senior manager with a mortgage, no 25-year-old with 20 years of professional experience.
- preferred_options is the person's initial private preference, not a final vote. Add at most one secondary acceptable option if it fits.
- Also provide option_stances for EVERY option, using discrete rank: 5=preferred, 4=acceptable, 3=neutral/untested, 2=disliked but negotiable, 1=rejected/hard blocked.
- The assigned primary preference must have rank 5. A secondary preferred option, if any, should have rank 4.
- Keep most non-preferred options neutral or acceptable. Only give rank 2/1 when the option clearly conflicts with the person's background/goal.
- Give short reason_for/reason_against only where useful; leave neutral reasons empty. Do not write long explanations.
- A participant with agreeableness=1 must have exactly one preferred option (no secondary).
- Participants want a workable group decision. High openness/agreeableness means easier compromise; low agreeableness means more resistance.{hard_blocker_rules}
- background and private_goal must be one sentence each, specific to this topic, grounded in the option cards/shared context, and age-plausible.
- Backgrounds, private goals, and constraints must fit the shared context where relevant and must never contradict it (group size, hard caps, timing, or other public facts).
- For participants with agreeableness above 1, phrase needs as preferences ("prefers", "values", "cares most about"), never as absolute constraints ("cannot", "must", "refuses", "allergic", "strictly").
- background and private_goal must be consistent with the participant's assigned primary preference: the goal should explain why they would initially lean toward that option, and must never state a need that the preferred option's card explicitly fails to meet.{fixed_field_rule}

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
    lines.append(NEUTRAL_OPENING_LINE)
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
    focus = _option_names(state, focus_options or [])
    focus_line = f"\nFocus options: {focus}" if focus else ""
    candidate_line = f"\nCurrent likely common ground: {candidate_name}" if candidate_name else ""
    target = target_name or "the group"
    action = requested_action or "move the decision forward with one concrete next step"
    return f"""You are the neutral moderator of a casual group decision chat.
Verbalize the intervention below in your own words — when to intervene, whom to
address, and what to accomplish have already been decided.

Why now: {reason}
Address: {target}
Do: {action}{candidate_line}{focus_line}

Recent chat:
{recent}

Write one short progress nudge, under {cfg.utterances.word_budgets.moderator} words.
Follow the requested action exactly — if it says not to name an option, name none.
Do not vote, decide, add facts, or repeat the option board. Sound like a person in
the chat, not a script: never dictate a quoted reply template, and vary your
phrasing between interventions. No speaker prefix. One sentence only."""


def moderator_closure_line(outcome: RunOutcome, scenario: Scenario, state: DialogueState) -> str:
    """Deterministic status-correct social closure.

    The outcome is already final. A generated paraphrase can accidentally turn
    a majority into apparent unanimity or announce a winner for an unresolved
    result, so closing wording is kept deterministic and participant-facing.
    """
    if outcome.status == "unresolved" or not outcome.final_option:
        return "We haven't reached a majority, so we'll leave this unresolved for now."

    final = scenario.option(outcome.final_option).name
    if outcome.status == "successful":
        return f"We've all agreed on {final}, so that's the group decision."

    votes = outcome.metadata.get("visible_votes", {}) if outcome.metadata else {}
    holdouts = [
        persona.name for persona in state.personas
        if votes.get(persona.id, state.runtimes[persona.id].explicit_vote) != outcome.final_option
    ]
    if holdouts:
        names = ", ".join(holdouts)
        return f"{final} has the majority; {names} still preferred another option."
    return f"{final} has the majority, so that's the group decision."


def _stance_summary(
    state: DialogueState,
    persona: Persona,
    restrict_to: set[str] | None = None,
    skip_bare_current: str | None = None,
) -> str:
    """Compact private-stance line; with ``restrict_to`` only the options the
    current move actually touches are listed (item 3: no full stance maps when
    one or two stances matter). ``skip_bare_current`` drops a reason-less
    preferred entry for that option — it would only restate the current pick."""
    rt = state.runtimes[persona.id]
    aliases = short_alias_map(state.scenario.options)
    labels = {
        STANCE_PREFERRED: "preferred",
        STANCE_ACCEPTABLE: "acceptable",
        STANCE_NEUTRAL: "neutral",
        STANCE_DISLIKED: "disliked",
        STANCE_REJECTED: "rejected",
    }
    parts = []
    for oid in state.scenario.option_ids:
        if restrict_to is not None and oid not in restrict_to:
            continue
        rank = rt.rank(oid)
        if rank == STANCE_NEUTRAL:
            continue
        reason = rt.reason_for(oid) if rank >= STANCE_ACCEPTABLE else rt.reason_against(oid)
        if oid == skip_bare_current and rank >= STANCE_PREFERRED and not reason:
            continue
        text = f"{aliases.get(oid, oid)}={labels.get(rank, str(rank))}"
        if reason:
            text += f" ({compact_words(reason, 7)})"
        parts.append(text)
    return "; ".join(parts[:5]) or "mostly neutral"


# One concise semantic requirement per dialogue act (todo_new item 4): what the
# realized line must MEAN, never how it must be phrased. Vote turns get their
# requirement from the decision instruction, which also carries the
# controller's target and the parser-visible commitment vocabulary.
_ACT_REQUIREMENTS = {
    ActType.OPENING: (
        "State your current favorite and one grounded reason; a tiny chat greeting "
        "is fine. Do not make it sound like a final vote."
    ),
    ActType.SUPPORT: (
        "Give real support: one clearly positive reason or consequence for the focused "
        "option. You may briefly acknowledge a concern first, but your supportive "
        "position must stay clear. Do not add a new factual feature to justify it."
    ),
    ActType.CONCERN: (
        "Raise one concrete downside, risk, unknown, or unmet need — an actual "
        "objection, not a vague cautious remark. A missing detail must be described "
        "as unknown, not filled in with a plausible external fact."
    ),
    ActType.ASK: (
        "Ask exactly one genuine, answerable question about the intended issue. Do not "
        "bundle several unrelated questions or dress a statement up as a question."
    ),
    ActType.ANSWER: (
        "Answer the question first, directly. If it asks for information that is not in "
        "the option cards or shared context, say plainly that we don't know that here. "
        "A short explanation may follow. Do not replace the answer with a generic "
        "reaction or a counter-question."
    ),
    ActType.COMPARE: (
        "Make both options identifiable (short names or clear references are fine, "
        "anywhere in the message) and base the trade-off on facts explicitly listed on "
        "their cards. Reasonable consequences are fine when phrased as judgments, not "
        "as invented factual features. If a needed detail is missing, say it is unknown."
    ),
    ActType.COMMENT: (
        "This is a light comment, not an argument: acknowledge, interpret, or name a "
        "priority in one beat. You do not have to support, reject, compare, or ask "
        "anything."
    ),
    ActType.COMPROMISE: (
        "Propose exactly ONE of the existing options as the possible common ground; a "
        "visible condition on it is fine. Do not invent blends, split plans, or "
        "combinations of options."
    ),
    ActType.PROCESS: (
        "Make one concrete procedural suggestion about how the group should continue. "
        "Keep it short and socially natural. Do not cast your own final vote in this "
        "line unless explicitly asked."
    ),
    ActType.VOTE: "Make your chosen option and your commitment unambiguous.",
}


def _length_instruction(min_words: int, max_words: int, act: ActType) -> str:
    """Turn a numeric budget into a clear surface-length instruction.

    Verbosity still reaches generation only through the assigned range.  The
    wording is stronger at the extremes so the trait becomes visible in the
    actual chat instead of only in logged budgets.
    """
    if max_words <= 10 and act is not ActType.VOTE:
        return (
            f"Keep it to one brief chat-style sentence of about {min_words}-{max_words} words. "
            "Make one point only; do not add a second explanation."
        )
    if max_words >= 18 and act not in {ActType.VOTE, ActType.ASK}:
        return (
            f"Develop the point in about {min_words}-{max_words} words. Use enough detail to reach "
            f"roughly {min_words} words: give the main reason and one useful consequence, contrast, "
            f"or qualification. Two natural sentences are fine; do not exceed {max_words + 2} words "
            "and do not collapse it into a terse one-liner."
        )
    return (
        f"Aim for {min_words}-{max_words} words in one natural chat message; do not exceed "
        f"{max_words + 2} words."
    )


def sim_utterance(
    *,
    persona: Persona,
    state: DialogueState,
    intent: MoveIntent,
    recent_lines: list[str],
    focus_options: list[OptionCard],
    addressee_name: str | None,
    max_words: int,
    min_words: int = 1,
) -> str:
    aliases = short_alias_map(state.scenario.options)
    cards = _option_cards(focus_options or state.scenario.options)
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    current = state.runtimes[persona.id].top_option() or persona.preferred_option
    current_name = aliases.get(current, state.scenario.option(current).name if current in state.scenario.option_ids else "undecided")
    # Stance restricted to the options this move touches (item 3): the current
    # lean plus the focus options; the full map appears only without a focus.
    # The current pick itself is listed only when it carries a stored reason —
    # a bare "X=preferred" would duplicate "current pick X" (todo_prompt item 8).
    relevant = {current, *intent.option_focus} if intent.option_focus else None
    stance = _stance_summary(state, persona, restrict_to=relevant, skip_bare_current=current)
    initial_name = aliases.get(persona.preferred_option, persona.preferred_option)
    initial_note = f" (initial preference {initial_name})" if persona.preferred_option != current else ""
    stance_note = f"; stance: {stance}" if stance != "mostly neutral" else ""
    required_name = aliases.get(intent.required_vote, intent.required_vote) if intent.required_vote else ""
    old_required = intent.old_preference or current
    old_name = aliases.get(old_required, old_required) if old_required else current_name
    allowed_reason = intent.allowed_reason or "the listed facts and visible support make it workable"
    blocked = ""
    rejected_options = state.runtimes[persona.id].rejected_options()
    if rejected_options and len(rejected_options) >= len(state.scenario.option_ids) - 1:
        # Exclusive hard blocker: only the current pick is acceptable at all.
        only = current if current in state.scenario.option_ids else persona.preferred_option
        reason = state.runtimes[persona.id].reason_for(only)
        blocked = (
            f"\nHard constraint: only {aliases.get(only, only)} is acceptable to them"
            + (f" ({reason})" if reason else "")
            + ". They reject every other option; do not accept or vote for any other option, however politely."
        )
    elif rejected_options:
        oid = sorted(rejected_options)[0]
        reason = state.runtimes[persona.id].reason_against(oid)
        blocked = f"\nHard blocker: they strongly reject {aliases.get(oid, oid)}" + (f" because {reason}" if reason else "") + ". Do not accept or vote for that option."
    target_block = _target_block(state, intent, len(recent_lines))
    address = f"\nAddress {addressee_name} if it sounds natural." if addressee_name else ""
    context = "; ".join(compact_words(item, 14) for item in state.scenario.shared_context) if state.scenario.shared_context else "none"
    params = persona.sim_params
    decision_instruction = ""
    if intent.act == ActType.VOTE:
        # Semantic requirement, no phrase menu (todo_prompt item 7): the parser
        # vocabulary is broad; a failed line still gets one example-backed
        # repair. The avoid-lists below stay — they are adaptive anti-chorus
        # evidence from this round, not a palette.
        if intent.required_vote:
            target_clause = f"commit clearly to {required_name} and no other option"
        else:
            target_clause = "commit clearly to exactly ONE option"
        decision_instruction = (
            f"\nFor this decision turn, {target_clause}, in plain words of your own that state the "
            "commitment directly with an explicit first-person commitment verb (vote, choose, pick, "
            "back, commit, or settle) and the option name right next to it. Do not merely say the option "
            "seems best. The sentence may otherwise take any shape. One short reason may follow. "
            "No hedging, no 'leaning', no conditions, no question after it."
        )
        if intent.avoid_phrases:
            forbidden = "; ".join(f"'{p}'" for p in intent.avoid_phrases)
            decision_instruction += f"\nEarlier speakers already used these phrasings — do NOT use them: {forbidden}."
        if intent.avoid_reasons:
            used = "; ".join(f"'{r}'" for r in intent.avoid_reasons[:3])
            decision_instruction += f"\nEarlier voters already gave these justifications — give a DIFFERENT reason of your own, in your own words: {used}."
        if intent.allow_vote_change and intent.old_preference and intent.required_vote and intent.old_preference != intent.required_vote:
            # The bridge is REQUIRED, matching validation (switch_bridge_ok):
            # an unexplained flip is rejected, so the prompt must not present
            # the earlier pick as optional (item 9).
            # required_name is already fixed by the commit clause above; only
            # the earlier pick and the allowed reason are new here (item 8).
            decision_instruction += (
                f"\nThis is a genuine compromise switch away from {old_name}. Your line MUST make the "
                f"change of mind visible — mention {old_name} briefly or otherwise make the movement "
                f"explicit — and give this reason in your own words: {allowed_reason}. "
                "Use one natural sentence or two short clauses; vary the wording, not the required content. "
                "Do not add any other factual reason, condition, or pressure language."
            )
        elif intent.required_vote:
            decision_instruction += (
                f"\nThis is a confirmation, not a switch: one short grounded reason may paraphrase: {allowed_reason}. "
                "Do not mention an earlier preference unless it is a different option."
            )
    continuation_note = ""
    if intent.continuation:
        continuation_note = (
            "\nThis is a quick follow-up to YOUR OWN previous message (you spoke last): one short "
            "add-on thought. Stay on the same point and option as your last message — do not switch "
            "to a different option or open a new issue. Do not repeat or rephrase anything you just "
            "said, do not re-ask the same question, and do not address the same person with the same "
            "request again."
        )
    settled_unknowns = ""
    # Settled-issue suppression derives from thread history (threads are the
    # single owner of issue repetition): a concern/blocker thread that ended
    # resolved or stale had its raise/response arc play out already.
    settled_issues = sorted({
        thread.issue_key.replace("_", " ")
        for thread in state.threads.values()
        if thread.thread_type in (ThreadType.CONCERN, ThreadType.BLOCKER)
        and thread.status in (ThreadStatus.RESOLVED, ThreadStatus.STALE)
        and thread.issue_key
        and thread.issue_key != "general"
        and not thread.issue_key.startswith("sig:")
    })
    if settled_issues:
        settled_unknowns = (
            f"\nAlready raised and settled in this chat: {', '.join(settled_issues[:5])}. "
            "Do not re-raise these as new concerns or questions; argue from the listed facts instead."
        )
    # Verbosity reaches the prompt only as this numeric word range; the range
    # itself (not a persona label) controls how much the turn develops.
    length_note = _length_instruction(min_words, max_words, intent.act)
    # At most one lexical variation note reaches a turn (policy picks the most
    # relevant pattern, item 8); the tail-question note is separate flow control.
    style_notes = ""
    if intent.suppress_tail_question:
        style_notes += "\n- Enough questions are already open; end with a statement, not a question."
    if intent.suppress_name_prefix:
        style_notes += "\n- Several recent messages open with a name; just reply directly this time."
    if intent.suppress_option_opening:
        style_notes += "\n- Several recent messages open with an option name; lead with your point this time (naming options mid-sentence is fine)."
    if intent.suppress_i_opening:
        style_notes += "\n- Many recent messages start with 'I …'; open with the topic, the other person's point, or a fact this time."
    if intent.suppress_we_opening:
        style_notes += "\n- Many recent messages start with 'We …'; open with the point itself this time."
    if intent.vary_opening:
        style_notes += "\n- The last few messages all opened with the same word; start this one differently."
    if intent.avoid_pattern in {"concede_but", "worry_but", "tradeoff_but"}:
        style_notes += "\n- The last few messages all used a concession-then-objection shape; state your point directly this time."

    # Background/private goal shape CONTENT choices (which reasons feel
    # personal); moves whose content is prescribed elsewhere — answers (the
    # question), procedure beats, and votes (the controller's target/reason) —
    # skip them (todo_prompt item 8).
    persona_line = (
        ""
        if intent.act in {ActType.ANSWER, ActType.PROCESS, ActType.VOTE}
        else f"\nBackground: {compact_words(persona.background, 14)}; goal: {compact_words(persona.private_goal, 14)}."
    )
    return f"""Write one natural chat message for {persona.name}.

Voice: age {persona.age}; {persona.speech_style}. Directness {_scale_1_5(params.directness)}/5 (5 = blunt plain wording, 1 = soft tentative wording). Stubbornness {_scale_1_5(params.stubbornness)}/5 (5 = defends the current stance firmly and concedes slowly, 1 = concedes easily).{persona_line}
Your stance: current pick {current_name}{initial_note}{stance_note}.{blocked}

Move: {intent.act.value}. {_ACT_REQUIREMENTS.get(intent.act, "")}
This turn: {intent.reason}{continuation_note}{target_block}{address}{decision_instruction}

Topic: {state.scenario.topic}
Shared context: {context}
Closed-world grounding rule: the shared context and option cards below are the complete factual world for this chat. You may naturally repeat exact listed values such as prices, durations, capacities, or distances when they help the point. Never alter those values or add exact times, locations, amenities, capacities, schedules, availability, guarantees, reliability claims, service features, or outside conditions unless they are explicitly written below. Do not use general knowledge to complete a plausible detail. If a needed detail is unlisted, omit it or say naturally that we do not know it here.
Allowed facts:
{cards}

Recent chat:
{recent}

Output: exactly one message wrapped in <utterance></utterance> tags, like <utterance>your message</utterance>. Nothing outside the tags. No speaker prefix, no bullets/metadata inside. {length_note} Match the speech style and age naturally. Add something new instead of restating points already made. Refer to options naturally — full name, short name, or a clear reference, anywhere in the message, preferably not at the start of a sentence. Use only the listed option facts and shared context. Exact values printed on a card may be repeated naturally when relevant. Treat them as closed-world facts: when information is not explicitly given, omit it, say naturally that it is unknown here, or ask about it. Never alter a listed value or invent a plausible exact time, place, capacity, amenity, availability, guarantee, product/service capability, or external condition.{settled_unknowns}{style_notes}"""


def _scale_1_5(value: float) -> int:
    """Map a [0,1] simulator parameter onto the 1-5 scale shown in prompts."""
    return max(1, min(5, round(1 + 4 * float(value))))


# Commitment-form examples for the REPAIR path only (a first attempt already
# failed to parse there, so concrete parser-visible examples are justified).
# Keyed by parsing._PHRASE_FAMILIES labels so families already used this round
# (intent.avoid_phrases) drop out of the menu.
_COMMIT_FORM_EXAMPLES = {
    "I vote for": "I vote for X",
    "gets my vote": "X gets my vote",
    "works for me": "X works for me",
    "I'm going with": "I'm going with X",
    "I can live with": "I can live with X",
    "my pick is": "my pick is X",
}


def repair_utterance(
    *,
    original_text: str,
    issues: list,
    persona: Persona,
    state: DialogueState,
    recent_lines: list[str],
    intent: MoveIntent,
    max_words: int,
    min_words: int = 1,
) -> str:
    """Targeted, meaning-preserving repair request (todo_validation item 10).

    ``issues`` are ValidationIssue objects: the prompt receives the exact
    explanations and offending spans, not opaque codes. Everything the
    assessment did not flag must be preserved — intended move, option focus,
    target, stance, style, and commitment target.
    """
    issue_codes = [issue.code for issue in issues]
    focus = [state.scenario.option(o) for o in intent.option_focus if o in state.scenario.option_ids]
    cards = _option_cards(focus or state.scenario.options)
    recent = "\n".join(recent_lines[-3:]) if recent_lines else "(no recent turns)"
    aliases = short_alias_map(state.scenario.options)
    required_name = aliases.get(intent.required_vote, intent.required_vote) if intent.required_vote else ""
    current = state.runtimes[persona.id].top_option() or persona.preferred_option
    old_required = intent.old_preference or current
    old_name = aliases.get(old_required, old_required) if old_required else "your earlier pick"
    allowed_reason = intent.allowed_reason or "the listed facts make it workable"
    clear_commit = ""
    if intent.required_vote and intent.act == ActType.VOTE:
        switch_note = (
            f" This is a switch away from {old_name}: mention {old_name} briefly or otherwise make "
            "the change of mind explicit."
            if intent.old_preference and intent.old_preference != intent.required_vote
            else " This is not a switch; do not mention an earlier preference."
        )
        clear_commit = (
            f" The line MUST visibly commit to {required_name} and no other option."
            f"{switch_note} Use only this allowed reason, paraphrased naturally: {allowed_reason}. "
            "State the commitment plainly (for example 'I vote for X' or 'X works for me'). "
            "No hedging, no conditions, no question after it."
        )
    elif intent.act == ActType.VOTE:
        # Offer only commitment forms not yet used this round / by this speaker,
        # in shuffled order, so repaired vote turns stop converging on one
        # fixed menu (I19).
        fresh = [ex for label, ex in _COMMIT_FORM_EXAMPLES.items() if label not in intent.avoid_phrases]
        forms = fresh or list(_COMMIT_FORM_EXAMPLES.values())
        random.shuffle(forms)
        menu = ", ".join(f"'{f}'" for f in forms[:4])
        clear_commit = (
            f" The line MUST contain an explicit commitment to exactly one option — for example: {menu}. "
            "Add one short reason in your own words. No hedging, no 'even if', no 'leaning', "
            "and no question after it."
        )
    required_focus = ""
    if intent.option_focus and "MISSING_REQUIRED_OPTION_FOCUS" in issue_codes:
        required_focus = f" Mention and discuss Option {intent.option_focus[0]} explicitly."
    if intent.required_vote and "REQUIRED_VOTE_MISMATCH" in issue_codes:
        required_focus += f" Your previous line committed to the wrong option; commit to {required_name} instead."
    if "ANSWER_DOES_NOT_ADDRESS_QUESTION" in issue_codes:
        required_focus += " The line dodged the question it was asked; answer that question first, directly."
    if "CONTINUATION_REPEATS" in issue_codes:
        required_focus += " The follow-up repeated your previous message; add one genuinely new thought instead."
    if "WRONG_OPTION_FOCUS" in issue_codes and intent.option_focus:
        focus_name = aliases.get(intent.option_focus[0], intent.option_focus[0])
        required_focus += f" The line was about the wrong option; keep the same move but make it about {focus_name}."
    malformed = ""
    if "MALFORMED_UTTERANCE" in issue_codes:
        malformed = (
            " The line is an incomplete fragment (a lead-in with no content, or a lone subordinate clause). "
            "Write the complete short thought it was building toward — one clear point, still brief."
        )
    if "CONTINUATION_TOPIC_JUMP" in issue_codes:
        malformed += (
            " The follow-up jumped to a different option than your previous message; stay on the same "
            "option and point you just made."
        )
    if "HYBRID_COMPROMISE" in issue_codes:
        malformed += (
            " The line combined two options into one plan; propose exactly ONE existing option "
            "(a condition on it is fine), not a blend."
        )
    bridge = ""
    if "UNBRIDGED_SWITCH" in issue_codes:
        if intent.old_preference and intent.required_vote and intent.old_preference != intent.required_vote:
            # clear_commit above already states the commitment, movement, and
            # allowed reason; only name the specific defect here (item 11).
            bridge = (
                f" The previous line switched away from {old_name} without making that movement visible — "
                "fix exactly that."
            )
        else:
            bridge = (
                f" Commit directly to {required_name or 'the target option'} with one short grounded reason. "
                "Do not mention an earlier preference because this is not a switch."
            )
    params = persona.sim_params
    target_block = _target_block(state, intent, len(recent_lines[-3:]))
    requirement = _ACT_REQUIREMENTS.get(intent.act, "")
    problem_lines = "\n".join(
        f"- {issue.code}"
        + (f": {issue.explanation}" if issue.explanation else "")
        + (f" (offending words: \"{issue.span}\")" if issue.span else "")
        for issue in issues
    ) or "- (none listed)"
    length_note = _length_instruction(min_words, max_words, intent.act)
    return f"""Repair this generated chat line — keep its intended move, fix only the problems.
Preserve everything not flagged below: the option you are talking about, whom you address,
your stance and commitment target, and your speaking style.

Speaker: {persona.name} — {persona.speech_style}; directness {_scale_1_5(params.directness)}/5, stubbornness {_scale_1_5(params.stubbornness)}/5.
Move: {intent.act.value}. {requirement}{target_block}
Original line: {original_text}
Problems:
{problem_lines}
Closed-world rule: only the option facts below and the shared context are factual. Preserve and naturally reuse exact listed values when useful. Do not alter them or add exact times, places, capacities, amenities, availability, guarantees, capabilities, or outside conditions that are not listed. Remove an unsupported detail, or replace it with a natural phrase such as “we don't know that here”; do not append a robotic disclaimer to an otherwise complete sentence.
Allowed option facts:
{cards}
Recent chat:
{recent}

{length_note} Wrap exactly one natural chat line in <utterance></utterance> tags with nothing outside them. No speaker prefix. Preserve only listed facts; replace any unlisted factual detail with an explicit unknown. Avoid generic filler.{clear_commit}{required_focus}{malformed}{bridge} Do not append metadata, JSON, or bracketed labels inside the tags."""



def _option_names(state: DialogueState, ids: list[str]) -> str:
    aliases = short_alias_map(state.scenario.options)
    names = [aliases.get(option_id, option_id) for option_id in ids if option_id in state.scenario.option_ids]
    return ", ".join(names)


def _recent_chat(state: DialogueState, limit: int) -> str:
    rows = [f"{t.speaker_name}: {t.text}" for t in recent_turn_window(state, limit)]
    return "\n".join(rows) if rows else "(none yet)"


def recent_turn_window(state: DialogueState, limit: int) -> list:
    """Last ``limit`` turns shown inside sim prompts.

    The moderator's opening board turn is setup scaffolding: its content is
    already carried by the Allowed facts and Shared context sections, so it is
    never repeated as chat history (item 3)."""
    turns = [t for t in state.turns if not (t.speaker_id == "moderator" and t.phase is Phase.OPENING)]
    return turns[-limit:] if limit > 0 else []


def _target_block(state: DialogueState, intent: MoveIntent, recent_count: int) -> str:
    """Respond-to instruction without duplicating the recent-chat block (item 3).

    A target inside the shown recent window is pointed at, not re-quoted in
    full; only an older target gets its own compact quote."""
    if intent.respond_to_turn is None:
        return ""
    target = next((t for t in state.turns if t.index == intent.respond_to_turn), None)
    if target is None:
        return ""
    window = recent_turn_window(state, recent_count)
    if window and target.index >= window[0].index:
        if target.index == window[-1].index:
            return f"\nRespond to {target.speaker_name}'s last message in the recent chat."
        pointer = compact_words(target.text, 8)
        return f"\nRespond to {target.speaker_name}'s message in the recent chat (\"{pointer}\")."
    text = compact_words(target.text, int(cfg.utterances.response_target_max_words))
    return f"\nRespond to this earlier point — {target.speaker_name}: {text}"
