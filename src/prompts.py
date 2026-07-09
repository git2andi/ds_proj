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
from models import ActType, DialogueState, MoveIntent, OptionCard, Persona, RunOutcome, Scenario, STANCE_ACCEPTABLE, STANCE_DISLIKED, STANCE_NEUTRAL, STANCE_PREFERRED, STANCE_REJECTED
from parsing import unused_commitment_phrases
from utils import compact_words


# Fixed neutral line that opens general discussion after the option board.
# Deliberately criteria-free: the setup must not steer the first turns toward
# predefined decision dimensions.
NEUTRAL_OPENING_LINE = "Let's discuss which option fits best overall."


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


def setup_personas(
    topic: str,
    n: int,
    trait_rows: list[dict],
    required_preferences: dict[str, str],
    options_json: list[dict],
    shared_context: list[str],
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
- Participants want a workable group decision. High openness/agreeableness means easier compromise; low agreeableness means more resistance.
- For agreeableness=1 only, you may set one grounded rejection if an option conflicts with their background/goal. That rejection is a hard blocker.
- For all other participants, rejection must be null.
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
    candidate = candidate_name or "no single option yet"
    focus = _option_names(state, focus_options or []) or "not specified"
    target = target_name or "the group"
    action = requested_action or "move the decision forward with one concrete next step"
    public = _public_state_summary(state)
    return f"""You are the neutral moderator of a casual group decision chat.

Use MUCA-style control: decide what to ask, when to intervene, and who to address.
Write one short progress nudge, under {cfg.utterances.word_budgets.moderator} words.
Do not vote, do not decide, do not add facts, and do not repeat the option board.
Sound like a person in the chat, not a script: never dictate a quoted reply
template (no "please state your final vote clearly by saying 'I vote for …'").
Vary your phrasing between interventions; avoid stock phrases like "where everyone stands".
Follow the requested action exactly — if it says not to name an option, name none.
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
    """Status-aware closing line. A majority close must acknowledge holdouts —
    a wrap-up that sounds like full agreement is socially dishonest (I17)."""
    final = scenario.option(outcome.final_option).name if outcome.final_option else "no option"
    budget = int(cfg.utterances.word_budgets.closure)
    if outcome.status == "majority":
        holdouts = [
            p.name for p in state.personas
            if state.runtimes[p.id].explicit_vote != outcome.final_option
        ]
        names = ", ".join(holdouts) if holdouts else "some of the group"
        instruction = (
            f"The group did NOT fully agree: {names} did not back {final}. "
            f"Say the group goes ahead with {final} as the majority choice and briefly "
            f"acknowledge that {names} preferred something else. Never word it as if "
            "everyone agreed."
        )
        budget += 8
    elif outcome.status == "successful":
        instruction = f"Everyone visibly agreed on {final}; wrap up warmly in plain words."
    else:
        instruction = (
            "No agreement was reached. Say plainly that the group leaves this undecided "
            "for now, and do not present any option as chosen."
        )
    return f"""You are the neutral moderator closing a casual group decision chat.

Outcome status: {outcome.status}
Final option: {final}
Reason: {outcome.reason}

{instruction}
Write one short closing line under {budget} words.
Sound conversational, like a person wrapping up a chat, not a formal announcement.
Do not add new reasons or facts. No farewell. No speaker prefix."""


def _stance_summary(state: DialogueState, persona: Persona) -> str:
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
        rank = rt.rank(oid)
        if rank == STANCE_NEUTRAL:
            continue
        reason = rt.reason_for(oid) if rank >= STANCE_ACCEPTABLE else rt.reason_against(oid)
        text = f"{aliases.get(oid, oid)}={labels.get(rank, str(rank))}"
        if reason:
            text += f" ({compact_words(reason, 7)})"
        parts.append(text)
    return "; ".join(parts[:5]) or "mostly neutral"


def sim_utterance(
    *,
    persona: Persona,
    state: DialogueState,
    intent: MoveIntent,
    recent_lines: list[str],
    focus_options: list[OptionCard],
    addressee_name: str | None,
    max_words: int,
    min_words: int = 6,
) -> str:
    aliases = short_alias_map(state.scenario.options)
    cards = _option_cards(focus_options or state.scenario.options)
    recent = "\n".join(recent_lines) if recent_lines else "(no recent turns)"
    current = state.runtimes[persona.id].top_option() or persona.preferred_option
    current_name = aliases.get(current, state.scenario.option(current).name if current in state.scenario.option_ids else "undecided")
    stance = _stance_summary(state, persona)
    initial_name = aliases.get(persona.preferred_option, persona.preferred_option)
    required_name = aliases.get(intent.required_vote, intent.required_vote) if intent.required_vote else ""
    old_required = intent.old_preference or current
    old_name = aliases.get(old_required, old_required) if old_required else current_name
    allowed_reason = intent.allowed_reason or "the listed facts and visible support make it workable"
    blocked = ""
    rejected_options = state.runtimes[persona.id].rejected_options()
    if rejected_options:
        oid = sorted(rejected_options)[0]
        reason = state.runtimes[persona.id].reason_against(oid)
        blocked = f"\nHard blocker: they strongly reject {aliases.get(oid, oid)}" + (f" because {reason}" if reason else "") + ". Do not accept or vote for that option."
    target = _target_line(state, intent)
    target_block = f"\nRespond to this recent point: {target}" if target else ""
    address = f"\nAddress {addressee_name} if it sounds natural." if addressee_name else ""
    context = "; ".join(compact_words(item, 14) for item in state.scenario.shared_context) if state.scenario.shared_context else "none"
    params = persona.sim_params
    decision_instruction = ""
    if intent.act == ActType.VOTE:
        # P9: steer vote lines into the parser's own commitment vocabulary — a
        # rotating menu of families not yet used this round, instead of pushing
        # later voters into unparseable "variety". Order the phrase menu by
        # trait fit so vote wording reflects stubbornness/directness/compromise
        # without adding a separate personality subsystem.
        pool = unused_commitment_phrases(intent.avoid_phrases or [], limit=99)
        staying = not intent.option_focus or intent.option_focus[0] == current
        if not staying:
            pool = [f for f in pool if f not in {"I'm still on", "I'll stay with"}]
        preferred = [f for f in _trait_phrase_preferences(persona, staying) if f in pool]
        rest = [f for f in pool if f not in preferred]
        random.shuffle(rest)
        suggestions = (preferred[:2] + rest)[:3]
        menu = ", ".join(f"'… {s} …'" for s in suggestions) if suggestions else "'… gets my vote'"
        if intent.required_vote:
            target_clause = f"commit clearly to {required_name} and no other option"
        else:
            target_clause = "commit clearly to exactly ONE option"
        decision_instruction = (
            f"\nFor this decision turn, {target_clause} using a commitment phrasing "
            f"such as {menu} (fill in the option name yourself). Put the final option immediately next to the commitment phrase, "
            "preferably at the start of the sentence. One short reason may follow the commitment. "
            "No hedging, no 'leaning', no conditions, no question after it."
        )
        if intent.avoid_phrases:
            forbidden = "; ".join(f"'{p}'" for p in intent.avoid_phrases)
            decision_instruction += f"\nEarlier speakers already used these phrasings — do NOT use them: {forbidden}."
        if intent.avoid_reasons:
            used = "; ".join(f"'{r}'" for r in intent.avoid_reasons[:3])
            decision_instruction += f"\nEarlier voters already gave these justifications — give a DIFFERENT reason of your own, in your own words: {used}."
        if intent.allow_vote_change and intent.old_preference and intent.required_vote and intent.old_preference != intent.required_vote:
            decision_instruction += (
                f"\nThis is a genuine compromise switch: earlier pick={old_name}; final vote={required_name}; "
                f"allowed reason={allowed_reason}. Use one natural sentence or two short clauses. "
                "Start with the final vote. You may briefly mention the earlier pick, but avoid a repeated 'I preferred OLD, but ...' template. "
                "Do not add any other factual reason, condition, or pressure language."
            )
        elif intent.required_vote:
            decision_instruction += (
                f"\nThis is a confirmation, not a switch. Vote for {required_name} directly with one short grounded reason. "
                "Do not mention an earlier preference unless it is a different option."
            )
    elif intent.act == ActType.ANSWER:
        decision_instruction = "\nActually answer the question asked. If it asks for information that is not in the option cards or shared context (forecasts, headcounts, outside facts), say plainly that we don't know that here — then give your take. Do not ignore the question."
    elif intent.act == ActType.SOFTEN_TOWARD:
        decision_instruction = "\nThis is not a final vote. Say that another option is becoming more convincing, name what moved you, and also mention what you still give up from your earlier lean."
    elif intent.act == ActType.COMPROMISE:
        decision_instruction = "\nPropose exactly ONE of the existing options as the possible common ground; a visible condition on it is fine. Do not invent blends, split plans, or two-venue combinations."
    elif intent.act == ActType.PROCESS:
        decision_instruction = "\nThis is a procedural group-management move. Keep it concrete, short, and socially natural. Do not cast your own final vote in this line unless explicitly asked."
    elif intent.act == ActType.OPENING:
        decision_instruction = "\nThis is the opening view. Optionally use a tiny chat greeting, then state your current favorite and one grounded reason. Do not make it sound like a final vote."
    continuation_note = ""
    if intent.continuation:
        continuation_note = (
            "\nThis is a quick follow-up to YOUR OWN previous message (you spoke last): one short "
            "add-on thought. Stay on the same point and option as your last message — do not switch "
            "to a different option or open a new issue. Do not repeat or rephrase anything you just "
            "said, do not re-ask the same question, and do not address the same person with the same "
            "request again."
        )
    agenda = ""
    if intent.agenda_key:
        item = next((entry for entry in state.discussion_agenda if entry.key == intent.agenda_key), None)
        if item is not None:
            agenda = f"\nCurrent group agenda item: {item.act.value} about {item.option or 'the decision'} — {item.reason}"
    settled_unknowns = ""
    # An issue earns suppression after its raise->"we don't know" pair played
    # out (mentions >= 2); the first raise is useful and gets answered normally.
    settled_issues = sorted(k for k, v in state.issue_ledger.items() if v.get("mentions", 0) >= 2)
    if settled_issues:
        settled_unknowns = (
            f"\nAlready settled as unknown here (nobody can answer them): {', '.join(settled_issues[:5])}. "
            "Do not ask about or re-raise these; argue from the listed facts instead."
        )
    # Verbosity reaches the prompt only as this numeric word range; the range
    # itself (not a persona parameter) picks the phrasing of the length rule.
    if max_words <= 8 and intent.act != ActType.VOTE:
        # A short-beat draw (P4): one genuine quick reaction, not a compressed argument.
        length_note = (
            f"Reply with ONE quick chat-style reaction of at most {max_words} words — a quick agreement, "
            "brief objection, short answer, or one small condition. Do NOT summarize or argue; one beat, "
            "then stop. A complete short sentence or natural fragment is fine."
        )
    elif max_words >= 20:
        length_note = f"Two short clauses or sentences are okay ({min_words}-{max_words} words)."
    else:
        length_note = f"Aim for {min_words}-{max_words} words, one casual message; a sentence fragment is fine."
    style_notes = ""
    if intent.suppress_tail_question:
        style_notes += "\n- Enough questions are already open; end with a statement, not a question."
    if intent.suppress_name_prefix:
        style_notes += "\n- Recent turns over-used names; do NOT open with another participant's name, just reply."
    if intent.suppress_option_opening:
        style_notes += "\n- Do NOT start with an option name or 'The <option>'; lead with your point, a verb, or a question. You may still name the option mid-sentence — if you mean a different option than the previous message discussed, name it instead of saying 'this one' or 'it'."
    if intent.suppress_i_opening:
        style_notes += "\n- Too many recent messages start with 'I …'; open this one differently — with the topic, the other person's point, an option fact, or a question."
    if intent.suppress_we_opening:
        style_notes += "\n- Too many recent messages start with 'We …'; open this one differently — with the point itself, the option's detail, the other person, or a question."
    if intent.vary_opening:
        style_notes += "\n- Recent turns all opened with the same word; start this one a different way."
    if intent.avoid_pattern in {"concede_but", "worry_but", "tradeoff_but"}:
        style_notes += "\n- Avoid the 'fair point, but…' / 'X is good but I worry…' concession-objection shape used just now; make a different move (a plain claim, a direct question, a concrete comparison, or a firm stance)."

    return f"""Write one natural chat message for {persona.name}.

Topic: {state.scenario.topic}
Context: {context}
Speaker: age={persona.age}; background={compact_words(persona.background, 14)}; goal={compact_words(persona.private_goal, 14)}; initial={initial_name}; current={current_name}; stance={stance}{blocked}
Speech style: {persona.speech_style}. Directness: {_scale_1_5(params.directness)}/5. Stubbornness: {_scale_1_5(params.stubbornness)}/5.
Move: {intent.act.value}. Purpose: {intent.reason}{continuation_note}{agenda}{target_block}{address}{decision_instruction}

Allowed facts:
{cards}

Recent:
{recent}

Rules: one message only, no speaker prefix, no bullets/metadata. {length_note} Match the speech style and age naturally; do not overdo slang or formality. High directness means blunt plain wording, low directness soft tentative wording. Add one new point, answer, concern, or stance shift. Vary the opening; do not start with an option name, "I'm leaning", or "feels". Use only allowed facts; never state a practical detail that isn't listed as if it were fact.{settled_unknowns}{style_notes}"""


def _scale_1_5(value: float) -> int:
    """Map a [0,1] simulator parameter onto the 1-5 scale shown in prompts."""
    return max(1, min(5, round(1 + 4 * float(value))))


# Commitment-form examples keyed by their parsing._PHRASE_FAMILIES label, so a
# family in intent.avoid_phrases can be dropped from the repair menu (I19).
# Every form parses as a direct vote in parsing.py.
_COMMIT_FORM_EXAMPLES = {
    "I'd go with": "I'd go with X",
    "gets my vote": "X gets my vote",
    "my pick is": "my pick is X",
    "works for me": "X works for me",
    "count me in for": "count me in for X",
    "my vote is": "my vote goes to X",
    "I'm going with": "I'm going with X",
    "I'm choosing": "I'm choosing X",
    "I'm sold on": "I'm sold on X",
    "I'll back": "then I'll back X",
    "I can live with": "I can live with X",
    "I'd be happy with": "I'd be happy with X",
}


def _trait_phrase_preferences(persona: Persona, staying: bool) -> list[str]:
    """Commitment phrase families that fit the current stance and traits."""
    p = persona.sim_params
    prefs: list[str] = []
    if staying:
        if p.stubbornness >= 0.60:
            prefs += ["I'm still on", "I'll stay with"]
        if p.directness >= 0.65:
            prefs += ["I vote for", "gets my vote"]
    else:
        if p.stubbornness <= 0.40:
            prefs += ["I can live with", "I'll back"]
        if persona.traits.agreeableness >= 4 or p.directness <= 0.35:
            prefs += ["I'd be happy with"]
        if p.directness >= 0.65:
            prefs += ["I'm going with"]
    return prefs


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
            f" If this is a switch away from {old_name}, you may mention that earlier pick briefly."
            if intent.old_preference and intent.old_preference != intent.required_vote
            else " This is not a switch; do not mention an earlier preference."
        )
        clear_commit = (
            f" The line MUST visibly commit to {required_name} and no other option."
            f"{switch_note} Use only this allowed reason, paraphrased naturally: {allowed_reason}. "
            "Use a clear parser-friendly phrase such as 'I vote for X', 'X gets my vote', "
            "'I'm going with X', or 'I can live with X'. No hedging, no conditions, no question after it."
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
    grounding = ""
    if "UNSUPPORTED_FACT" in issue_codes:
        grounding = " The line invented a fact not in the option cards/context; remove any invented service, fee, policy, location, time, or number and keep only what the cards state (uncertainty like 'we don't know if…' is fine)."
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
            bridge = (
                f" The line switches away from {old_name} with no valid bridge. "
                f"Keep the required commitment to {required_name or 'the target option'}, briefly mention the earlier pick {old_name}, "
                f"and use only this reason: {allowed_reason}. Vary the wording naturally; do not use a fixed template."
            )
        else:
            bridge = (
                f" Commit directly to {required_name or 'the target option'} with one short grounded reason. "
                "Do not mention an earlier preference because this is not a switch."
            )
    return f"""Repair this generated chat line.

Speaker: {persona.name}
Original line: {original_text}
Problems: {', '.join(issue_codes)}
Allowed option facts:
{cards}
Recent chat:
{recent}

Write one natural chat line under {max_words} words. No speaker prefix. Do not invent facts. Avoid generic filler.{clear_commit}{required_focus}{grounding}{malformed}{bridge} Do not append metadata, tags, JSON, or bracketed labels."""


def grounding_check(*, utterance: str, state: DialogueState, focus_options: list[OptionCard]) -> str:
    """Prompt a strict fact-checker: does the line invent facts beyond the board?"""
    cards = _grounding_cards(focus_options or state.scenario.options)
    context = "; ".join(compact_words(item, 14) for item in state.scenario.shared_context) or "none"
    return f"""You are a strict fact-checker for a simulated group decision.
The ONLY facts that exist in this world are in the option cards and shared context.

Option cards:
{cards}
Shared context: {context}

Message to check:
"{utterance}"

A message is UNSUPPORTED if it states a NEW concrete fact that is not in, and
not directly implied by, the cards/context: e.g. an invented service, included or
excluded feature, fee, policy, location, exact time/number, or operational detail.
A message is ALSO unsupported if it attributes a real card fact to the WRONG
option (claiming option X has a feature that only option Y's card lists), if it
misstates what an option is about, or if it compares values of different kinds as
if they measured the same thing (e.g. an object count against a storage size).
Opinions, priorities, trade-off reasoning, questions, and uncertainty are ALWAYS allowed.
Reasoning that follows from a listed attribute is allowed.
Paraphrasing or summarizing a card's wording is allowed for any option, as long as
each fact stays tied to the option whose card lists it. Comparing options through
their listed attributes is allowed and grounded (e.g. "X costs more and takes
longer than Y" when the cards list those costs/durations). Commonsense risk that
follows from an attribute is allowed (an outdoor activity depending on weather; a
long session being tiring), and statements of uncertainty are ALWAYS allowed ("we
don't know the forecast", "it might get canceled").
STRICT RULE for specifics: any number, range, count, fee, schedule, menu item,
feature name, or measurement that does not appear in the cards/context is
UNSUPPORTED — even if the rest of the message expresses uncertainty, and even if
it sounds plausible. The only allowed new numbers are simple arithmetic on listed
numbers (a group total, a difference, a per-person split). If every concrete
claim traces back to the right option's attribute, upside, or concern —
or is such reasoning, arithmetic, or uncertainty — reply false.

    Reply with JSON only: {{"unsupported": true or false, "snippet": "the offending phrase, or empty"}}"""


def _grounding_cards(options: Iterable[OptionCard]) -> str:
    """Compact fact base for the grounding judge.

    The participant prompt needs rich cards for fluent generation, but the judge
    only needs stable facts. Keeping this compact reduces validation-token cost
    without weakening the source-of-truth boundary.
    """
    rows: list[str] = []
    for option in options:
        facts = [f"{k.replace('_', ' ')}={v}" for k, v in option.attrs.items()]
        if option.upside:
            facts.append(f"upside={option.upside}")
        if option.concern:
            facts.append(f"concern={option.concern}")
        rows.append(f"- {option.id}) {option.name}: " + compact_words("; ".join(facts), 46))
    return "\n".join(rows)


def _option_names(state: DialogueState, ids: list[str]) -> str:
    aliases = short_alias_map(state.scenario.options)
    names = [aliases.get(option_id, option_id) for option_id in ids if option_id in state.scenario.option_ids]
    return ", ".join(names)


def _public_state_summary(state: DialogueState) -> str:
    aliases = short_alias_map(state.scenario.options)
    votes = []
    for persona in state.personas:
        vote = state.runtimes[persona.id].explicit_vote
        if vote:
            votes.append(f"{persona.name}->{aliases.get(vote, vote)}")
    untouched = [oid for oid, cov in state.coverage.items() if cov.mentions == 0]
    open_q = [f"{state.name_for(q.target_id)} owes answer" for q in state.open_questions[-2:]]
    parts = []
    parts.append("votes: " + (", ".join(votes) if votes else "none"))
    if untouched:
        parts.append("untouched options: " + ", ".join(untouched))
    if open_q:
        parts.append("open questions: " + ", ".join(open_q))
    return "; ".join(parts)


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
