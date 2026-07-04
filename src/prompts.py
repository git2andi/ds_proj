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
from models import DialogueState, MoveIntent, OptionCard, Persona, RunOutcome, Scenario
from utils import compact_words


# Fallback opening question for manual environments that do not specify one.
DEFAULT_OPENING_QUESTION = "What matters most to each of us in this decision?"


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
- If shared_context states a numeric limit (budget cap, max distance, max duration), every option must satisfy it — never create an option that violates a stated cap.
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
    # Manual participant profiles may fix background/private_goal; tell the LLM
    # to keep them verbatim so its generated fields stay consistent with them.
    fixed_field_rule = (
        "\n- If a trait row already contains background or private_goal, copy that text exactly "
        "and keep the other fields consistent with it."
        if any(row.get("background") or row.get("private_goal") for row in trait_rows)
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
        decision_instruction = "\nFor this decision turn, commit clearly to exactly one option in your own words (like 'I'd go with X', 'X gets my vote', 'my pick is X', or 'X works for me'). Use a different commitment phrasing than the previous voters in the chat. No hedging, no 'leaning', no conditions, no question after it."
        if intent.avoid_phrases:
            forbidden = "; ".join(f"'{p}'" for p in intent.avoid_phrases)
            decision_instruction += f"\nEarlier speakers already used these phrasings — do NOT use them: {forbidden}."
        if intent.avoid_reasons:
            used = "; ".join(f"'{r}'" for r in intent.avoid_reasons[:3])
            decision_instruction += f"\nEarlier voters already gave these justifications — give a DIFFERENT reason of your own, in your own words: {used}."
        if intent.allow_vote_change:
            decision_instruction += (
                f"\nIf you move to a different option than {current_name}, you MUST bridge the switch: "
                f"briefly name that you preferred {current_name} (or concede it, e.g. 'I still like…, but…') "
                "and give one honest reason you can move now. A short concessive clause is fine here; a "
                "fresh condition or question is not."
            )
    elif intent.act.value == "reject":
        decision_instruction = "\nFor this decision turn, visibly reject the blocked option and name the acceptable alternative if there is one."
    elif intent.act.value == "answer":
        decision_instruction = "\nActually answer the question asked. If it asks for information that is not in the option cards or shared context (forecasts, headcounts, outside facts), say plainly that we don't know that here — then give your take. Do not ignore the question."
    elif intent.act.value == "soften":
        decision_instruction = "\nThis is not a final vote. Say that another option is becoming more convincing, name what moved you, and also mention what you still give up from your earlier lean."
    elif intent.act.value in {"call_vote", "summarize_split", "probe_holdout", "suggest_narrowing"}:
        decision_instruction = "\nThis is a procedural group-management move. Keep it concrete, short, and socially natural. Do not cast your own final vote in this line unless explicitly asked."
    continuation_note = ""
    if intent.continuation:
        continuation_note = (
            "\nThis is a quick follow-up to YOUR OWN previous message (you spoke last): one short "
            "add-on thought. Do not repeat or rephrase anything you just said, do not re-ask the same "
            "question, and do not address the same person with the same request again."
        )
    agenda = ""
    if intent.agenda_index is not None and 0 <= intent.agenda_index < len(persona.agenda):
        item = persona.agenda[intent.agenda_index]
        agenda = f"\nPending simulator agenda item: {item.act.value} about {item.option or 'the decision'} — {item.reason}"
    if params.verbosity <= 0.4:
        length_note = f"Keep it very short ({min_words}-{max_words} words), blunt and to the point; a sentence fragment is fine."
    elif params.verbosity >= 0.7:
        length_note = f"Two short clauses or sentences are okay ({min_words}-{max_words} words)."
    else:
        length_note = f"Aim for {min_words}-{max_words} words, one casual sentence."
    if params.directness >= 0.65:
        tone_note = " Say it plainly, almost no hedging."
    elif params.directness <= 0.35:
        tone_note = " Keep the wording soft and tentative."
    else:
        tone_note = ""
    style_notes = ""
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

    return f"""Write {persona.name}'s next message in a natural group decision chat.

Topic: {state.scenario.topic}
Shared context: {context}
Options: {', '.join(f'{o.id}={aliases[o.id]}' for o in state.scenario.options)}

Speaker:
- background: {persona.background}
- private goal: {persona.private_goal}
- voice: {voice}
- initial preference: {initial_name}; current internal lean: {current_name}{blocked}

Move to render: {intent.act.value}
Purpose: {intent.reason}{continuation_note}{agenda}{target_block}{address}{decision_instruction}

Use only these option facts:
{cards}

Recent chat:
{recent}

Style:
- One message only; no name prefix, quotes, bullets, metadata, or bracketed labels.
- {length_note}{tone_note}
- Follow the voice exactly — sentence shape, bluntness, and energy should make the speaker recognizable without the name. Contractions and casual interjections fit.
- Vary sentence shape and opening; do not open with an option name, "I'm leaning", or "feels". Names, "you", "we", or no option name are all fine.
- Add one new point, concern, answer, or stance shift; don't repeat reasons from the recent chat. Never invent facts beyond the option facts above. If something isn't in the cards or context (parking, weather, crowds, policies, extra services), don't state it as fact — treat it as unknown or ask.{style_notes}"""


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
    "I'm sold on": "I'm sold on X",
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
    focus = [state.scenario.option(o) for o in intent.option_focus if o in state.scenario.option_ids]
    cards = _option_cards(focus or state.scenario.options)
    recent = "\n".join(recent_lines[-3:]) if recent_lines else "(no recent turns)"
    clear_commit = ""
    if intent.act.value in {"vote", "accept"}:
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
    grounding = ""
    if "UNSUPPORTED_FACT" in issue_codes:
        grounding = " The line invented a fact not in the option cards/context; remove any invented service, fee, policy, location, time, or number and keep only what the cards state (uncertainty like 'we don't know if…' is fine)."
    bridge = ""
    if "UNBRIDGED_SWITCH" in issue_codes:
        aliases = short_alias_map(state.scenario.options)
        current = state.runtimes[persona.id].current_preference or persona.preferred_option
        old_name = aliases.get(current, current)
        bridge = (
            f" The line switches away from {old_name} (your earlier pick) with no explanation. "
            f"Keep the new commitment, but bridge it: name that you preferred {old_name} or concede it "
            "(e.g. 'I still like…, but…'), and give one honest reason you can move now."
        )
    return f"""Repair this generated chat line.

Speaker: {persona.name}
Original line: {original_text}
Problems: {', '.join(issue_codes)}
Allowed option facts:
{cards}
Recent chat:
{recent}

Write one natural chat line under {max_words} words. No speaker prefix. Do not invent facts. Avoid generic filler.{clear_commit}{required_focus}{grounding}{bridge} Do not append metadata, tags, JSON, or bracketed labels."""


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
don't know the forecast", "it might get canceled"). If every concrete claim traces
back to the right option's attribute, upside, tradeoff, or concern — or is such
reasoning or uncertainty — reply false.

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
        if option.tradeoff:
            facts.append(f"tradeoff={option.tradeoff}")
        if option.concern:
            facts.append(f"concern={option.concern}")
        if option.best_for:
            facts.append(f"best_for={option.best_for}")
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


def _voice_guidance(persona: Persona) -> str:
    """Contrastive, concrete register instructions. Abstract adjectives get
    flattened by the model into one polite voice; micro-examples do not."""
    p = persona.sim_params
    parts: list[str] = []
    if p.directness >= 0.75:
        parts.append("blunt: plain declaratives, no softeners (e.g. 'Too pricey. Not worth it.')")
    elif p.directness >= 0.60:
        parts.append("direct, concrete, low hedging")
    elif p.directness <= 0.35:
        parts.append("tentative: hedges like 'maybe' or 'I guess', suggests rather than insists")
    if p.stubbornness >= 0.80:
        parts.append("digs in: dismisses alternatives curtly, keeps returning to their own priority, concedes nothing without a strong reason")
    elif p.stubbornness >= 0.60:
        parts.append("pushes their concern instead of agreeing too quickly")
    elif p.compromise_threshold <= 0.35:
        parts.append("actively looks for workable compromise")
    if p.engagement >= 0.70:
        parts.append("energetic, jumps in with opinions; the odd exclamation fits")
    elif p.engagement <= 0.35:
        parts.append("dry and minimal; only speaks when it adds something")
    if p.verbosity <= 0.35:
        parts.append("clipped: fragments over full sentences (e.g. 'Games. Cheap, fun, done.')")
    elif p.verbosity >= 0.70:
        parts.append("flowing: happily adds a second thought or a small aside")
    if p.responsiveness >= 0.70:
        parts.append("reacts to the previous speaker by name when natural")
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
