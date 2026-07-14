"""Prompt construction for setup and action-to-language realization."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from config_loader import cfg
from models import (
    ActionType,
    DialogueState,
    Persona,
    RunOutcome,
    Scenario,
    UserAction,
)


def _schema(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def setup_scenario(topic: str, n: int) -> str:
    labels = [str(value) for value in cfg.scenario.option_labels]
    example = {
        "scenario": {
            "shared_context": ["one brief public fact"],
            "options": [
                {
                    "id": label,
                    "name": f"full option name {label}",
                    "short_name": f"short name {label}",
                    "attrs": {"topic-specific attribute": "public value"},
                    "upside": "brief public advantage",
                    "concern": "brief public drawback",
                }
                for label in labels
            ],
        }
    }
    return f"""Create a fixed public option board for an option-grounded group decision.

Topic: {topic}
Number of deciding participants: {n}
Required option IDs in this exact order: {labels}

Rules:
- Return JSON only.
- Create exactly {len(labels)} distinct, realistic options.
- Choose attributes that are natural for this topic.
- Every objective fact must appear in shared_context or an option card.
- Each option needs {int(cfg.scenario.public_attr_min)}..{int(cfg.scenario.public_attr_max)} concise factual attributes.
- short_name is required, natural, unique, and at most {int(cfg.scenario.short_alias_max_words)} words.
- Keep shared_context to at most {int(cfg.scenario.shared_context_max_items)} short items.
- Do not include an opening question, decision kind, hidden facts, winner, or recommendation.

JSON shape:
{_schema(example)}
"""


def alias_repair(*, topic: str, option_rows: list[dict], invalid: dict[str, str], duplicates: dict[str, str]) -> str:
    labels = [str(row.get("id", "")) for row in option_rows]
    return f"""Repair only the short_name values of these option cards.
Return JSON only as {{"short_names": {{"A": "..."}}}} for every option ID {labels}.
Do not change full names or facts. Each short name must be unique, natural, and at most
{int(cfg.scenario.short_alias_max_words)} words.

Topic: {topic}
Invalid aliases: {_schema(invalid)}
Duplicate aliases: {_schema(duplicates)}
Options: {_schema(option_rows)}
"""


def setup_personas(
    topic: str,
    n: int,
    trait_rows: list[dict],
    required_preferences: dict[str, str],
    options_json: list[dict],
    shared_context: list[str],
    hard_blocker_id: str | None = None,
) -> str:
    examples = []
    for row in trait_rows:
        pid = row["id"]
        examples.append({
            "id": pid,
            "name": row["name"],
            "background": "one short relevant background sentence",
            "private_goal": "one short private decision priority",
            "age": row.get("age", 35),
            "preferred_options": [required_preferences[pid]],
            "option_stances": {
                option["id"]: {
                    "rank": 5 if option["id"] == required_preferences[pid] else 3,
                    "reason_for": "brief grounded reason or empty",
                    "reason_against": "brief grounded concern or empty",
                }
                for option in options_json
            },
        })
    blocker_rule = (
        f"Participant {hard_blocker_id} is the sole hard blocker: exactly one preferred option, "
        "rank every alternative 1, and give a clear grounded reason_against for every alternative."
        if hard_blocker_id else
        "No participant is a hard blocker. Do not rank every alternative 1 for anyone."
    )
    return f"""Create {n} concise participant persona cards for the fixed option board.
Return JSON only. Behavioral traits are already supplied and must not be rewritten.

Topic: {topic}
Shared public context: {_schema(shared_context)}
Public options: {_schema(options_json)}
Fixed participant identities and direct traits: {_schema(trait_rows)}
Required initial primary preferences: {_schema(required_preferences)}

Rules:
- Preserve every id and name exactly.
- Each background and private_goal must be short, plausible, and relevant.
- Private persona content may motivate preferences but must not invent hidden option facts.
- preferred_options must begin with the required option for that participant.
- Rank every option from 1..5: 5 preferred, 4 acceptable, 3 neutral, 2 disliked, 1 hard rejected.
- Normal participants may have uneven reasons; do not force one pro and con for every option.
- Reasons must be grounded in public option attributes or clearly personal priorities.
- {blocker_rule}
- Age may be supplied when absent; do not create implausible age/backstory combinations.
- Do not output speech_style or behavioral traits.

JSON shape:
{_schema({"participants": examples})}
"""


def word_budget(action: ActionType, verbosity: int) -> tuple[int, int]:
    """Soft realization targets; verbosity changes language length only."""
    normal = {
        1: (4, 11),
        2: (7, 16),
        3: (11, 24),
        4: (16, 32),
        5: (22, 44),
    }[int(verbosity)]
    lo, hi = normal
    if action is ActionType.ACKNOWLEDGE:
        return 2, {1: 7, 2: 9, 3: 11, 4: 13, 5: 15}[int(verbosity)]
    if action is ActionType.VOTE:
        return max(3, lo - 4), min(22, max(10, hi - 5))
    if action is ActionType.ANSWER:
        return max(3, lo - 3), max(10, hi - 2)
    if action is ActionType.ASK:
        return max(4, lo - 2), max(11, hi - 3)
    if action is ActionType.COMMENT:
        return max(3, lo - 3), max(10, hi - 4)
    if action is ActionType.OPENING:
        return max(8, min(lo, 18)), min(36, max(18, hi))
    if action in {ActionType.CONCERN, ActionType.COMPARE, ActionType.COMPROMISE}:
        return lo, hi + 4
    return lo, hi


def directness_instruction(level: int) -> str:
    return {
        1: "Use noticeably tentative, qualified wording without becoming vague.",
        2: "Use somewhat softened wording.",
        3: "Use neutral, clear wording.",
        4: "Be explicit and fairly direct.",
        5: "Be very clear and direct, without becoming rude.",
    }[int(level)]


def _option_cards(state: DialogueState, option_ids: Iterable[str]) -> str:
    ids = set(option_ids)
    rows = [option.prompt_card() for option in state.scenario.options if option.id in ids]
    return "\n".join(rows) if rows else "None needed for this action."


def _public_summary(state: DialogueState) -> str:
    """Only public facts useful for wording the already-selected action."""
    names = {persona.id: persona.name for persona in state.personas}
    preferences = [
        f"{names[pid]}→{option_id}"
        for pid, runtime in state.runtimes.items()
        if (option_id := runtime.public_preference)
    ]
    acceptances = [
        f"{names[pid]} accepts {','.join(sorted(runtime.public_acceptances))}"
        for pid, runtime in state.runtimes.items() if runtime.public_acceptances
    ]
    narrowing = ",".join(state.narrowing_options) or "none"
    return "\n".join([
        "Preferences: " + ("; ".join(preferences) if preferences else "none yet"),
        "Acceptances: " + ("; ".join(acceptances) if acceptances else "none"),
        f"Finalists: {narrowing}",
    ])


def _recent_chat(state: DialogueState) -> str:
    limit = int(cfg.conversation.recent_turns_in_prompt)
    turns = state.turns[-limit:]
    return "\n".join(f"{turn.speaker_name}: {turn.text}" for turn in turns) or "No visible turns yet."


def _recent_own_language(state: DialogueState, speaker_id: str) -> str:
    turns = [
        turn.text for turn in state.participant_turns
        if turn.speaker_id == speaker_id
    ][-3:]
    return "\n".join(f"- {text}" for text in turns) or "- none yet"


def _issue_effect_instruction(action: UserAction) -> str:
    if action.issue_effect is None:
        return ""
    return {
        "maintain": "Make it visible that the concern still remains.",
        "partial": "Make it visible that the response helped but did not fully solve the concern.",
        "resolve": "Make it visible that the concern is now sufficiently addressed.",
        "answered": "Answer the exact active question directly before adding any explanation.",
        "continue": "Respond to the active issue without claiming that it is resolved.",
        "open": "State the concrete information need, concern, or comparison being opened.",
    }.get(action.issue_effect.value, "")


def _action_block(action: UserAction) -> str:
    payload = {
        "act": action.act.value,
        "option_focus": list(action.option_focus),
        "reason": action.reason,
    }
    optional = {
        "addressee_id": action.addressee_id,
        "reason_source": asdict(action.reason_source) if action.reason_source else None,
        "optional_personal_context": action.personal_context,
        "issue_id": action.issue_id,
        "issue_effect": action.issue_effect.value if action.issue_effect else None,
        "issue_response_kind": action.issue_response_kind.value if action.issue_response_kind else None,
        "question_intent": action.question_intent.value if action.question_intent else None,
        "question_key": action.question_key,
        "stance_update": asdict(action.stance_update) if action.stance_update else None,
        "vote_option": action.vote_option,
        "stimulus_id": action.stimulus_id,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def realization_prompt(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    *,
    target_question: str | None = None,
) -> str:
    lo, hi = word_budget(action.act, persona.sim_params.verbosity)
    focused = set(action.option_focus)
    if action.reason_source:
        focused.add(action.reason_source.option_id)
    if action.vote_option:
        focused.add(action.vote_option)
    if action.stance_update and action.stance_update.previous_option_id:
        focused.add(action.stance_update.previous_option_id)
    if not focused:
        focused.update(state.narrowing_options or state.scenario.option_ids)
    issue = state.active_issue
    issue_text = "None"
    if issue:
        issue_text = json.dumps({
            "id": issue.id,
            "kind": issue.kind.value,
            "options": list(issue.option_focus),
            "opened_by": state.persona(issue.opened_by).name,
            "addressed_to": state.persona(issue.addressed_to).name if issue.addressed_to else None,
            "summary": issue.summary,
        }, ensure_ascii=False, separators=(",", ":"))
    addressee = state.persona(action.addressee_id).name if action.addressee_id else None
    stimulus_text = state.group_stimulus.prompt_text if (
        state.group_stimulus and action.stimulus_id == state.group_stimulus.id
    ) else "none"
    speaker_lines = [
        f"age={persona.age}",
        f"style={persona.speech_style}",
        f"background={persona.background}",
        f"private goal={persona.private_goal}",
        f"current preference={state.runtimes[persona.id].preferred_option}",
        f"directness={persona.sim_params.directness}/5",
        f"verbosity={persona.sim_params.verbosity}/5",
    ]
    if persona.hard_blocker:
        speaker_lines.append(f"hard blocker; rejection reason={persona.rejection_reason}")
    speaker_card = "\n".join(f"- {line}" for line in speaker_lines)

    if action.act is ActionType.VOTE:
        phase_language_rule = (
            f'Cast one explicit formal vote for exactly Option {action.vote_option}. '
            'Use vote wording and name no other option as a possible vote.'
        )
    else:
        phase_language_rule = (
            'This is not a formal vote. Do not use the words vote, voting, ballot, or my vote. '
            'Express preference with natural wording such as prefer, lean toward, support, favor, or would choose.'
        )
    stance_language_rule = ''
    if action.stance_update:
        if action.stance_update.kind.value == 'switch_preferred':
            if action.act is ActionType.VOTE:
                stance_language_rule = (
                    f' Make the old-to-new change explicit: previously Option '
                    f'{action.stance_update.previous_option_id}, now voting for Option {action.stance_update.option_id}.'
                )
            else:
                stance_language_rule = (
                    f' Make the change to Option {action.stance_update.option_id} explicit with natural movement language '
                    '(for example now prefer, now lean toward, switch to, or move to).'
                )
        elif action.stance_update.kind.value == 'make_acceptable':
            stance_language_rule = (
                f' Make willingness to accept Option {action.stance_update.option_id} explicit '
                '(for example can accept, happy to go with, or would work for me).'
            )
    return f"""Realize one authoritative structured simulator action as natural dialogue.

PRIVATE SPEAKER CARD — only for {persona.name}:
{speaker_card}

AUTHORITATIVE ACTION:
{_action_block(action)}
Resolved addressee name: {addressee or 'group / none'}
Exact target question when answering: {target_question or 'none'}
Structured moderator/group stimulus being answered: {stimulus_text}

RELEVANT PUBLIC OPTION FACTS:
{_option_cards(state, focused)}
Shared public context: {'; '.join(state.scenario.shared_context) or 'none'}

PUBLIC STATE:
{_public_summary(state)}
Active issue: {issue_text}

RECENT VISIBLE CHAT (speaker labels are part of public context):
{_recent_chat(state)}

AVOID REPEATING YOUR OWN WORDING OR POINT:
{_recent_own_language(state, persona.id)}

OUTPUT RULES:
- Output exactly one utterance, with no speaker label, JSON, analysis, quotation marks, or metadata.
- Preserve the action, option focus, addressee, vote, and stance update exactly.
- Do not choose a different act or option.
- Current runtime phase: {state.phase.value}. {phase_language_rule}
- Express the action naturally. Do not name the dialogue act or say phrases such as "I open the discussion", "I acknowledge", or "I compare".
- Mention the action's required option using "Option X" or its public name. For a discussion preference switch, the new option is required and the previous option is optional. For a formal vote switch, name both the previous and new option.{stance_language_rule}
- Treat reason_source as exact provenance. Paraphrase it, but do not derive a new price/time/distance comparison from it.
- When mentioning facts about multiple options, bind every fact directly to its option name. Avoid ambiguous pronouns such as "its" or "that option" after naming several options.
- Use only public option facts. Do not invent numbers, prices, times, distances, capacities, facilities, specifications, or unsupported claims such as cheaper, longer, faster, earlier, or later.
- Do not invent new personal facts beyond the private card and optional personal-context seed.
- Personal background and private goal are optional. Use them only when they explain this action, and do not repeat a biographical detail already stated in recent chat.
- Use age only for a subtle, age-consistent lexical register. Avoid stereotypes, caricature, slang overload, and catchphrases. The explicit speech style is more important. Age and style must not change the action, stance, frequency, or target length.
- {directness_instruction(persona.sim_params.directness)} Directness changes wording only, never the selected action or stance.
- {_issue_effect_instruction(action)}
- Aim for roughly {lo}–{hi} words as a soft target; do not fill the range when a short acknowledgment, answer, acceptance, or vote is sufficient. Never clip a complete sentence.
- Conversationally narrow actions may be one concise sentence. Do not force every turn to restate the full option name, private goal, position, and reason.
- Do not repeatedly begin with the same preference formula or repeat the speaker's name as an introduction.
"""


def repair_prompt(
    state: DialogueState,
    persona: Persona,
    action: UserAction,
    rejected_text: str,
    errors: list[str],
    *,
    target_question: str | None = None,
) -> str:
    focused = set(action.option_focus)
    if action.vote_option:
        focused.add(action.vote_option)
    if action.reason_source:
        focused.add(action.reason_source.option_id)
    if action.stance_update and action.stance_update.previous_option_id:
        focused.add(action.stance_update.previous_option_id)
    if action.act is ActionType.VOTE:
        phase_rule = f'Use an explicit formal vote for exactly Option {action.vote_option}.'
    else:
        phase_rule = 'Do not use vote, voting, ballot, or my vote; this action occurs outside formal voting.'
    stance_rule = ''
    if action.stance_update:
        if action.stance_update.kind.value == 'switch_preferred':
            stance_rule = (
                f' Explicitly communicate the change to Option {action.stance_update.option_id}; '
                + (
                    f'name the previous Option {action.stance_update.previous_option_id} as well.'
                    if action.act is ActionType.VOTE else
                    'the previous option need not be repeated outside formal voting.'
                )
            )
        elif action.stance_update.kind.value == 'make_acceptable':
            stance_rule = f' Explicitly state willingness to accept or go with Option {action.stance_update.option_id}.'
    lo, hi = word_budget(action.act, persona.sim_params.verbosity)
    return f"""Rewrite one rejected utterance. Preserve the structured action exactly.

Speaker: {persona.name}
Structured action: {_action_block(action)}
Exact target question: {target_question or 'none'}
Rejected text: {rejected_text}
Hard failures to fix: {_schema(errors)}
Relevant public facts (including previous and new options for a switch):
{_option_cards(state, focused or state.scenario.option_ids)}

Return exactly one corrected utterance with no speaker label or metadata. {phase_rule}{stance_rule}
{_issue_effect_instruction(action)} Aim for roughly {lo}–{hi} words as a soft target, but keep a narrow action concise.
Express the action naturally without naming the dialogue act. Do not add facts, change the option,
change the vote, change the addressee, or change the stance update. When several options are named,
attach each public fact directly to its option name and avoid ambiguous pronouns for option facts.
Do not reuse the rejected sentence structure merely by replacing a few words.
"""


def moderator_opening(scenario: Scenario) -> str:
    context = " ".join(scenario.shared_context)
    board = " ".join(option.public_line() for option in scenario.options)
    return f"Today we're deciding: {scenario.topic}. {context} Options: {board}"


def moderator_stall_prompt() -> str:
    return "We seem to be pausing. Is there another relevant reason, concern, or question before we narrow this down?"


def moderator_coverage_prompt(scenario: Scenario, option_id: str) -> str:
    option = scenario.option(option_id)
    return f"We have not really considered {option.short_name or option.name}. Is there a reason to keep it or rule it out?"


def moderator_narrowing(scenario: Scenario, options: tuple[str, ...]) -> str:
    names = [scenario.option(option_id).short_name or scenario.option(option_id).name for option_id in options]
    if len(names) == 1:
        return f"The discussion currently points most strongly to {names[0]}. Please raise any final concern before the vote."
    return f"The current leading options are {names[0]} and {names[1]}. Please focus the final discussion on that comparison."


def moderator_vote_request(*, revote: bool = False) -> str:
    return "Please cast one final vote for exactly one option." if not revote else "No option reached a majority. After this final narrowing, please vote once more for exactly one option."


def moderator_closure(outcome: RunOutcome, scenario: Scenario) -> str:
    if outcome.final_option:
        name = scenario.option(outcome.final_option).short_name or scenario.option(outcome.final_option).name
        return f"The result is {outcome.status}: {name}."
    return "No option reached a majority after the re-vote, so the result is unresolved."
