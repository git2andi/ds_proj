"""
reasoning.py
------------
Consensus engine + evaluation-only signals + scoped fact-check.

ConsensusEngine        -- uses public StanceTable; no private beliefs.
fisher_ratios()        -- Fisher (1970) favor/disfavor/ambiguous/conditional.
                          EVALUATION-ONLY (kept in .eval.json; not used for control).
deliberation_metrics() -- Stage 5 + deliberative-quality framework:
                          justification rate, reciprocity rate, reflexivity rate,
                          and the gating signals used by orchestrator._update_phase.
fact_check()           -- Stage 4: SCOPED to claims-about-options. World knowledge
                          (Toulmin warrants from persona experience) is permitted;
                          numbers asserted close to an option mention are flagged.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, TYPE_CHECKING

from config_loader import cfg

if TYPE_CHECKING:
    from state import StanceTable, StructuredState


# =============================================================================
# ConsensusEngine
# =============================================================================

ConsensusState = str   # none | candidate_emerging | majority_candidate |
                       # conditional_consensus | full_consensus | blocked | failed


class ConsensusEngine:
    """Derive consensus state and best-available decision from public StanceTable."""

    def compute_state(
        self, structured: "StructuredState", participant_names: list[str]
    ) -> ConsensusState:
        if not participant_names:
            return "none"

        stance  = structured.stance_table
        options = list(structured.options.keys())
        n       = len(participant_names)

        candidate = structured.candidate_option
        if candidate and stance.unresolved_blockers(candidate):
            return "blocked"

        best = _leading_option(stance, options)
        if best is None:
            return "none"

        supporters: set[str]  = set()
        conditional: set[str] = set()
        ambiguous: set[str]   = set()
        opposing: set[str]    = set()

        for name in participant_names:
            lbl = stance.current_stance_label(name, best)
            if lbl == "support":
                supporters.add(name)
            elif lbl == "conditional_support":
                conditional.add(name)
            elif lbl == "ambiguous":
                ambiguous.add(name)
            elif lbl in ("oppose", "blocker"):
                opposing.add(name)

        if len(supporters) == n:
            return "full_consensus"
        if len(supporters) + len(conditional) == n:
            return "conditional_consensus"
        if not opposing and len(supporters) + len(conditional) + len(ambiguous) == n:
            return "majority_candidate"
        if supporters or conditional:
            return "candidate_emerging"
        return "none"

    def leading_candidate(
        self, structured: "StructuredState", participant_names: list[str]
    ) -> Optional[str]:
        del participant_names  # not needed; leading is purely stance-driven
        return _leading_option(structured.stance_table, list(structured.options.keys()))

    def best_available_decision(
        self,
        structured: "StructuredState",
        participant_names: list[str],
        explicit_votes: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """Pick the best option for force-close.

        Priority order:
          1. Explicit vote plurality (each vote counts as a strong +2 signal).
          2. Stance-table weights as tiebreaker.
        """
        weights = cfg.consensus.stance_weights
        stance  = structured.stance_table
        options = list(structured.options.keys())
        if not options:
            return None

        votes = explicit_votes or {}

        def score(opt: str) -> float:
            vote_bonus = sum(2.0 for v in votes.values() if v == opt)
            stance_score = sum(
                weights.get(stance.current_stance_label(name, opt), 0.0)
                for name in participant_names
            )
            return vote_bonus + stance_score

        return max(options, key=score)


def _leading_option(stance: "StanceTable", options: list[str]) -> Optional[str]:
    weights = cfg.consensus.leading_weights
    best_opt: Optional[str] = None
    best_score: float = -1.0
    for opt in options:
        score = sum(
            weights.get(su.stance, 0.0)
            for (_, o), su in stance.current_items()
            if o == opt
        )
        if score > best_score:
            best_score = score
            best_opt = opt
    return best_opt if best_score > 0.0 else None


# =============================================================================
# Fisher (1970) -- evaluation-only ratios
# =============================================================================

def fisher_ratios(structured: "StructuredState", window: Optional[int] = None) -> dict[str, float]:
    w = window if window is not None else cfg.fisher.window_size
    recent = structured.stance_table.history()[-w:]
    if not recent:
        return {"favor": 0.0, "disfavor": 0.0, "ambiguous": 0.0, "conditional": 0.0}
    counts: Counter = Counter(su.stance for su in recent)
    total = len(recent)
    return {
        "favor":       round(counts.get("support", 0) / total, 3),
        "disfavor":    round((counts.get("oppose", 0) + counts.get("blocker", 0)) / total, 3),
        "ambiguous":   round((counts.get("ambiguous", 0) + counts.get("neutral", 0)) / total, 3),
        "conditional": round(counts.get("conditional_support", 0) / total, 3),
    }


# =============================================================================
# Stage 5 -- deliberation metrics + gating signals
# =============================================================================

def deliberation_metrics(structured: "StructuredState",
                         participant_names: list[str]) -> dict[str, float]:
    """Deliberative-quality signals + the gating booleans the orchestrator reads.

    Quality metrics (logged; useful for evaluation):
      - justification_rate  : share of stance turns that carry a reason marker.
                              Stand-in for Toulmin warrants visible in transcript.
      - reciprocity_rate    : share of participant turns addressing or answering
                              another sim (Ouchi/Tsuboi). Proxy for "talking to,
                              not past, each other".
      - reflexivity_rate    : share of turns showing update -- concession or
                              vote flip.

    Gating signals (consumed by orchestrator phase transitions):
      - positions_with_reason : sims that have stated a position WITH a reason.
      - answered_challenges   : count of challenges that received an answer.
      - exhaustion            : repetition pressure exceeds the configured low
                                bar -- live arguments are exhausted.
    """
    from state import DialogueAct, _REASON_MARKERS as reason_re  # local to avoid cycles
    sim_turns = [t for t in structured.turns if not t.is_moderator]
    total = max(1, len(sim_turns))
    stance_acts = {
        DialogueAct.COMMIT_VOTE, DialogueAct.ASSERT_SUPPORT,
        DialogueAct.ASSERT_OPPOSITION, DialogueAct.CHALLENGE,
        DialogueAct.CONDITIONAL_ACCEPT, DialogueAct.REJECT_WITH_REASON,
    }
    update_acts = {DialogueAct.CONCEDE, DialogueAct.CONDITIONAL_ACCEPT}

    justified = sum(
        1 for t in sim_turns
        if t.dialogue_act in stance_acts and reason_re.search(t.text)
    )
    reciprocal = sum(
        1 for t in sim_turns
        if t.addressees or t.answers_question_id is not None or t.answered_challenge_id is not None
    )
    reflexive = sum(1 for t in sim_turns if t.dialogue_act in update_acts)

    positions_with_reason = sum(
        1 for name in participant_names
        if structured.participants.get(name) and
           structured.participants[name].position_with_reason_stated
    )
    answered = sum(
        1 for c in structured.discourse.challenges if c.answered_turn_id is not None
    )

    return {
        "justification_rate":  round(justified / total, 3),
        "reciprocity_rate":    round(reciprocal / total, 3),
        "reflexivity_rate":    round(reflexive / total, 3),
        "positions_with_reason": positions_with_reason,
        "answered_challenges":   answered,
    }


# =============================================================================
# Grounding -- Stage 4: scoped to claims-about-options only
# =============================================================================

# Numbers that are always option attributes (currency, percentages) are flagged
# even outside the proximity window. Plain integers are world-knowledge unless
# they sit within `option_proximity_chars` of an option mention.
_CURRENCY_OR_PCT = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?%")
_PLAIN_NUMBER    = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def fact_check(turn_text: str, option_texts: list[str], topic: str) -> list[str]:
    """Return phrases that look like INVENTED OPTION ATTRIBUTES.

    Stage 4 scope:
      - currency / percentages -> always flagged if not in source.
      - bare numbers           -> flagged only when they sit within the
                                  configured character window of an option
                                  letter or alias mention (treated as an
                                  attribute of that option).
      - quoted strings         -> flagged if absent from source (an invented
                                  named feature).
      - parenthesised digit-asides -> flagged (typical fabrication pattern).
    World knowledge ("CRISPR has been around since 2012") is now permitted.
    """
    source = " ".join([topic.lower(), *[o.lower() for o in option_texts]])
    suspicious: list[str] = []

    # Currency / percentages -- always option-attribute-like.
    for m in _CURRENCY_OR_PCT.finditer(turn_text):
        token = m.group(0)
        if token.lower() not in source:
            suspicious.append(token)

    # Bare numbers -- only if proximate to an option mention.
    proximity = cfg.grounding.option_proximity_chars
    option_spans = _option_attribute_spans(turn_text, option_texts, proximity)
    for m in _PLAIN_NUMBER.finditer(turn_text):
        token = m.group(0)
        if token.lower() in source:
            continue
        if _CURRENCY_OR_PCT.match(token):
            continue  # already handled above
        if any(s <= m.start() <= e for s, e in option_spans):
            suspicious.append(token)

    # Quoted strings (likely invented named features).
    for m in re.finditer(r'"([^"]{3,80})"', turn_text):
        phrase = m.group(1).strip()
        if phrase.lower() not in source:
            suspicious.append(f'"{phrase}"')

    # Parenthesised digit-asides -- the classic fabrication shape.
    for m in re.finditer(r"\(([^)]{4,60})\)", turn_text):
        phrase = m.group(1).strip()
        if re.search(r"\d", phrase) and phrase.lower() not in source:
            suspicious.append(f"({phrase})")

    return suspicious


def _option_attribute_spans(turn_text: str, option_texts: list[str],
                            proximity: int) -> list[tuple[int, int]]:
    """Build (start, end) windows around every option mention. A bare number
    inside any of these windows is treated as an asserted option attribute."""
    from utils import OptionResolver  # local: utils only depends on cfg

    resolver = OptionResolver(option_texts)
    raw_spans = resolver.option_mention_spans(turn_text)
    if not raw_spans:
        return []

    spans = [
        (max(0, s - proximity), min(len(turn_text), e + proximity))
        for s, e in raw_spans
    ]
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged


def repair_directive() -> str:
    return (
        "IMPORTANT: You may bring your own knowledge and experience to bear, but "
        "do not invent specific attributes of the OPTIONS (no fake prices, fake "
        "named features, or fabricated numbers attached to an option)."
    )


# =============================================================================
# Phase 6 -- post-dialogue evaluation summary
# =============================================================================

def evaluation_summary(
    outcome: str,
    final_option: Optional[str],
    speaker_targets: dict[str, int],
    turn_records: list[dict],
) -> dict:
    """Automatic quality metrics written to the .eval.json file.

    Args:
        outcome: "success" | "force_close" | "failed"
        final_option: the winning option letter, or None
        speaker_targets: {speaker: target_word_count} from persona.response_length
        turn_records: buffered turn dicts from DialogueLogger

    Returns a dict suitable for meta["evaluation"].
    """
    import re as _re

    valid_options = {"A", "B", "C", "D"}
    outcome_valid = (
        outcome in ("success", "force_close") and (final_option or "") in valid_options
    ) or outcome == "failed"

    participant_records = [r for r in turn_records if not r.get("is_moderator", False)]
    total_turns = max(1, len(participant_records))

    # Per-speaker word counts from actual text
    word_counts: dict[str, list[int]] = {}
    for r in participant_records:
        speaker = r["speaker"]
        text = r.get("text", "")
        wc = len(_re.sub(r"\s+", " ", text).split()) if text.strip() else 0
        word_counts.setdefault(speaker, []).append(wc)

    avg_words = {s: round(sum(wcs) / len(wcs), 1) for s, wcs in word_counts.items() if wcs}

    # Length adherence: 1 - |avg - target| / target, floored at 0
    adherence: dict[str, float] = {}
    for s, avg in avg_words.items():
        target = speaker_targets.get(s, 24)
        adherence[s] = round(max(0.0, 1.0 - abs(avg - target) / max(1, target)), 2)

    # Repetition rate: share of turns flagged SELF_REPETITION by verifier
    rep_counts: dict[str, int] = {}
    rep_totals: dict[str, int] = {}
    for r in participant_records:
        speaker = r["speaker"]
        rep_totals[speaker] = rep_totals.get(speaker, 0) + 1
        if "SELF_REPETITION" in r.get("verification_issues", []):
            rep_counts[speaker] = rep_counts.get(speaker, 0) + 1
    rep_rate = {
        s: round(rep_counts.get(s, 0) / max(1, rep_totals.get(s, 1)), 3)
        for s in rep_totals
    }

    # Overall repair rate
    repairs = sum(1 for r in participant_records if r.get("repair_attempted", False))
    repair_rate = round(repairs / total_turns, 3)

    # Participation ratio (normalised)
    turn_counts = {s: len(wcs) for s, wcs in word_counts.items()}
    total_tc = sum(turn_counts.values()) or 1
    participation_ratio = {s: round(v / total_tc, 3) for s, v in turn_counts.items()}

    return {
        "outcome_valid": outcome_valid,
        "avg_words_per_turn": avg_words,
        "target_words_per_turn": {s: speaker_targets.get(s, 24) for s in avg_words},
        "length_adherence_per_speaker": adherence,
        "repetition_rate_per_speaker": rep_rate,
        "repair_rate": repair_rate,
        "participation_ratio": participation_ratio,
    }
