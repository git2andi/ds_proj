"""
tools/analyze_logs.py
---------------------
Offline analysis of dialogue logs produced by DialogueLogger.

Usage:
  python tools/analyze_logs.py                    # analyse all logs in logs/
  python tools/analyze_logs.py logs/20260521_*    # glob specific files
  python tools/analyze_logs.py --json             # output as JSON instead of text

Reads per-dialogue:
  {id}_summary.json  — aggregate stats from DialogueLogger.flush()
  {id}.jsonl         — per-turn trace records (TurnTrace, Stage 1)
  {id}_state.jsonl   — per-turn TurnRecord (Stage 5 structured state)
  {id}.csv           — structured row-per-turn data

Computed metrics:
  force_close_rate              fraction of dialogues that ended via force-close or best_available_decision
  natural_consensus_rate        fraction that ended with natural consensus (outcome=success)
  confirmation_rollback_rate    fraction that had at least one confirmation rejection
  hard_blocker_dialogue_rate    fraction with persistent opposition across 3+ consecutive turns
  avg_participation_gini        mean Gini coefficient across dialogues (0=equal, 1=monopoly)
  question_answer_rate          fraction of question turns followed by a non-question turn
  repeat_opener_rate            fraction of participant turns that reuse the same opening word
  avg_tokens_in / avg_tokens_out  mean prompt / completion tokens per dialogue
  avg_llm_calls_per_dialogue    mean LLM calls (sum of tokens_in > 0 turns)
  consensus_tier_distribution   how often each tier (soft/regex/reduced_opposition/llm) fired
  phase_turn_distribution       mean turns spent in each phase
  vote_flip_distribution        mean vote flips per speaker
  avg_total_turns               mean total participant turns per dialogue
  state_recovery_agreement_rate (Stage 5) fraction of votes where new StanceTable agrees with
                                old regex-based vote extraction
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_summaries(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*_summary.json"))


def _find_jsonl(log_dir: Path, dialogue_id: str) -> Path:
    return log_dir / f"{dialogue_id}.jsonl"


def _find_state_jsonl(log_dir: Path, dialogue_id: str) -> Path:
    """Stage 5 structured state JSONL ({id}_state.jsonl)."""
    return log_dir / f"{dialogue_id}_state.jsonl"


def _find_csv(log_dir: Path, dialogue_id: str) -> Path:
    return log_dir / f"{dialogue_id}.csv"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Per-dialogue metrics
# ---------------------------------------------------------------------------

def _question_answer_rate(traces: list[dict]) -> float:
    """Fraction of participant question turns immediately followed by a non-question turn."""
    participant_traces = [t for t in traces if not t.get("is_moderator", False)]
    if len(participant_traces) < 2:
        return 1.0
    answered = 0
    questions = 0
    for i, t in enumerate(participant_traces[:-1]):
        if t.get("is_question", False):
            questions += 1
            next_t = participant_traces[i + 1]
            if not next_t.get("is_question", False):
                answered += 1
    return answered / questions if questions > 0 else 1.0


def _repeat_opener_rate(traces: list[dict]) -> float:
    """
    Fraction of participant turns whose opening word was already used as an
    opener in an earlier participant turn in this dialogue.
    """
    participant_traces = [t for t in traces if not t.get("is_moderator", False)]
    if len(participant_traces) < 2:
        return 0.0
    seen_openers: set[str] = set()
    repeat_count = 0
    for t in participant_traces:
        text = t.get("text", "").strip()
        if not text:
            continue
        opener = text.split()[0].rstrip(",.!?").lower()
        if opener in seen_openers:
            repeat_count += 1
        seen_openers.add(opener)
    return repeat_count / len(participant_traces) if participant_traces else 0.0


def _hard_blocker_rate_from_traces(traces: list[dict]) -> bool:
    """
    Heuristic: True if any participant speaker had 3+ consecutive turns in
    the same phase expressing opposition (dialogue_act_estimated=ASSERT_OPPOSITION
    or REJECT_WITH_REASON) without any CONCEDE or CONFIRM in between.
    Approximates "persistent opposition" without requiring private beliefs.
    """
    participant_traces = [t for t in traces if not t.get("is_moderator", False)]
    # Build per-speaker run-length of opposition acts
    speaker_opp_run: dict[str, int] = {}
    for t in participant_traces:
        speaker = t.get("speaker", "")
        act = t.get("dialogue_act_estimated", "")
        if act in ("ASSERT_OPPOSITION", "REJECT_WITH_REASON"):
            speaker_opp_run[speaker] = speaker_opp_run.get(speaker, 0) + 1
            if speaker_opp_run[speaker] >= 3:
                return True
        elif act in ("CONCEDE", "CONFIRM", "COMMIT_VOTE"):
            speaker_opp_run[speaker] = 0
    return False


def _llm_calls_from_traces(traces: list[dict]) -> int:
    """Count turns with tokens_in > 0 as LLM call proxy."""
    return sum(1 for t in traces if t.get("tokens_in", 0) > 0)


def _consensus_tiers_from_traces(traces: list[dict]) -> list[str]:
    """Return list of non-'none' consensus tier hits from the trace."""
    return [
        t["consensus_tier_used"]
        for t in traces
        if t.get("consensus_tier_used") and t["consensus_tier_used"] != "none"
    ]


def _phase_turn_counts(csv_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in csv_rows:
        if row.get("is_moderator", "").lower() in ("false", "0", ""):
            phase = row.get("phase", "unknown")
            counts[phase] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Stage 5 — state recovery agreement rate
# ---------------------------------------------------------------------------

def _old_votes_from_csv(csv_rows: list[dict]) -> dict[str, str]:
    """
    Extract the final (most recent) regex-based vote per participant from the CSV.
    Replicates the logic of utils.extract_preference_vote over the text column.
    """
    import re

    def _extract(msg: str) -> str | None:
        text = msg.strip().lower()
        fp_patterns = [
            r"\bi\s+prefer\s+option\s+([a-d])\b",
            r"\bi(?:'m|\s+am)\s+(?:going\s+with|for)\s+option\s+([a-d])\b",
            r"\bi(?:'d|\s+would)\s+(?:go\s+with|choose|prefer)\s+option\s+([a-d])\b",
            r"\bi\s+(?:choose|pick|want|vote\s+for)\s+option\s+([a-d])\b",
            r"\bmy\s+(?:choice|pick|preference)\s+is\s+(?:option\s+)?([a-d])\b",
        ]
        for pat in fp_patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).upper()
        return None

    votes: dict[str, str] = {}
    # CSV rows arrive oldest-first; we want latest-last, so iterate normally
    for row in csv_rows:
        if row.get("is_moderator", "").lower() in ("true", "1"):
            continue
        speaker = row.get("speaker", "").strip()
        text = row.get("text", "")
        vote = _extract(text)
        if vote:
            votes[speaker] = vote   # overwrite with newer vote each time
    return votes


def _new_votes_from_state_jsonl(state_jsonl: Path) -> dict[str, str]:
    """
    Extract final committed vote per participant from Stage 5 _state.jsonl.
    Uses TurnRecord entries where dialogue_act == 'COMMIT_VOTE' and
    stance_updates contain a support stance with confidence >= 1.0.
    """
    if not state_jsonl.exists():
        return {}

    votes: dict[str, str] = {}
    with open(state_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("dialogue_act") != "COMMIT_VOTE":
                continue
            speaker = record.get("speaker", "")
            for su in record.get("stance_updates", []):
                if su.get("stance") == "support" and su.get("confidence", 0) >= 1.0:
                    votes[speaker] = su["option"]
                    break
    return votes


def _state_recovery_agreement(old_votes: dict[str, str], new_votes: dict[str, str]) -> float:
    """Fraction of old-path votes where new-path agrees. N/A (1.0) if no old votes."""
    if not old_votes:
        return 1.0
    agreements = sum(
        1 for speaker, opt in old_votes.items()
        if new_votes.get(speaker) == opt
    )
    return round(agreements / len(old_votes), 3)


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def analyse(log_dir: Path) -> dict[str, Any]:
    summaries = _find_summaries(log_dir)
    if not summaries:
        return {"error": f"No *_summary.json files found in {log_dir}"}

    n = len(summaries)
    force_closed = 0
    natural_consensus = 0
    confirmation_rollbacks = 0
    hard_blocker_count = 0
    gini_values: list[float] = []
    tokens_in_values: list[int] = []
    tokens_out_values: list[int] = []
    total_turns: list[int] = []
    question_answer_rates: list[float] = []
    repeat_opener_rates: list[float] = []
    llm_calls_list: list[int] = []
    consensus_tier_hits: Counter = Counter()
    phase_turns_acc: dict[str, list[int]] = defaultdict(list)
    vote_flip_totals: list[float] = []
    state_recovery_rates: list[float] = []

    for summary_path in summaries:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        dialogue_id = summary.get("dialogue_id", summary_path.stem.replace("_summary", ""))
        outcome = summary.get("outcome", "")

        if outcome in ("force_close", "best_available_decision"):
            force_closed += 1
        if outcome == "success":
            natural_consensus += 1

        if summary.get("force_closed_after_confirmation_failure", False) or \
           summary.get("reopened_after_confirmation", False) or \
           summary.get("confirmation_rejection_count", 0) > 0:
            confirmation_rollbacks += 1

        gini = summary.get("participation_gini")
        if gini is not None:
            gini_values.append(float(gini))

        ti = summary.get("total_in", 0)
        to = summary.get("total_out", 0)
        tokens_in_values.append(int(ti))
        tokens_out_values.append(int(to))

        speaker_counts = summary.get("speaker_turn_counts", {})
        if speaker_counts:
            total_turns.append(sum(speaker_counts.values()))

        # Vote flips per speaker (average across speakers)
        flips = summary.get("vote_flips_per_speaker", {})
        if flips:
            vote_flip_totals.append(sum(flips.values()) / len(flips))

        # Per-turn data from Stage 1 JSONL trace
        jsonl_path = _find_jsonl(log_dir, dialogue_id)
        traces = _load_jsonl(jsonl_path)
        if traces:
            question_answer_rates.append(_question_answer_rate(traces))
            repeat_opener_rates.append(_repeat_opener_rate(traces))
            llm_calls_list.append(_llm_calls_from_traces(traces))
            if _hard_blocker_rate_from_traces(traces):
                hard_blocker_count += 1
            for tier in _consensus_tiers_from_traces(traces):
                consensus_tier_hits[tier] += 1

        csv_path = _find_csv(log_dir, dialogue_id)
        csv_rows = _load_csv(csv_path)
        if csv_rows:
            for phase, count in _phase_turn_counts(csv_rows).items():
                phase_turns_acc[phase].append(count)

        # Stage 5: state recovery agreement rate
        state_jsonl_path = _find_state_jsonl(log_dir, dialogue_id)
        if state_jsonl_path.exists():
            old_votes = _old_votes_from_csv(csv_rows)
            new_votes = _new_votes_from_state_jsonl(state_jsonl_path)
            rate = _state_recovery_agreement(old_votes, new_votes)
            state_recovery_rates.append(rate)

    def _mean(lst: list) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    phase_turn_means = {
        phase: _mean(counts) for phase, counts in sorted(phase_turns_acc.items())
    }

    result: dict[str, Any] = {
        "dialogues_analysed": n,
        "natural_consensus_rate": round(natural_consensus / n, 3),
        "force_close_rate": round(force_closed / n, 3),
        "confirmation_rollback_rate": round(confirmation_rollbacks / n, 3),
        "hard_blocker_dialogue_rate": round(hard_blocker_count / n, 3),
        "avg_participation_gini": _mean(gini_values),
        "question_answer_rate": _mean(question_answer_rates),
        "repeat_opener_rate": _mean(repeat_opener_rates),
        "avg_tokens_in": _mean(tokens_in_values),
        "avg_tokens_out": _mean(tokens_out_values),
        "avg_llm_calls_per_dialogue": _mean(llm_calls_list),
        "avg_total_participant_turns": _mean(total_turns),
        "avg_vote_flips_per_speaker": _mean(vote_flip_totals),
        "consensus_tier_distribution": dict(consensus_tier_hits),
        "avg_phase_turn_counts": phase_turn_means,
    }
    if state_recovery_rates:
        result["state_recovery_agreement_rate"] = _mean(state_recovery_rates)
        result["state_recovery_dialogues"] = len(state_recovery_rates)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(data: dict[str, Any]) -> None:
    if "error" in data:
        print(f"Error: {data['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  Dialogue Log Analysis  ({data['dialogues_analysed']} dialogue(s))")
    print(f"{'='*60}")
    print(f"  Natural consensus rate        : {data.get('natural_consensus_rate', 0):.1%}")
    print(f"  Force-close rate              : {data['force_close_rate']:.1%}")
    print(f"  Confirmation rollback rate    : {data['confirmation_rollback_rate']:.1%}")
    print(f"  Hard-blocker dialogue rate    : {data.get('hard_blocker_dialogue_rate', 0):.1%}  (heuristic)")
    print(f"  Avg participation Gini        : {data['avg_participation_gini']:.3f}  (0=equal, 1=monopoly)")
    print(f"  Question→answer rate          : {data['question_answer_rate']:.1%}")
    print(f"  Repeat opener rate            : {data.get('repeat_opener_rate', 0):.1%}")
    print(f"  Avg tokens in / out           : {data['avg_tokens_in']:.0f} / {data['avg_tokens_out']:.0f}")
    print(f"  Avg LLM calls per dialogue    : {data.get('avg_llm_calls_per_dialogue', 0):.1f}")
    print(f"  Avg total participant turns   : {data['avg_total_participant_turns']:.1f}")
    print(f"  Avg vote flips per speaker    : {data.get('avg_vote_flips_per_speaker', 0):.2f}")

    if "state_recovery_agreement_rate" in data:
        print(
            f"  State recovery agreement rate : "
            f"{data['state_recovery_agreement_rate']:.1%}  "
            f"({data['state_recovery_dialogues']} dialogues with _state.jsonl)"
        )

    print(f"\n  Consensus tier distribution:")
    tiers = data.get("consensus_tier_distribution", {})
    if tiers:
        total_hits = sum(tiers.values())
        for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
            print(f"    {tier:<25} {count:>4}  ({count/total_hits:.0%})")
    else:
        print("    (no consensus tier data)")

    print(f"\n  Avg turns per phase:")
    for phase, mean in data.get("avg_phase_turn_counts", {}).items():
        print(f"    {phase:<20} {mean:.1f}")
    print()


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]

    log_dir = Path(args[0]) if args else Path(__file__).parent.parent / "logs"

    data = analyse(log_dir)

    if as_json:
        print(json.dumps(data, indent=2))
    else:
        _print_report(data)


if __name__ == "__main__":
    main()
