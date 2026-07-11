"""Tests for the deterministic thread engine (TODO 6): issue keys, creation
identity, lifecycle transitions, aging/reactivation, and primary selection."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  # puts src/ on sys.path before src imports

from models import BlockingStrength, ThreadStatus, ThreadType
from controller.threads import (
    age_threads,
    find_thread,
    hot_blocking_thread_against,
    mark_response,
    normalize_issue_key,
    normalize_pair,
    open_thread,
    reactivate_thread,
    resolve_thread,
    select_primary_thread,
)

from tests.fixtures import append_turn, make_scenario, make_state


def _names(state) -> list[str]:
    return [p.name for p in state.personas]


class IssueKeyTests(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario()
        self.names = ["Mira", "Jonas", "Lea"]

    def test_attribute_key_has_priority(self):
        key = normalize_issue_key(
            "The cost worries me a bit here.", self.scenario, self.names, focus_options=["A"]
        )
        self.assertEqual(key, "cost")

    def test_deterministic_category_fallback(self):
        key = normalize_issue_key(
            "Can we even still get a reservation for Saturday?", self.scenario, self.names
        )
        self.assertEqual(key, "availability")

    def test_content_signature_is_deterministic_and_name_free(self):
        text = "Jonas, the Museum lighting felt gloomy and cramped last visit."
        key1 = normalize_issue_key(text, self.scenario, self.names)
        key2 = normalize_issue_key(text, self.scenario, self.names)
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith("sig:"))
        self.assertNotIn("jonas", key1)
        self.assertNotIn("museum", key1)

    def test_pair_normalization_is_deterministic(self):
        self.assertEqual(normalize_pair(["C", "A", "C"]), ["A", "C"])
        self.assertEqual(normalize_pair(["A", "C"]), normalize_pair(["C", "A"]))

    def test_card_concern_paraphrases_share_one_key(self):
        # Closeout 4: equivalent paraphrases of a card-listed concern converge
        # on one stable key built from the card's tokens, not per-wording sigs.
        key1 = normalize_issue_key(
            "It might be a bit too quiet at the Museum for a whole day.",
            self.scenario, self.names, focus_options=["A"],
        )
        key2 = normalize_issue_key(
            "Won't the Museum end up feeling really quiet, though?",
            self.scenario, self.names, focus_options=["A"],
        )
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith("concern:"))

    def test_different_issues_on_same_option_stay_separate(self):
        cost = normalize_issue_key(
            "The Museum cost seems high for what it is.", self.scenario, self.names, focus_options=["A"]
        )
        quiet = normalize_issue_key(
            "Won't the Museum end up feeling really quiet, though?", self.scenario, self.names, focus_options=["A"]
        )
        self.assertEqual(cost, "cost")
        self.assertNotEqual(cost, quiet)

    def test_card_upside_relevance_gets_upside_key(self):
        key = normalize_issue_key(
            "Is the Museum actually easy to adjust if plans change?",
            self.scenario, self.names, focus_options=["A"],
        )
        self.assertTrue(key.startswith("upside:"))

    def test_other_cards_concern_does_not_capture_focused_issue(self):
        # B's concern mentions "tired"; with focus on A the key must not come
        # from another option's card.
        key = normalize_issue_key(
            "Is the Museum actually easy to adjust if someone gets tired early?",
            self.scenario, self.names, focus_options=["A"],
        )
        self.assertTrue(key.startswith("upside:"))


class ThreadCreationTests(unittest.TestCase):
    def test_same_issue_on_two_options_creates_two_threads_with_shared_key(self):
        state = make_state()
        t1 = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        t2 = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["B"], issue_key="cost",
            started_by="p1", source_turn_index=2,
        )
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertNotEqual(t1.thread_id, t2.thread_id)
        self.assertEqual(t1.issue_key, t2.issue_key)

    def test_same_identity_touches_instead_of_duplicating(self):
        state = make_state()
        t1 = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        t2 = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p2", source_turn_index=3,
        )
        self.assertIs(t1, t2)
        self.assertEqual(len(state.threads), 1)
        self.assertIn("p2", t1.participants_involved)
        self.assertEqual(t1.last_touched_turn, 3)

    def test_repeated_resolved_issue_is_suppressed(self):
        state = make_state()
        t1 = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        resolve_thread(state, t1, reason="accepted tradeoff")
        suppressed = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p2", source_turn_index=4,
        )
        self.assertIsNone(suppressed)
        reopened = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p2", source_turn_index=5, reopen_resolved=True,
        )
        self.assertIs(reopened, t1)
        self.assertEqual(t1.status, ThreadStatus.HOT)
        self.assertIsNone(t1.resolution_reason)


class LifecycleTests(unittest.TestCase):
    def _spend_participant_turns(self, state, count: int) -> None:
        speakers = [p.id for p in state.personas]
        for i in range(count):
            append_turn(state, speakers[i % len(speakers)], "Just adding a small thought here.")

    def test_hot_to_cooling_on_response(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1, required_respondent="p2", question_scope="direct",
        )
        mark_response(state, thread, responder_id="p2", turn_index=2)
        self.assertEqual(thread.status, ThreadStatus.COOLING)

    def test_cooling_question_resolves_after_quiet_window(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.QUESTION, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, required_respondent="p2", question_scope="direct",
        )
        mark_response(state, thread, responder_id="p2", turn_index=0)
        self._spend_participant_turns(state, 2)  # cooling_turns = 2
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.RESOLVED)

    def test_unanswered_thread_goes_stale_after_timeout(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0,
        )
        self._spend_participant_turns(state, 3)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.HOT)  # below stale_after_turns = 4
        self._spend_participant_turns(state, 1)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.STALE)

    def test_hard_blocker_keeps_priority_longer(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.BLOCKER, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, blocking_strength=BlockingStrength.HARD,
        )
        self._spend_participant_turns(state, 4)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.HOT)  # blocker timeout = 6
        self._spend_participant_turns(state, 2)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.STALE)

    def test_stale_thread_reactivates_on_visible_reopen(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0,
        )
        self._spend_participant_turns(state, 4)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.STALE)
        again = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p3", source_turn_index=state.turn_index,
        )
        self.assertIs(again, thread)
        self.assertEqual(thread.status, ThreadStatus.HOT)

    def test_staleness_never_touches_stance_state(self):
        state = make_state()
        state.runtimes["p1"].mark_rejected("A", reason_against="hard constraint")
        thread = open_thread(
            state, thread_type=ThreadType.BLOCKER, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, blocking_strength=BlockingStrength.HARD,
        )
        self._spend_participant_turns(state, 6)
        age_threads(state)
        self.assertEqual(thread.status, ThreadStatus.STALE)
        self.assertIn("A", state.runtimes["p1"].rejected_options())


class PrimarySelectionTests(unittest.TestCase):
    def _open(self, state, thread_type, focus, key, *, turn, strength=BlockingStrength.NONE,
              scope=None, respondent=None):
        return open_thread(
            state, thread_type=thread_type, focus_options=focus, issue_key=key,
            started_by="p1", source_turn_index=turn, blocking_strength=strength,
            question_scope=scope, required_respondent=respondent,
        )

    def test_priority_order(self):
        state = make_state()
        comparison = self._open(state, ThreadType.COMPARISON, ["A", "B"], "cost", turn=1)
        concern_other = self._open(state, ThreadType.CONCERN, ["C"], "risk", turn=2)
        concern_candidate = self._open(state, ThreadType.CONCERN, ["A"], "timing", turn=3)
        hard_blocker = self._open(
            state, ThreadType.BLOCKER, ["A"], "cost", turn=4, strength=BlockingStrength.HARD
        )
        group_q = self._open(state, ThreadType.QUESTION, ["B"], "cost", turn=5, scope="group", respondent="p2")
        direct_q = self._open(
            state, ThreadType.QUESTION, ["A"], "capacity", turn=6, scope="direct", respondent="p3"
        )

        pick = lambda: select_primary_thread(state, candidate_options=["A"])
        self.assertIs(pick(), direct_q)
        resolve_thread(state, direct_q, reason="answered")
        self.assertIs(pick(), group_q)
        resolve_thread(state, group_q, reason="answered")
        self.assertIs(pick(), hard_blocker)
        resolve_thread(state, hard_blocker, reason="mitigated")
        self.assertIs(pick(), concern_candidate)
        resolve_thread(state, concern_candidate, reason="accepted")
        # Remaining hot: the comparison includes candidate A (relevant) and the
        # off-candidate concern does not, so relevance breaks the tie.
        self.assertIs(pick(), comparison)
        resolve_thread(state, comparison, reason="pair settled")
        self.assertIs(pick(), concern_other)

    def test_tie_break_prefers_candidate_relevance_then_recency(self):
        state = make_state()
        off_candidate = self._open(state, ThreadType.CONCERN, ["C"], "risk", turn=9)
        on_candidate = self._open(state, ThreadType.CONCERN, ["B"], "timing", turn=2)
        self.assertIs(
            select_primary_thread(state, candidate_options=["B"]),
            on_candidate,
        )
        # Without candidate relevance, recency decides.
        self.assertIs(select_primary_thread(state, candidate_options=[]), off_candidate)

    def test_cooling_thread_only_selected_when_caller_allows(self):
        state = make_state()
        thread = self._open(state, ThreadType.CONCERN, ["A"], "cost", turn=1)
        mark_response(state, thread, responder_id="p2", turn_index=2)
        self.assertIsNone(select_primary_thread(state, candidate_options=["A"]))
        self.assertIs(
            select_primary_thread(state, candidate_options=["A"], include_cooling=True),
            thread,
        )

    def test_no_random_choice_needed(self):
        state = make_state()
        a = self._open(state, ThreadType.CONCERN, ["A"], "cost", turn=3)
        b = self._open(state, ThreadType.CONCERN, ["B"], "cost", turn=3)
        picks = {select_primary_thread(state, candidate_options=[]).thread_id for _ in range(20)}
        self.assertEqual(len(picks), 1)  # earliest creation order breaks the tie

    def test_hot_blocking_gate_helper(self):
        state = make_state()
        self.assertIsNone(hot_blocking_thread_against(state, ["A"]))
        blocker = self._open(
            state, ThreadType.BLOCKER, ["A"], "cost", turn=1, strength=BlockingStrength.HARD
        )
        self.assertIs(hot_blocking_thread_against(state, ["A"]), blocker)
        self.assertIsNone(hot_blocking_thread_against(state, ["B"]))


class ContributionCapTests(unittest.TestCase):
    """Thread caps count accepted contributions, not unique participants (cleanup 4)."""

    def test_counter_counts_accepted_turns_not_unique_speakers(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        self.assertEqual(thread.contribution_count, 1)
        mark_response(state, thread, responder_id="p2", turn_index=2)
        # The same participant restating on a later turn still counts a turn.
        from controller.threads import touch_thread
        touch_thread(state, thread, turn_index=3, participant_id="p1")
        self.assertEqual(thread.contribution_count, 3)
        self.assertEqual(len(thread.participants_involved), 2)

    def test_same_turn_touched_twice_counts_once(self):
        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        from controller.threads import touch_thread
        touch_thread(state, thread, turn_index=1, participant_id="p1")  # same accepted turn
        self.assertEqual(thread.contribution_count, 1)

    def test_hard_cap_stops_a_thread_from_driving_turns(self):
        from config_loader import cfg

        state = make_state()
        thread = open_thread(
            state, thread_type=ThreadType.CONCERN, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=1,
        )
        self.assertIs(select_primary_thread(state, candidate_options=["A"]), thread)
        thread.contribution_count = int(cfg.threads.max_thread_turns_hard)
        self.assertIsNone(select_primary_thread(state, candidate_options=["A"]))
        mark_response(state, thread, responder_id="p2", turn_index=2)
        self.assertIsNone(
            select_primary_thread(state, candidate_options=["A"], include_cooling=True)
        )

    def test_soft_cap_blocks_optional_cooling_continuation(self):
        import random

        from config_loader import cfg
        from tests.stubs import make_runner

        state = make_state()
        runner = make_runner(state)
        thread = open_thread(
            state, thread_type=ThreadType.BLOCKER, focus_options=["A"], issue_key="cost",
            started_by="p1", source_turn_index=0, blocking_strength=BlockingStrength.HARD,
        )
        mark_response(state, thread, responder_id="p2", turn_index=0)
        thread.contribution_count = int(cfg.threads.max_thread_turns_soft)
        for seed in range(30):
            random.seed(seed)
            self.assertIsNone(runner._maybe_cooling_continuation(state, thread))


if __name__ == "__main__":
    unittest.main()
