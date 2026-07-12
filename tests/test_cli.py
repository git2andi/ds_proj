"""CLI entry-point behavior (todo_validation item 2).

Precedence: an explicit CLI topic, topic file, or piped topic always requests
automatic scenario generation for that topic — even when environment.mode is
manual. Only with no explicit topic does manual mode run the configured
environment (and auto mode prompt interactively). participants.mode stays
independent of the scenario source.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import tests  # noqa: F401  # puts src/ and the project root on sys.path

import main as main_module
from builders import SetupBuilder


class FakeCfg:
    """Minimal cfg stand-in for main(): environment mode + no limits section."""

    def __init__(self, env_mode: str = "auto") -> None:
        self._env = {"mode": env_mode}

    def get(self, key, default=None):
        if key == "environment":
            return self._env
        return default

    def __contains__(self, key) -> bool:
        return False


class FakeRunResult:
    def __init__(self) -> None:
        class _Outcome:
            status = "successful"
            final_option = "A"

        self.outcome = _Outcome()
        self.token_summary = {
            "setup_tokens_in": 0, "setup_tokens_out": 0,
            "dialogue_tokens_in": 0, "dialogue_tokens_out": 0,
            "validator_tokens_in": 0, "validator_tokens_out": 0,
            "total_tokens_in": 0, "total_tokens_out": 0,
        }
        self.log_paths = {"dir": "fake"}


class FakeRunner:
    """Records the (topic, force_auto_scenario) pairs main() dispatched."""

    calls: list[tuple[str, bool]] = []

    def __init__(self, topic: str, *, force_auto_scenario: bool = False) -> None:
        FakeRunner.calls.append((topic, force_auto_scenario))

    def run(self) -> FakeRunResult:
        return FakeRunResult()


def run_main(argv: list[str], *, env_mode: str = "auto", stdin_text: str | None = None,
             prompted: str | None = None) -> tuple[int, list[tuple[str, bool]]]:
    """Drive main.main() with fakes; returns (exit_code, dispatched calls)."""
    FakeRunner.calls = []
    fake_stdin = io.StringIO(stdin_text or "")
    fake_stdin.isatty = lambda: stdin_text is None  # type: ignore[method-assign]
    patches = [
        mock.patch.object(main_module, "cfg", FakeCfg(env_mode)),
        mock.patch.object(main_module, "DialogueRunner", FakeRunner),
        mock.patch.object(main_module.sys, "argv", ["main.py", *argv]),
        mock.patch.object(main_module.sys, "stdin", fake_stdin),
    ]
    if prompted is not None:
        patches.append(mock.patch("builtins.input", return_value=prompted))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        code = main_module.main()
    return code, list(FakeRunner.calls)


class TopicsFromArgsTests(unittest.TestCase):
    def test_topic_file_skips_blanks_and_comments(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.txt"
            path.write_text("# comment\n\nTopic one\n  Topic two  \n", encoding="utf-8")
            topics = main_module._explicit_topics(["main.py", str(path)])
        self.assertEqual(topics, ["Topic one", "Topic two"])

    def test_explicit_topic_joins_argv_words(self):
        topics = main_module._explicit_topics(["main.py", "Book", "a", "flight"])
        self.assertEqual(topics, ["Book a flight"])


class CliPrecedenceTests(unittest.TestCase):
    def test_explicit_topic_runs_in_auto_mode(self):
        code, calls = run_main(["Book a flight from Miami to Stockholm"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Book a flight from Miami to Stockholm", True)])

    def test_no_args_auto_mode_prompts_interactively(self):
        code, calls = run_main([], prompted="Prompted topic")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Prompted topic", False)])

    def test_piped_topics_run_in_auto_mode(self):
        code, calls = run_main([], stdin_text="Topic one\nTopic two\n")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Topic one", True), ("Topic two", True)])

    def test_topic_file_runs_every_topic_in_order(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.txt"
            path.write_text("First topic\n# skip\nSecond topic\n", encoding="utf-8")
            code, calls = run_main([str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("First topic", True), ("Second topic", True)])

    def test_no_topic_manual_mode_runs_configured_environment(self):
        code, calls = run_main([], env_mode="manual")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("", False)])

    def test_explicit_topic_beats_manual_environment(self):
        # Item 2: explicit command input is never silently discarded.
        code, calls = run_main(["Book a flight from Miami to Stockholm"], env_mode="manual")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Book a flight from Miami to Stockholm", True)])

    def test_piped_topic_beats_manual_environment(self):
        code, calls = run_main([], env_mode="manual", stdin_text="Book a flight\n")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("Book a flight", True)])


class RunnerEnvironmentOverrideTests(unittest.TestCase):
    def test_forced_auto_setup_ignores_manual_environment_keeps_profiles(self):
        """participants.mode stays independent: manual profiles combine with a
        CLI-forced automatic scenario; the manual environment is bypassed."""
        manual_env = {"topic": "Configured manual topic", "options": [], "shared_context": []}
        profiles = [{"name": "Kira", "description": "", "private_goal": "",
                     "preferred_option": None, "age": None, "speech_style": "",
                     "rejection": None, "rejection_reason": "", "traits": {}, "parameters": {}}]
        with mock.patch("builders.manual_environment", return_value=manual_env), \
                mock.patch("builders.manual_participant_profiles", return_value=profiles), \
                mock.patch("builders.get_llm_client", return_value=object()):
            forced = SetupBuilder("CLI topic", force_auto_scenario=True)
            configured = SetupBuilder("CLI topic")
        self.assertIsNone(forced._manual_env)
        self.assertEqual(forced.topic, "CLI topic")
        self.assertEqual(forced._profiles, profiles)
        self.assertEqual(configured._manual_env, manual_env)
        self.assertEqual(configured.topic, "Configured manual topic")


if __name__ == "__main__":
    unittest.main()
