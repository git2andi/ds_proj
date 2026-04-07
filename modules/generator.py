import os
import datetime
from typing import Any

from dotenv import load_dotenv

from modules.llm_client import get_llm_client

load_dotenv()


class MultiUserSimulator:
    EXCLUDED_SPEAKERS = {"Moderator", "Travel Agent"}

    FOCUS_KEYWORDS = {
        "saving money": [
            "budget", "cheap", "cost", "price", "expensive",
            "save", "deal", "affordable", "lowest price"
        ],
        "safety first": [
            "safe", "safety", "risk", "reliable", "trustworthy"
        ],
        "quickest route": [
            "fast", "quick", "time", "duration", "soon", "direct", "convenient"
        ],
        "comfort": [
            "comfort", "seat", "legroom", "relaxed",
            "smooth", "comfortable", "cozy", "atmosphere"
        ],
        "fewest layovers": [
            "layover", "direct", "stop", "connection", "nonstop"
        ],
    }

    PARTICIPATION_BONUS = {
        "active": 0.25,
        "reactive": 0.15,
        "reserved": -0.05,
        "opinionated": 0.20,
    }

    def __init__(self, persona, setting, persona_manager=None):
        self.persona = persona
        self.name = persona["name"]
        self.behavior = persona["behavior"]
        self.setting = setting

        self.focus = persona.get("focus", "")
        self.participation = persona.get("participation", "reactive")
        self.agreeableness = persona.get("agreeableness", 0.5)
        self.flexibility = persona.get("flexibility", 0.5)
        self.initiative = persona.get("initiative", 0.5)

        self.style = persona.get("style", "friendly")
        self.length = persona.get("length", "medium")
        self.speech_mode = persona.get("speech_mode", "natural")

        self.persona_manager = persona_manager

        project_root = os.path.dirname(os.path.dirname(__file__))
        self.log_file = os.path.join(project_root, "token_log.txt")

        self.llm = get_llm_client()

        if not self.persona.get("goal"):
            self.goal = self._generate_initial_goal()
            self.persona["goal"] = self.goal
            if self.persona_manager:
                self.persona_manager.save_persona(self.persona)
        else:
            self.goal = self.persona["goal"]

    def _extract_total_tokens(self, usage: Any) -> int:
        if usage is None:
            return 0

        provider = self.llm.provider
        if provider == "gemini":
            return int(getattr(usage, "total_token_count", 0) or 0)
        if provider == "groq":
            return int(getattr(usage, "total_tokens", 0) or 0)

        return 0

    def _update_log(self, usage: Any) -> None:
        today = str(datetime.date.today())
        total_tokens = self._extract_total_tokens(usage)

        lines = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if line.startswith(today):
                try:
                    parts = line.split("|")
                    old_tokens = int(parts[1].split(":")[1].strip())
                    old_calls = int(parts[2].split(":")[1].strip())
                    lines[i] = (
                        f"{today} | Tokens: {old_tokens + total_tokens} "
                        f"| Calls: {old_calls + 1}\n"
                    )
                    found = True
                except (IndexError, ValueError):
                    pass
                break

        if not found:
            lines.append(f"{today} | Tokens: {total_tokens} | Calls: 1\n")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _call_llm(self, prompt: str) -> str:
        text, usage = self.llm.generate(prompt)

        if usage is not None:
            self._update_log(usage)

        return text.strip()

    def _generate_initial_goal(self):
        prompt = (
            f"Persona: {self.behavior}\n"
            f"Setting: {self.setting}\n"
            "Generate exactly one short internal goal sentence for this participant. "
            "Do not include dialogue, quotes, bullet points, markdown, or explanation."
        )

        try:
            raw = self._call_llm(prompt)
            return raw if raw else f"Promote interests related to {self.focus}."
        except Exception as e:
            print(f"!! Goal Gen Error for {self.name}: {e}")
            return f"Promote interests related to {self.focus}."

    def _extract_recent_speakers(self, history, max_count=3):
        recent_speakers = []

        for line in reversed(history):
            if ":" not in line:
                continue

            speaker = line.split(":", 1)[0].strip()
            if speaker not in self.EXCLUDED_SPEAKERS:
                recent_speakers.append(speaker)

            if len(recent_speakers) >= max_count:
                break

        return recent_speakers

    def _extract_recent_participant_points(self, history, max_count=3):
        points = []

        for line in reversed(history):
            if ":" not in line:
                continue

            speaker, msg = line.split(":", 1)
            speaker = speaker.strip()
            msg = msg.strip()

            if speaker in self.EXCLUDED_SPEAKERS or speaker == self.name:
                continue

            points.append((speaker, msg))

            if len(points) >= max_count:
                break

        points.reverse()
        return points

    def _format_recent_points(self, history, max_count=3):
        points = self._extract_recent_participant_points(history, max_count=max_count)

        if not points:
            return "None."

        return "\n".join(f"- {speaker}: {msg}" for speaker, msg in points)

    def _is_discussion_stalling(self, history):
        if len(history) < 4:
            return False

        recent_texts = []
        for line in reversed(history):
            if ":" not in line:
                continue

            speaker, msg = line.split(":", 1)
            if speaker.strip() not in self.EXCLUDED_SPEAKERS:
                recent_texts.append(msg.strip().lower())

            if len(recent_texts) >= 3:
                break

        if len(recent_texts) < 3:
            return False

        keyword_sets = []
        for txt in recent_texts:
            tokens = {
                word.strip(".,!?*'\"()")
                for word in txt.split()
                if len(word.strip(".,!?*'\"()")) > 3
            }
            keyword_sets.append(tokens)

        overlap_1 = len(keyword_sets[0].intersection(keyword_sets[1]))
        overlap_2 = len(keyword_sets[1].intersection(keyword_sets[2]))

        return overlap_1 >= 1 and overlap_2 >= 1

    def should_speak(self, history, last_speaker=None):
        if not history:
            return 0.0

        last_msg = history[-1].lower()
        score = 0.0

        for kw in self.FOCUS_KEYWORDS.get(self.focus, []):
            if kw in last_msg:
                score += 0.35
                break

        if "?" in history[-1]:
            score += 0.20

        score += self.PARTICIPATION_BONUS.get(self.participation, 0.0)
        score += 0.20 * self.initiative

        own_name_prefix = f"{self.name}:"
        has_spoken = any(line.startswith(own_name_prefix) for line in history)
        if not has_spoken:
            score += 0.30

        recent_speakers = self._extract_recent_speakers(history, max_count=3)
        if len(recent_speakers) >= 1 and recent_speakers[0] == self.name:
            score -= 0.45
        elif len(recent_speakers) >= 2 and recent_speakers[1] == self.name:
            score -= 0.20

        recent_count = sum(1 for speaker in recent_speakers if speaker == self.name)
        score -= 0.10 * recent_count

        if self._is_discussion_stalling(history):
            score += 0.12 * self.flexibility
            score += 0.12 * self.agreeableness
            if not has_spoken:
                score += 0.08

        return max(0.0, min(score, 1.0))

    def generate_turn(self, history):
        chat_str = "\n".join(history[-10:])

        prompt = f"""
Identity: {self.name}
Persona: {self.behavior}
Secret Goal: {self.goal}

Recent History:
{chat_str}

Instructions:
- Reply with only the response text.
- Stay consistent with your persona and internal goal.
- React to the current discussion, not just your own preference.
- Move the group toward a practical decision.
- Do not invent exact facts that are not in the discussion history.
- Refer to the listed options if they exist.
- If options are listed, choose from those options only.
- Prefer naming one preferred option directly.
- Optionally name one backup option.
- Do not ask the moderator for additional facts or new searches.
- Do not propose checking new options outside the listed set.
- Do not repeat your previous point unless necessary.
- Keep the reply concise.
- Avoid always starting with "I think", "I agree", or "I'm".
- Short natural replies are allowed, but not always rely on them.

Interaction rules:
- Sometimes directly respond to one participant's point by name.
- If another participant raised a valid concern, acknowledge it briefly.
- You may disagree, but do it naturally and briefly.
- Keep your own goal as the main priority, but you can compromise if progress is possible.
- If someone else's concern is relevant, mention it before giving your choice.

Style constraints:
- Your style is {self.style}.
- Your preferred response length is {self.length}.
- Your speech mode is {self.speech_mode}.
- Sound like a real participant in a group chat, not like an essay.
- Fragments are allowed.
- Informal wording is allowed when it fits the persona.
- Only use abbreviations if they fit the persona naturally.
"""
        try:
            raw = self._call_llm(prompt)
            return raw if raw else "[SILENCE]"
        except Exception as e:
            print(f"!! Turn Gen Error for {self.name}: {e}")
            return "[SILENCE]"