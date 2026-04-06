import os
import datetime
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()


class MultiUserSimulator:
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

        self.persona_manager = persona_manager

        project_root = os.path.dirname(os.path.dirname(__file__))
        self.log_file = os.path.join(project_root, "token_log.txt")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment.")

        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-lite"

        if not self.persona.get("goal"):
            self.goal = self._generate_initial_goal()
            self.persona["goal"] = self.goal
            if self.persona_manager:
                self.persona_manager.save_persona(self.persona)
        else:
            self.goal = self.persona["goal"]

    def _update_log(self, usage):
        today = str(datetime.date.today())
        total_tokens = getattr(usage, "total_token_count", 0)

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
                    lines[i] = f"{today} | Tokens: {old_tokens + total_tokens} | Calls: {old_calls + 1}\n"
                    found = True
                except (IndexError, ValueError):
                    pass
                break

        if not found:
            lines.append(f"{today} | Tokens: {total_tokens} | Calls: 1\n")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _generate_initial_goal(self):
        prompt = (
            f"Persona: {self.behavior}\n"
            f"Setting: {self.setting}\n"
            "Generate exactly one short internal goal sentence for this participant. "
            "Do not include dialogue, quotes, bullet points, markdown, or explanation."
        )
        try:
            time.sleep(12)
            res = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )

            if hasattr(res, "usage_metadata"):
                self._update_log(res.usage_metadata)

            raw = (res.text or "").strip()
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
            if speaker not in ["Moderator", "Travel Agent"]:
                recent_speakers.append(speaker)
            if len(recent_speakers) >= max_count:
                break
        return recent_speakers

    def _is_discussion_stalling(self, history):
        if len(history) < 4:
            return False

        # Look at last few actual message bodies
        recent_texts = []
        for line in reversed(history):
            if ":" in line:
                speaker, msg = line.split(":", 1)
                if speaker.strip() not in ["Moderator", "Travel Agent"]:
                    recent_texts.append(msg.strip().lower())
            if len(recent_texts) >= 3:
                break

        if len(recent_texts) < 3:
            return False

        # Weak heuristic: very short repetitive push / reply turns
        keyword_sets = []
        for txt in recent_texts:
            tokens = set(
                word.strip(".,!?*'\"()")
                for word in txt.split()
                if len(word.strip(".,!?*'\"()")) > 3
            )
            keyword_sets.append(tokens)

        overlap_1 = len(keyword_sets[0].intersection(keyword_sets[1]))
        overlap_2 = len(keyword_sets[1].intersection(keyword_sets[2]))

        return overlap_1 >= 1 and overlap_2 >= 1

    def should_speak(self, history, last_speaker=None):
        if not history:
            return 0.0

        last_msg = history[-1].lower()
        score = 0.0

        focus_keywords = {
            "saving money": ["budget", "cheap", "cost", "price", "expensive", "save", "deal", "option a", "option d"],
            "safety first": ["safe", "safety", "risk", "reliable", "airline"],
            "quickest route": ["fast", "quick", "time", "duration", "book now", "soon", "direct", "option b"],
            "comfort": ["comfort", "seat", "legroom", "relaxed", "smooth", "comfortable", "option c"],
            "fewest layovers": ["layover", "direct", "stop", "connection", "option b"]
        }

        # 1. Topic / focus relevance
        for kw in focus_keywords.get(self.focus, []):
            if kw in last_msg:
                score += 0.35
                break

        # 2. Questions invite responses
        if "?" in history[-1]:
            score += 0.20

        # 3. Participation style
        participation_bonus = {
            "active": 0.25,
            "reactive": 0.15,
            "reserved": -0.05,
            "opinionated": 0.20
        }
        score += participation_bonus.get(self.participation, 0.0)

        # 4. Initiative
        score += 0.20 * self.initiative

        # 5. Fairness bonus if never spoke
        own_name_prefix = f"{self.name}:"
        has_spoken = any(line.startswith(own_name_prefix) for line in history)
        if not has_spoken:
            score += 0.30

        # 6. Recency penalties
        recent_speakers = self._extract_recent_speakers(history, max_count=3)
        if len(recent_speakers) >= 1 and recent_speakers[0] == self.name:
            score -= 0.45
        elif len(recent_speakers) >= 2 and recent_speakers[1] == self.name:
            score -= 0.20

        recent_count = sum(1 for s in recent_speakers if s == self.name)
        score -= 0.10 * recent_count

        # 7. If discussion is stalling, flexible / agreeable people may step in
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
- Acknowledge other participants when appropriate.
- You may compromise if it helps the group make progress.
- Do not invent exact facts that are not in the discussion history.
- Refer to the listed options if they exist.
- Do not repeat your previous point unless necessary.
- Keep the reply concise.
"""
        try:
            time.sleep(12)
            res = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )

            if hasattr(res, "usage_metadata"):
                self._update_log(res.usage_metadata)

            raw = (res.text or "").strip()
            return raw if raw else "[SILENCE]"
        except Exception as e:
            print(f"!! Turn Gen Error for {self.name}: {e}")
            return "[SILENCE]"