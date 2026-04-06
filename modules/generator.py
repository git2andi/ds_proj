# modules/generator.py
import os
import datetime
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

class MultiUserSimulator:
    def __init__(self, persona, setting):
        self.name = persona["name"]
        self.behavior = persona["behavior"]
        self.setting = setting
        
        # Absolute path for logging usage (3W Efficiency)
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.log_file = os.path.join(project_root, "token_log.txt")
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        # Initialize without manual http_options to let the 2026 SDK auto-route
        self.client = genai.Client(api_key=api_key)
        
        # Use the high-performance 2026 preview model found in your ListModels
        self.model_id = "gemini-3-flash-preview" 
        
        # Initialize dynamic 'What' intelligence 
        self.goal = self._generate_initial_goal()

    def _update_log(self, usage):
        """Updates the token log with absolute path safety."""
        today = str(datetime.date.today())
        total_tokens = getattr(usage, 'total_token_count', 0)
        
        log_path = os.path.abspath(self.log_file)
        lines = []
        
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
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
                    continue
                break
        
        if not found:
            lines.append(f"{today} | Tokens: {total_tokens} | Calls: 1\n")

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _generate_initial_goal(self):
        """MUCA Sub-topic Generator: Defines the 'What' dimension."""
        prompt = f"Persona: {self.behavior}\nSetting: {self.setting}\nCreate a 1-sentence secret goal."
        try:
            # Use the SDK auto-routing for the preview model
            res = self.client.models.generate_content(model=self.model_id, contents=prompt)
            
            if hasattr(res, 'usage_metadata'):
                self._update_log(res.usage_metadata)
            
            raw_text = res.text if res.text is not None else ""
            return raw_text.strip() if raw_text else "Engage in the discussion."
        except Exception as e:
            # Fallback for API spikes or quota issues
            print(f"!! Goal Gen Error for {self.name}: {e}")
            return f"Act as {self.name} and pursue interests related to {self.setting}."

    def generate_turn(self, history):
        """MUCA 3W Dimension Logic: Deciding Who, When, and What."""
        chat_str = "\n".join(history)
        prompt = f"""
        Identity: {self.name} ({self.behavior})
        Goal: {self.goal}
        History: {chat_str}
        
        INSTRUCTIONS:
        1. If the discussion is active, you MUST contribute unless you truly have nothing to add.
        2. Keep your persona consistent.
        3. Format:
        Thought: <reasoning>
        Response: <text>
        """
        try:
            # Short sleep to mitigate 429 Resource Exhausted on Free Tier
            time.sleep(1)
            res = self.client.models.generate_content(model=self.model_id, contents=prompt)
            
            if hasattr(res, 'usage_metadata'):
                self._update_log(res.usage_metadata)
            
            raw = res.text if res.text is not None else ""
            if "Response:" in raw:
                parts = raw.split("Response:")
                return parts[0].replace("Thought:", "").strip(), parts[1].strip()
            
            return "Continuing conversation", raw.strip() if raw else "[SILENCE]"
        except Exception as e:
            print(f"!! Turn Gen Error for {self.name}: {e}")
            return "System Cooldown", "[SILENCE]"