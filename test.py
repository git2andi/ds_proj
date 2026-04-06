# test.py
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("--- Available Models ---")
for model in client.models.list():
    # 'supported_actions' is the correct 2026 attribute
    print(f"Name: {model.name} | Actions: {model.supported_actions}")