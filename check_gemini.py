from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
c = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

models_to_try = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
]

for model in models_to_try:
    try:
        r = c.models.generate_content(
            model=model,
            contents="Say: working"
        )
        print(f"✅ {model} — {r.text.strip()}")
    except Exception as e:
        err = str(e)[:60]
        print(f"❌ {model} — {err}")