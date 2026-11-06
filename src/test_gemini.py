
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.7-flash"

prompt = """
Explain what hypertension means
in two simple sentences.
"""

for attempt in range(3):
    try:
        print(f"Attempt {attempt + 1}/3")

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        print("\n===== GEMINI RESPONSE =====")
        print(response.text)
        break

    except errors.ServerError as e:
        if attempt == 2:
            print("Gemini is currently unavailable.")
            print("Error:", e)
            break

        wait_time = 15 * (2 ** attempt)
        print(f"Server busy. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)

    except Exception as e:
        print("Unexpected error:", e)
        break
