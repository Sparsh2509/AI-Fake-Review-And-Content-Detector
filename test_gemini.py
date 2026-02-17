import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API Key not found!")
else:
    print("✅ API Key Loaded")

# Configure Gemini
genai.configure(api_key=api_key)

# Use fast model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

try:
    response = model.generate_content("Say hello in one short sentence.")
    print("✅ Gemini Response:")
    print(response.text)
except Exception as e:
    print("❌ Error:", e)
