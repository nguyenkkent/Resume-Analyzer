import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

# Optionally use "gemini-2.5-flash"
model = genai.GenerativeModel("gemini-2.0-flash-001") 
print(model.generate_content("Say hello in 5 words.").text)

