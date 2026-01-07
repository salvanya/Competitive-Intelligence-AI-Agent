"""
Diagnostic Script: List Available Gemini Models
This will show which models your API key has access to
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("🔍 Listing all available models for your API key:\n")
print("="*60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description[:80]}...")
        print()

print("="*60)
print("\n💡 Use the model name WITHOUT 'models/' prefix in LangChain")