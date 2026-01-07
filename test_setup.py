"""
Phase 1 Verification: LCEL Hello World (Fixed Model Name)
Tests: API connectivity, runnable chains, streaming
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Verify API key loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}\n")

# Step 1: Initialize Gemini with CORRECT model name
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",  # ✅ Fixed: Use -latest suffix
    temperature=0,  # Deterministic for testing
    max_tokens=256,
)

# Step 2: Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond concisely."),
    ("human", "{question}")
])

# Step 3: Build LCEL chain using pipe operator
chain = prompt | llm | StrOutputParser()

# Step 4: Test synchronous invocation
if __name__ == "__main__":
    print("🧪 Testing LCEL Chain...\n")
    
    try:
        response = chain.invoke({"question": "What is LangChain?"})
        print(f"✅ Response: {response}\n")
        
        # Test streaming (critical for UX in later phases)
        print("🌊 Testing Streaming...\n")
        for chunk in chain.stream({"question": "Count from 1 to 5"}):
            print(chunk, end="", flush=True)
        
        print("\n\n" + "="*60)
        print("✅ Phase 1 Complete: Environment verified!")
        print("="*60)
        print("✅ LCEL chains working correctly")
        print("✅ Streaming working correctly")
        print("✅ Gemini API connected successfully")
        print("✅ Ready for Phase 2: LLM Fundamentals\n")
        
    except Exception as e:
        print(f"❌ Error occurred: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify API key in .env file")
        print("   2. Check internet connection")
        print("   3. Confirm Gemini API is enabled in Google AI Studio")
        raise