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
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}\n")

# Step 1: Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    temperature=0,  # Deterministic for testing
    # top_p=0.9, # Filters unlikely words
    # top_k=40, # Only considers the 40 best options per word
    max_tokens=256,
)

print(f"✅ Using model: models/gemini-2.0-flash-lite")

# Step 2: Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    # SYSTEM: Defines the behavior, tone, and limits of the model.
    ("system", "You are a helpful assistant. Respond concisely."), 
    
    # HUMAN: Represents user input.
    # ("human", "Help me with this task."),

    # AI: Used for "few-shot prompting" or to preload a previous AI response.
    # ("ai", "I'm here to help you with any task you need."),
    
    # Placeholder for history (useful in agents with memory)
    # ("placeholder", "{chat_history}")
    
    # HUMAN: The main user query with a variable.
    ("human", "{question}") 
])

# Step 3: Build LCEL chain using pipe operator
chain = prompt | llm | StrOutputParser()

# Step 4: Test synchronous invocation
if __name__ == "__main__":
    print("Testing LCEL Chain...\n")
    
    try:
        response = chain.invoke({"question": "What is LangChain in one sentence?"})
        print(f"✅ Response: {response}\n")
        
        # Test streaming
        print("Testing Streaming...\n")
        print("Output: ", end="")
        for chunk in chain.stream({"question": "Count from 1 to 5, separated by commas"}):
            print(chunk, end="", flush=True)
        
        print("\n\n" + "="*60)
        print("✅ Phase 1 Complete: Environment verified")
        print("="*60)
        print("✅ Model: Gemini 2.5 Flash")
        print("✅ LCEL chains working correctly")
        print("✅ Streaming working correctly")
        print("✅ Gemini API connected successfully")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error occurred: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify API key in .env file")
        print("   2. Check internet connection")
        print("   3. Confirm Gemini API is enabled in Google AI Studio")
        raise