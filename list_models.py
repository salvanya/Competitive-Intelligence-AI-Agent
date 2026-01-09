"""
Phase 2: Temperature Experimentation (FIXED)
Key Changes:
1. Using gemini-1.5-flash (1500 RPD instead of 20)
2. Full output display instead of truncation
3. Better error messages
"""
import os
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

def create_chain(temperature: float, model: str = "gemini-1.5-flash"):
    """
    🎯 Model Selection Rationale:
    - gemini-1.5-flash: 1,500 RPD (best for experiments)
    - gemini-1.5-pro: 50 RPD (use for production quality)
    - gemini-2.5-flash: 20 RPD (avoid for development)
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=150,  # Increased for complete sentences
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst. Provide concise, single-sentence answers."),
        ("human", "Describe {company}'s pricing strategy in one sentence.")
    ])
    
    return prompt | llm | StrOutputParser()


def invoke_with_retry(
    chain, 
    input_data: dict, 
    max_retries: int = 2,
    base_delay: float = 5.0
) -> Optional[str]:
    """
    Retry logic with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        
        except ResourceExhausted as e:
            error_msg = str(e)
            
            # Check if it's daily quota vs rate limit
            if "PerDay" in error_msg:
                print(f"\n❌ DAILY QUOTA EXHAUSTED")
                print(f"   You've hit the daily limit for this model.")
                print(f"   Solution: Wait until tomorrow OR switch API keys")
                raise
            
            if attempt == max_retries - 1:
                print(f"❌ Failed after {max_retries} attempts")
                raise
            
            wait_time = base_delay * (2 ** attempt)
            print(f"⏳ Rate limit hit. Waiting {wait_time:.0f}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
            raise


def test_temperature(temperature: float, runs: int = 3):
    """
    Run multiple invocations to observe temperature effects.
    """
    print(f"\n{'='*70}")
    print(f"TEMPERATURE: {temperature}")
    print(f"{'='*70}\n")
    
    chain = create_chain(temperature)
    
    for i in range(runs):
        print(f"🔄 Run {i+1}:")
        try:
            response = invoke_with_retry(
                chain, 
                {"company": "Netflix"},
                max_retries=2,
                base_delay=5.0
            )
            
            # ✅ FIXED: Show full response, not truncated
            print(f"   Output: {response}")
            print(f"   Length: {len(response)} characters\n")
            
            # Small delay between runs to avoid rate limits
            if i < runs - 1:
                time.sleep(3)
                
        except ResourceExhausted as e:
            if "PerDay" in str(e):
                print("\n⛔ Cannot continue - daily quota exhausted")
                return  # Exit function
            print(f"Skipping run {i+1} due to rate limit\n")
            continue
        except Exception as e:
            print(f"Skipping run {i+1} due to error: {e}\n")
            continue


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Phase 2: Temperature Experimentation (Model: gemini-1.5-flash)")
    print("="*70)
    print("📊 Objective: Observe output diversity across temperature settings")
    print("🔒 Free Tier Limits: 1,500 requests/day, 15 requests/minute")
    print("="*70)
    
    # Test different temperatures
    temperatures = [0.0, 0.7, 1.0]
    
    for temp in temperatures:
        test_temperature(temp, runs=3)
        
        # Wait between temperature tests
        if temp != temperatures[-1]:
            print(f"\n⏳ Waiting 5s before next temperature test...\n")
            time.sleep(5)
    
    print("\n" + "="*70)
    print("✅ Temperature Experiment Complete!")
    print("="*70)
    print("\n📝 Analysis Questions:")
    print("   1. Did temp=0.0 produce identical outputs?")
    print("   2. How much variation did you see at temp=0.7?")
    print("   3. Was temp=1.0 noticeably more diverse?")
    print("\n💡 Save your observations - we'll need them for Phase 3!")