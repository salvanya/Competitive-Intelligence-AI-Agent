"""
Phase 2: Temperature Experimentation with Rate Limit Protection
Demonstrates: Temperature effects + Production-grade error handling
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

def create_chain(temperature: float, model: str = "models/flash-lite"):
    """
    Factory function for creating LCEL chains with different temperatures.
    
    Why Temperature Matters:
    - 0.0: Deterministic (extraction, classification)
    - 0.7: Creative (synthesis, brainstorming)
    - 1.0: Maximum diversity (creative writing)
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=100,  # Keep outputs short for experiments
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst. Provide concise answers."),
        ("human", "Describe {company}'s pricing strategy in one sentence.")
    ])
    
    return prompt | llm | StrOutputParser()


def invoke_with_retry(
    chain, 
    input_data: dict, 
    max_retries: int = 3,
    base_delay: float = 60.0
) -> Optional[str]:
    """
    Invoke chain with exponential backoff on rate limits.
    
    Production Pattern: Always wrap LLM calls in retry logic.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        
        except ResourceExhausted as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed after {max_retries} attempts")
                raise
            
            # Extract retry delay from error message
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"⏳ Rate limit hit. Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
            raise


def test_temperature(temperature: float, runs: int = 3):
    """
    Run multiple invocations to observe temperature effects.
    """
    print(f"\n{'='*60}")
    print(f"TEMPERATURE: {temperature}")
    print(f"{'='*60}\n")
    
    chain = create_chain(temperature)
    
    for i in range(runs):
        print(f"Run {i+1}:")
        try:
            response = invoke_with_retry(
                chain, 
                {"company": "Netflix"},
                max_retries=2,
                base_delay=60.0  # Wait 60s on first retry
            )
            print(response + "...\n")  # Show first 50 chars
            
            # Rate limit protection: Wait between runs
            if i < runs - 1:
                print("⏳ Waiting 15s to respect rate limits...\n")
                time.sleep(15)
                
        except Exception as e:
            print(f"Skipping run {i+1} due to error\n")
            continue


if __name__ == "__main__":
    print("🧪 Phase 2: Temperature Experimentation")
    print("="*60)
    print("📊 Objective: Observe how temperature affects output diversity")
    print("🔒 Rate Limit: 5 RPM (Free Tier) - Script includes delays")
    print("="*60)
    
    # Test different temperatures with delays
    temperatures = [0.0, 0.7, 1.0]
    
    for temp in temperatures:
        test_temperature(temp, runs=2)  # Reduced to 2 runs
        
        # Wait between temperature tests
        if temp != temperatures[-1]:
            print("\n⏳ Waiting 20s before next temperature test...\n")
            time.sleep(20)
    
    print("\n" + "="*60)
    print("✅ Temperature Experiment Complete!")
    print("="*60)
    print("\n📝 Key Observations to Note:")
    print("   1. Temperature=0.0: Should produce identical outputs")
    print("   2. Temperature=0.7: Balanced creativity and consistency")
    print("   3. Temperature=1.0: Maximum diversity between runs")