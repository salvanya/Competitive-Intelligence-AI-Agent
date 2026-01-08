"""
Experiment 1: Temperature Control
Demonstrates how temperature affects output consistency and creativity
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# The prompt we'll use for all tests
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a competitive intelligence analyst."),
    ("human", "Describe {company}'s pricing strategy in 2-3 sentences.")
])

def test_temperature(temp: float, runs: int = 3):
    """Run the same prompt multiple times with a given temperature"""
    print(f"\n{'='*60}")
    print(f"TEMPERATURE: {temp}")
    print(f"{'='*60}\n")
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=temp,
        max_tokens=150,
    )
    
    chain = prompt | llm | StrOutputParser()
    
    for i in range(runs):
        print(f"Run {i+1}:")
        response = chain.invoke({"company": "Netflix"})
        print(f"{response}\n")

if __name__ == "__main__":
    print("TEMPERATURE EXPERIMENT")
    print("Question: Describe Netflix's pricing strategy")
    print("We'll run this 3 times at different temperatures\n")
    
    # Test deterministic (extraction use case)
    test_temperature(0.0, runs=3)
    
    # Test balanced (synthesis use case)
    test_temperature(0.7, runs=3)
    
    # Test creative (risky for production)
    test_temperature(1.0, runs=3)
    
    print("\n" + "="*60)
    print("🎓 KEY TAKEAWAY:")
    print("="*60)
    print("temp=0.0 → Identical outputs (perfect for extraction)")
    print("temp=0.7 → Varied but coherent (good for analysis)")
    print("temp=1.0 → Highly varied (too risky for production)")