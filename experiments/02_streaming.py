# experiments/02_streaming.py
"""
Phase 2.2: Streaming vs Batch Comparison
Demonstrates real-time token streaming for production UX
"""
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def create_synthesis_chain(temperature: float = 0.7):
    """
    Synthesis chain for competitive analysis (uses temp=0.7)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        max_tokens=500,  # Longer output to see streaming effect
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a competitive intelligence analyst. 
        Write detailed, professional analysis with specific insights.
        Use varied vocabulary and natural transitions."""),
        ("human", """Analyze {company}'s competitive positioning in the streaming market. 
        Cover: pricing strategy, content differentiation, and market threats.
        Write 3-4 paragraphs.""")
    ])
    
    return prompt | llm | StrOutputParser()


def demonstrate_batch():
    """
    Traditional batch processing - user sees nothing until complete
    """
    print("\n" + "="*70)
    print("🔄 BATCH MODE (Traditional)")
    print("="*70)
    print("⏳ Generating report... (you see nothing until it's done)\n")
    
    chain = create_synthesis_chain()
    
    start_time = time.time()
    response = chain.invoke({"company": "Netflix"})
    elapsed = time.time() - start_time
    
    print(f"✅ Report generated in {elapsed:.2f}s")
    print("\n📄 Full Report:\n")
    print(response)
    print(f"\n⏱️  Total time: {elapsed:.2f}s")


def demonstrate_streaming():
    """
    Streaming mode - tokens appear in real-time
    """
    print("\n" + "="*70)
    print("🌊 STREAMING MODE (Production UX)")
    print("="*70)
    print("📡 Tokens appear as generated (watch the flow):\n")
    
    chain = create_synthesis_chain()
    
    start_time = time.time()
    token_count = 0
    
    for chunk in chain.stream({"company": "Netflix"}):
        print(chunk, end="", flush=True)  # Real-time display
        token_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Streamed {token_count} chunks")
    print(f"💡 User saw first token in ~1-2s vs waiting {elapsed:.2f}s")


def demonstrate_streaming_with_inline_processing():
    """
    Advanced: Stream with processing (e.g., save to DB, update UI)
    """
    print("\n" + "="*70)
    print("🔧 STREAMING + PROCESSING (Advanced Pattern)")
    print("="*70)
    print("📡 Streaming with real-time processing:\n")
    
    chain = create_synthesis_chain()
    
    chunks = []
    for i, chunk in enumerate(chain.stream({"company": "Netflix"})):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
        
        # Simulate real-time processing (e.g., updating database)
        # Auto-save: Save progress every 10  chunks
        if i % 10 == 0 and i > 0:
            print(f" [checkpoint: {len(''.join(chunks))} chars saved]", end="")
    
    full_response = "".join(chunks)
    print(f"\n\n✅ Saved {len(full_response)} characters to database")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Phase 2.2: Streaming vs Batch Experiment")
    print("="*70)
    print("🎯 Objective: Compare user experience between approaches")
    print("⚡ Model:gemini-2.0-flash (temp=0.7)")
    print("="*70)
    
    try:
        # Experiment 1: Batch (bad UX)
        demonstrate_batch()
        
        time.sleep(3)  # Rate limit protection
        
        # Experiment 2: Streaming (good UX)
        demonstrate_streaming()
        
        time.sleep(3)
        
        # Experiment 3: Streaming with processing
        demonstrate_streaming_with_inline_processing()
        
        print("\n" + "="*70)
        print("✅ Streaming Experiment Complete!")
        print("="*70)
        print("\n📝 Key Takeaways:")
        print("   1. Batch: Simple but poor UX for long responses")
        print("   2. Streaming: Better UX, users see progress immediately")
        print("   3. Production: Always use streaming for synthesis (>5s)")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   - Check if you hit rate limits (wait 60s)")
        print("   - Verify API key in .env")
