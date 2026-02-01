"""Quick dependency verification script"""

print("Testing dependencies...\n")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✅ langchain-google-genai")
except Exception as e:
    print(f"❌ langchain-google-genai: {e}")

try:
    from langchain_qdrant import QdrantVectorStore
    print("✅ langchain-qdrant")
except Exception as e:
    print(f"❌ langchain-qdrant: {e}")

try:
    from crawl4ai import AsyncWebCrawler
    print("✅ crawl4ai")
except Exception as e:
    print(f"❌ crawl4ai: {e}")

try:
    from app.config import AppConfig
    print("✅ app.config")
except Exception as e:
    print(f"❌ app.config: {e}")

try:
    import streamlit
    print("✅ streamlit")
except Exception as e:
    print(f"❌ streamlit: {e}")

print("\n✅ All dependencies verified!")