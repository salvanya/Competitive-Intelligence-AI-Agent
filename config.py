"""
Project Configuration
Centralized settings for the Competitive Intelligence Agent
Updated for Free Tier Compatibility
"""

# Model Selection (Free Tier Optimized)
EXTRACTION_MODEL = "models/gemini-2.5-flash-lite"  # Fast, structured output
SYNTHESIS_MODEL = "models/gemini-2.5-flash-lite"    # Same model for synthesis
EMBEDDING_MODEL = "models/text-embedding-004"  # For vector store (Phase 5)

# LLM Parameters
EXTRACTION_CONFIG = {
    "temperature": 0.0,      # Deterministic extraction
    "max_tokens": 2048,
    "top_p": 1.0,
}

SYNTHESIS_CONFIG = {
    "temperature": 0.7,      # Creative reasoning
    "max_tokens": 4096,
    "top_p": 0.95,
}

# API Settings
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

# Rate Limiting (Free Tier)
REQUESTS_PER_MINUTE = 15
DAILY_TOKEN_LIMIT = 1_000_000