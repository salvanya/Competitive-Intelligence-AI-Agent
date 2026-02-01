"""
Extraction Module
Handles structured data extraction from AI news articles using LLM chains
"""

from app.extraction.schemas import NewsArticleProfile, ImpactLevel
from app.extraction.chain import ExtractionChain

__all__ = [
    "NewsArticleProfile",
    "ImpactLevel",
    "ExtractionChain",
]