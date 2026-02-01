"""
Vector Store Module
Handles semantic search and storage of news article profiles using Qdrant
"""

from app.vectorstore.store import NewsVectorStore

__all__ = [
    "NewsVectorStore",
]