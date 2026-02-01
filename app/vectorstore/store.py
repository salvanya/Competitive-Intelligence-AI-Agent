"""
Qdrant In-Memory Vector Store
Semantic search and storage for AI news article profiles
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.extraction.schemas import NewsArticleProfile
from app.config import AppConfig


class NewsVectorStore:
    """
    In-memory vector store for AI news article profiles using Qdrant.
    
    Provides semantic search capabilities over news article data.
    Uses Google's text-embedding-004 model for 768-dimensional embeddings.
    Data is stored in-memory and does not persist between sessions.
    
    Attributes:
        config: Application configuration
        client: Qdrant client instance (in-memory)
        embeddings: Google embeddings model
        vectorstore: LangChain Qdrant wrapper
    
    Example:
        >>> config = AppConfig(google_api_key="your-key")
        >>> store = NewsVectorStore("your-key", config)
        >>> await store.ingest_article(profile)
        >>> results = await store.search_similar("GPT-5 release", k=3)
    """
    
    def __init__(self, api_key: str, config: AppConfig):
        """
        Initialize the vector store.
        
        Creates an in-memory Qdrant collection and initializes the
        embedding model for semantic search.
        
        Args:
            api_key: Google AI Studio API key
            config: Application configuration
        
        Raises:
            ValueError: If API key is invalid
            Exception: If Qdrant initialization fails
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.config = config
        
        # Initialize in-memory Qdrant client
        self.client = QdrantClient(location=":memory:")
        
        # Initialize Google embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=api_key,
            task_type="retrieval_document"  # Optimized for document retrieval
        )
        
        # Create collection
        self._initialize_collection()
        
        # Initialize LangChain wrapper
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding=self.embeddings
        )
    
    def _initialize_collection(self) -> None:
        """
        Create Qdrant collection with proper configuration.
        
        Sets up a collection with cosine distance metric for semantic
        similarity search. Handles case where collection already exists.
        
        Raises:
            Exception: If collection creation fails for unexpected reasons
        """
        try:
            self.client.create_collection(
                collection_name=self.config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.config.VECTOR_SIZE,  # 768 for text-embedding-004
                    distance=Distance.COSINE  # Cosine similarity for semantic search
                )
            )
        except Exception as e:
            # Collection may already exist - this is fine for in-memory
            # If it's a real error, it will surface on first operation
            pass
    
    async def ingest_article(self, profile: NewsArticleProfile) -> str:
        """
        Add a news article profile to the vector store.
        
        Converts the profile to searchable text, generates embeddings,
        and stores in Qdrant for semantic search.
        
        Args:
            profile: News article profile to ingest
        
        Returns:
            str: Document ID (UUID) assigned to this profile
        
        Raises:
            ValueError: If profile is invalid or scrape failed
            Exception: If embedding or storage fails
        
        Example:
            >>> profile = NewsArticleProfile(headline="GPT-5 Release", ...)
            >>> doc_id = await store.ingest_article(profile)
            >>> print(f"Stored with ID: {doc_id}")
        """
        # Skip profiles from failed scrapes
        if not profile.scrape_success:
            raise ValueError(
                f"Cannot ingest failed profile: {profile.error_message}"
            )
        
        # Create searchable text representation
        searchable_text = self._create_searchable_text(profile)
        
        # Create LangChain document with metadata
        doc = Document(
            page_content=searchable_text,
            metadata={
                "headline": profile.headline,
                "article_url": profile.article_url,
                "news_source": profile.news_source,
                "publication_date": profile.publication_date or "",
                "author": profile.author or "",
                "num_technologies": len(profile.key_technologies),
                "num_use_cases": len(profile.use_cases),
                "num_industries": len(profile.affected_industries),
                "potential_impact": profile.potential_impact.value if profile.potential_impact else "",
                "relevance_score": profile.relevance_score or 0.0,
                "recommended_priority": profile.recommended_priority or 5,
                "extraction_timestamp": profile.extraction_timestamp,
                "profile_id": str(uuid4())
            }
        )
        
        # Add to vector store
        ids = await self.vectorstore.aadd_documents([doc])
        
        return ids[0]
    
    def _create_searchable_text(self, profile: NewsArticleProfile) -> str:
        """
        Create searchable text representation of a news article profile.
        
        Formats all profile fields into a comprehensive text block
        that captures semantic meaning for embedding.
        
        Args:
            profile: News article profile
        
        Returns:
            str: Formatted searchable text
        """
        sections = [
            f"Headline: {profile.headline}",
            f"News Source: {profile.news_source}",
            f"Publication Date: {profile.publication_date or 'N/A'}",
            f"Author: {profile.author or 'N/A'}",
            "",
            "Article Summary:",
            profile.article_summary,
            "",
            "Key Technologies:",
            "\n".join([f"- {tech}" for tech in profile.key_technologies]),
            "",
            "Use Cases:",
            "\n".join([f"- {use_case}" for use_case in profile.use_cases]),
            "",
            "Affected Industries:",
            ", ".join(profile.affected_industries) if profile.affected_industries else "N/A",
            "",
            "Key Insights:",
            "\n".join([f"- {insight}" for insight in profile.key_insights]),
            "",
        ]
        
        # Add impact and relevance information
        if profile.potential_impact:
            sections.append(f"Potential Impact: {profile.potential_impact.value}")
        
        if profile.relevance_score is not None:
            sections.append(f"Relevance Score: {profile.relevance_score:.2f}")
        
        if profile.recommended_priority is not None:
            sections.append(f"Priority: {profile.recommended_priority}")
        
        # Add limitations if mentioned
        if profile.limitations_mentioned:
            sections.append("")
            sections.append("Limitations Mentioned:")
            sections.extend([f"- {limitation}" for limitation in profile.limitations_mentioned])
        
        return "\n".join(sections)
    
    async def search_similar(
        self, 
        query: str, 
        k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar news articles using semantic search.
        
        Performs cosine similarity search over embedded profiles
        to find the most relevant matches to the query.
        
        Args:
            query: Search query (e.g., "LLM reasoning improvements")
            k: Number of results to return (default: 3)
            filter_metadata: Optional metadata filters (e.g., {"news_source": "TechCrunch"})
        
        Returns:
            List[Dict]: List of search results with headline, score, and content
        
        Example:
            >>> results = await store.search_similar("GPT-5 capabilities", k=2)
            >>> for result in results:
            ...     print(f"{result['headline']}: {result['similarity_score']:.3f}")
        """
        if not query or not query.strip():
            return []
        
        try:
            # Perform similarity search with scores
            results = await self.vectorstore.asimilarity_search_with_score(
                query, 
                k=k,
                filter=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "headline": doc.metadata.get("headline", "Unknown"),
                    "article_url": doc.metadata.get("article_url", ""),
                    "news_source": doc.metadata.get("news_source", "Unknown"),
                    "similarity_score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "publication_date": doc.metadata.get("publication_date", "N/A"),
                    "potential_impact": doc.metadata.get("potential_impact", "N/A"),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "recommended_priority": doc.metadata.get("recommended_priority", 5),
                    "num_technologies": doc.metadata.get("num_technologies", 0),
                })
            
            return formatted_results
        
        except Exception as e:
            # Return empty results on error rather than failing
            print(f"Search error: {str(e)}")
            return []
    
    async def get_all_articles(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored news article profiles.
        
        Useful for generating comprehensive reports or debugging.
        
        Returns:
            List[Dict]: List of all stored profiles with metadata
        
        Example:
            >>> all_articles = await store.get_all_articles()
            >>> print(f"Total articles: {len(all_articles)}")
        """
        try:
            # Use a broad query to get all documents
            results = await self.vectorstore.asimilarity_search(
                "artificial intelligence news technology",
                k=100  # Assume max 100 articles per session
            )
            
            return [
                {
                    "headline": doc.metadata.get("headline", "Unknown"),
                    "article_url": doc.metadata.get("article_url", ""),
                    "news_source": doc.metadata.get("news_source", "Unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        
        except Exception as e:
            print(f"Error retrieving articles: {str(e)}")
            return []
    
    async def search_by_technology(self, technology: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles mentioning a specific technology.
        
        Args:
            technology: Technology name (e.g., "GPT-4", "LangChain")
            k: Number of results to return
        
        Returns:
            List[Dict]: Matching articles
        
        Example:
            >>> articles = await store.search_by_technology("GPT-5", k=3)
        """
        query = f"articles about {technology} technology development news"
        return await self.search_similar(query, k=k)
    
    async def search_by_impact(self, impact_level: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles with specific impact level.
        
        Args:
            impact_level: Impact level ("Low", "Medium", "High", "Critical")
            k: Number of results to return
        
        Returns:
            List[Dict]: Matching articles
        
        Example:
            >>> high_impact = await store.search_by_impact("High", k=5)
        """
        filter_metadata = {"potential_impact": impact_level}
        return await self.search_similar(
            "high impact AI developments",
            k=k,
            filter_metadata=filter_metadata
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dict: Collection statistics including count, vector size, etc.
        
        Example:
            >>> stats = store.get_collection_stats()
            >>> print(f"Stored articles: {stats['vectors_count']}")
        """
        try:
            collection_info = self.client.get_collection(
                collection_name=self.config.COLLECTION_NAME
            )
            
            return {
                "collection_name": self.config.COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "vector_size": self.config.VECTOR_SIZE,
                "status": collection_info.status.value if collection_info.status else "unknown"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "collection_name": self.config.COLLECTION_NAME
            }
    
    async def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Useful for starting fresh or cleaning up after testing.
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            >>> success = await store.clear_collection()
            >>> if success:
            ...     print("Collection cleared")
        """
        try:
            self.client.delete_collection(
                collection_name=self.config.COLLECTION_NAME
            )
            # Recreate empty collection
            self._initialize_collection()
            
            # Reinitialize vectorstore
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.config.COLLECTION_NAME,
                embedding=self.embeddings
            )
            
            return True
        
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False