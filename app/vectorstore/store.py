"""
Qdrant In-Memory Vector Store
Semantic search and storage for competitor profiles
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.extraction.schemas import CompetitorProfile
from app.config import AppConfig


class CompetitorVectorStore:
    """
    In-memory vector store for competitor profiles using Qdrant.
    
    Provides semantic search capabilities over competitor intelligence data.
    Uses Google's text-embedding-004 model for 768-dimensional embeddings.
    Data is stored in-memory and does not persist between sessions.
    
    Attributes:
        config: Application configuration
        client: Qdrant client instance (in-memory)
        embeddings: Google embeddings model
        vectorstore: LangChain Qdrant wrapper
    
    Example:
        >>> config = AppConfig(google_api_key="your-key")
        >>> store = CompetitorVectorStore("your-key", config)
        >>> await store.ingest_profile(profile)
        >>> results = await store.search_similar("SaaS pricing", k=3)
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
    
    async def ingest_profile(self, profile: CompetitorProfile) -> str:
        """
        Add a competitor profile to the vector store.
        
        Converts the profile to searchable text, generates embeddings,
        and stores in Qdrant for semantic search.
        
        Args:
            profile: Competitor profile to ingest
        
        Returns:
            str: Document ID (UUID) assigned to this profile
        
        Raises:
            ValueError: If profile is invalid or scrape failed
            Exception: If embedding or storage fails
        
        Example:
            >>> profile = CompetitorProfile(company_name="Acme", ...)
            >>> doc_id = await store.ingest_profile(profile)
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
                "company_name": profile.company_name,
                "website_url": profile.website_url,
                "target_market": profile.target_market or "",
                "num_features": len(profile.key_features),
                "num_pricing_tiers": len(profile.pricing_tiers),
                "num_usps": len(profile.unique_selling_points),
                "has_tech_stack": len(profile.technology_stack) > 0,
                "extraction_timestamp": profile.extraction_timestamp,
                "profile_id": str(uuid4())
            }
        )
        
        # Add to vector store
        ids = await self.vectorstore.aadd_documents([doc])
        
        return ids[0]
    
    def _create_searchable_text(self, profile: CompetitorProfile) -> str:
        """
        Create searchable text representation of a profile.
        
        Formats all profile fields into a comprehensive text block
        that captures semantic meaning for embedding.
        
        Args:
            profile: Competitor profile
        
        Returns:
            str: Formatted searchable text
        """
        sections = [
            f"Company: {profile.company_name}",
            f"Tagline: {profile.tagline or 'N/A'}",
            f"Target Market: {profile.target_market or 'N/A'}",
            "",
            "Key Features:",
            "\n".join([f"- {feature}" for feature in profile.key_features]),
            "",
            "Unique Selling Points:",
            "\n".join([f"- {usp}" for usp in profile.unique_selling_points]),
            "",
            "Technology Stack:",
            ", ".join(profile.technology_stack) if profile.technology_stack else "N/A",
            "",
            "Pricing Tiers:",
        ]
        
        # Add pricing information
        for tier in profile.pricing_tiers:
            tier_text = f"- {tier.name}: {tier.price or 'N/A'}"
            if tier.features:
                tier_text += f" (Features: {', '.join(tier.features[:3])})"
            sections.append(tier_text)
        
        return "\n".join(sections)
    
    async def search_similar(
        self, 
        query: str, 
        k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar competitors using semantic search.
        
        Performs cosine similarity search over embedded profiles
        to find the most relevant matches to the query.
        
        Args:
            query: Search query (e.g., "SaaS pricing strategies")
            k: Number of results to return (default: 3)
            filter_metadata: Optional metadata filters (e.g., {"target_market": "Enterprise"})
        
        Returns:
            List[Dict]: List of search results with company, score, and content
        
        Example:
            >>> results = await store.search_similar("enterprise pricing", k=2)
            >>> for result in results:
            ...     print(f"{result['company']}: {result['similarity_score']:.3f}")
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
                    "company": doc.metadata.get("company_name", "Unknown"),
                    "website_url": doc.metadata.get("website_url", ""),
                    "similarity_score": float(score),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "target_market": doc.metadata.get("target_market", "N/A"),
                    "num_features": doc.metadata.get("num_features", 0),
                })
            
            return formatted_results
        
        except Exception as e:
            # Return empty results on error rather than failing
            print(f"Search error: {str(e)}")
            return []
    
    async def get_all_profiles(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored competitor profiles.
        
        Useful for generating comprehensive reports or debugging.
        
        Returns:
            List[Dict]: List of all stored profiles with metadata
        
        Example:
            >>> all_profiles = await store.get_all_profiles()
            >>> print(f"Total competitors: {len(all_profiles)}")
        """
        try:
            # Use a broad query to get all documents
            results = await self.vectorstore.asimilarity_search(
                "company business product",
                k=100  # Assume max 100 competitors per session
            )
            
            return [
                {
                    "company": doc.metadata.get("company_name", "Unknown"),
                    "website_url": doc.metadata.get("website_url", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        
        except Exception as e:
            print(f"Error retrieving profiles: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dict: Collection statistics including count, vector size, etc.
        
        Example:
            >>> stats = store.get_collection_stats()
            >>> print(f"Stored profiles: {stats['vectors_count']}")
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