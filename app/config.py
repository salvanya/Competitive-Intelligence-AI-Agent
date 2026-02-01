"""
Project Configuration
Centralized settings for the Competitive Intelligence Agent
"""

from typing import Optional
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """
    Centralized configuration for the Competitive Intelligence Agent.
    
    Uses Pydantic for validation and type safety. All configuration parameters
    are defined here to avoid magic numbers scattered throughout the codebase.
    
    Attributes:
        GEMINI_MODEL: The Gemini model identifier for LLM operations
        EMBEDDING_MODEL: The embedding model for vector operations
        EXTRACTION_TEMPERATURE: Temperature for deterministic extraction (0.0)
        SYNTHESIS_TEMPERATURE: Temperature for creative synthesis (0.7)
        MAX_RPM: Maximum requests per minute (free tier limit)
        MAX_COMPETITORS: Maximum number of competitor URLs to analyze
        SCRAPE_TIMEOUT: Timeout in seconds for web scraping operations
        VECTOR_SIZE: Embedding dimension size for Gemini embeddings
        COLLECTION_NAME: Qdrant collection name for vector storage
        google_api_key: Google AI Studio API key (loaded from session state)
    """
    
    # LLM Configuration
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for all LLM operations (1M context window)"
    )
    
    EMBEDDING_MODEL: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model for vector operations (768 dimensions)"
    )
    
    # Temperature Settings
    EXTRACTION_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for extraction chain (deterministic)"
    )
    
    SYNTHESIS_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for synthesis agent (creative reasoning)"
    )
    
    # Rate Limiting (Google AI Studio Free Tier)
    MAX_RPM: int = Field(
        default=15,
        ge=1,
        description="Maximum requests per minute (free tier: 15 RPM)"
    )
    
    # Scraping Configuration
    MAX_COMPETITORS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of competitor URLs to analyze per session"
    )
    
    SCRAPE_TIMEOUT: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Timeout in seconds for web scraping operations"
    )
    
    # Vector Store Configuration
    VECTOR_SIZE: int = Field(
        default=768,
        description="Embedding dimension size for Gemini text-embedding-004"
    )
    
    COLLECTION_NAME: str = Field(
        default="competitors",
        description="Qdrant collection name for competitor profiles"
    )
    
    # API Keys (loaded dynamically from session state)
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google AI Studio API key (session-based, not persisted)"
    )
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        arbitrary_types_allowed = True
        
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is present.
        
        Returns:
            bool: True if API key is set, False otherwise
        """
        return self.google_api_key is not None and len(self.google_api_key) > 0
    
    def get_extraction_config(self) -> dict:
        """
        Get configuration dictionary for extraction chain.
        
        Returns:
            dict: Configuration parameters for extraction LLM
        """
        return {
            "model": self.GEMINI_MODEL,
            "temperature": self.EXTRACTION_TEMPERATURE,
            "google_api_key": self.google_api_key,
        }
    
    def get_synthesis_config(self) -> dict:
        """
        Get configuration dictionary for synthesis agent.
        
        Returns:
            dict: Configuration parameters for synthesis LLM
        """
        return {
            "model": self.GEMINI_MODEL,
            "temperature": self.SYNTHESIS_TEMPERATURE,
            "google_api_key": self.google_api_key,
        }
    
    def get_embedding_config(self) -> dict:
        """
        Get configuration dictionary for embeddings.
        
        Returns:
            dict: Configuration parameters for embedding model
        """
        return {
            "model": self.EMBEDDING_MODEL,
            "google_api_key": self.google_api_key,
        }