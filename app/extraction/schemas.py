"""
Pydantic Models for AI News Analysis
Defines structured schemas for news article data extraction and validation
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ImpactLevel(str, Enum):
    """Enumeration for potential impact levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class NewsArticleProfile(BaseModel):
    """
    Complete AI news article profile.
    
    Comprehensive structured representation of an AI news article
    extracted from news websites. Includes article metadata,
    content analysis, relevance scoring, and impact assessment.
    
    Attributes:
        headline: Article headline/title
        article_url: Source URL of the article
        news_source: Publication name (e.g., "TechCrunch", "VentureBeat")
        publication_date: When the article was published
        author: Article author name(s)
        article_summary: Concise summary of the article (2-3 sentences)
        key_technologies: AI technologies mentioned (e.g., "GPT-4", "LangChain")
        use_cases: Practical applications and use cases described
        affected_industries: Industries that could be impacted
        potential_impact: Overall impact level assessment
        relevance_score: Relevance score from 0.0 to 1.0
        recommended_priority: Investigation priority ranking (1=highest, 5=lowest)
        key_insights: Main takeaways and insights
        limitations_mentioned: Any limitations or concerns discussed
        extraction_timestamp: ISO timestamp of when data was extracted
        scrape_success: Whether the scrape operation succeeded
        error_message: Error description if scrape failed
    
    Example:
        >>> profile = NewsArticleProfile(
        ...     headline="OpenAI Releases GPT-5 with Enhanced Reasoning",
        ...     article_url="https://techcrunch.com/...",
        ...     news_source="TechCrunch",
        ...     article_summary="OpenAI announced GPT-5...",
        ...     key_technologies=["GPT-5", "Transformers"],
        ...     potential_impact=ImpactLevel.HIGH,
        ...     relevance_score=0.95
        ... )
    """
    
    # Core Article Metadata
    headline: str = Field(
        description="Article headline or title"
    )
    
    article_url: str = Field(
        description="Source URL of the article"
    )
    
    news_source: str = Field(
        description="Publication name (e.g., 'TechCrunch', 'The Verge', 'VentureBeat')"
    )
    
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date in ISO format or human-readable format"
    )
    
    author: Optional[str] = Field(
        default=None,
        description="Article author name(s)"
    )
    
    # Content Analysis
    article_summary: str = Field(
        description="Concise 2-3 sentence summary of the article content"
    )
    
    key_technologies: List[str] = Field(
        default_factory=list,
        description="AI technologies, models, or frameworks mentioned (e.g., 'GPT-4', 'LangChain', 'RAG')"
    )
    
    use_cases: List[str] = Field(
        default_factory=list,
        description="Practical applications and use cases described in the article"
    )
    
    affected_industries: List[str] = Field(
        default_factory=list,
        description="Industries that could be impacted by the developments (e.g., 'Healthcare', 'Finance')"
    )
    
    # Impact Assessment
    potential_impact: Optional[ImpactLevel] = Field(
        default=None,
        description="Overall potential impact level (Low, Medium, High, Critical)"
    )
    
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)"
    )
    
    recommended_priority: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Recommended investigation priority (1=highest priority, 5=lowest)"
    )
    
    # Insights
    key_insights: List[str] = Field(
        default_factory=list,
        description="Main takeaways and insights from the article"
    )
    
    limitations_mentioned: List[str] = Field(
        default_factory=list,
        description="Any limitations, concerns, or risks mentioned"
    )
    
    # Metadata
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of data extraction"
    )
    
    scrape_success: bool = Field(
        default=True,
        description="Whether the scrape operation succeeded"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error description if scrape/extraction failed"
    )
    
    @field_validator('headline')
    @classmethod
    def validate_headline(cls, v: str) -> str:
        """
        Validate headline is not empty.
        
        Args:
            v: Headline text
        
        Returns:
            str: Validated headline
        
        Raises:
            ValueError: If headline is empty
        """
        v = v.strip()
        if not v:
            raise ValueError("Headline cannot be empty")
        return v
    
    @field_validator('article_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """
        Validate URL format.
        
        Args:
            v: Article URL
        
        Returns:
            str: Validated URL
        
        Raises:
            ValueError: If URL is invalid
        """
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @field_validator('relevance_score')
    @classmethod
    def validate_relevance_score(cls, v: Optional[float]) -> Optional[float]:
        """
        Validate relevance score is within range.
        
        Args:
            v: Relevance score
        
        Returns:
            Optional[float]: Validated score
        """
        if v is not None:
            if v < 0.0 or v > 1.0:
                raise ValueError("Relevance score must be between 0.0 and 1.0")
        return v
    
    @field_validator('key_technologies', 'use_cases', 'affected_industries', 
                     'key_insights', 'limitations_mentioned')
    @classmethod
    def validate_string_lists(cls, v: List[str]) -> List[str]:
        """
        Validate and clean string lists.
        
        Removes empty strings and trims whitespace.
        
        Args:
            v: List of strings
        
        Returns:
            List[str]: Cleaned list
        """
        return [item.strip() for item in v if item and item.strip()]
    
    def get_summary(self) -> str:
        """
        Get a concise text summary of the news article profile.
        
        Useful for logging, debugging, or quick review.
        
        Returns:
            str: Multi-line summary of the profile
        
        Example:
            >>> print(profile.get_summary())
            Headline: OpenAI Releases GPT-5
            Source: TechCrunch
            Impact: High
            Priority: 1
            Status: ✅ Success
        """
        status = "✅ Success" if self.scrape_success else f"❌ Failed: {self.error_message}"
        
        summary_lines = [
            f"Headline: {self.headline}",
            f"Source: {self.news_source}",
            f"URL: {self.article_url}",
            f"Published: {self.publication_date or 'N/A'}",
            f"Author: {self.author or 'N/A'}",
            f"Technologies: {len(self.key_technologies)}",
            f"Use Cases: {len(self.use_cases)}",
            f"Industries: {len(self.affected_industries)}",
            f"Impact: {self.potential_impact.value if self.potential_impact else 'N/A'}",
            f"Relevance: {self.relevance_score if self.relevance_score else 'N/A'}",
            f"Priority: {self.recommended_priority if self.recommended_priority else 'N/A'}",
            f"Extracted: {self.extraction_timestamp}",
            f"Status: {status}"
        ]
        
        return "\n".join(summary_lines)
    
    def to_searchable_text(self) -> str:
        """
        Convert profile to searchable text for vector embeddings.
        
        Creates a comprehensive text representation that captures
        all important aspects of the news article for semantic search.
        
        Returns:
            str: Formatted text suitable for embedding
        
        Example:
            >>> text = profile.to_searchable_text()
            >>> # Use for vector store ingestion
        """
        sections = [
            f"Headline: {self.headline}",
            f"Source: {self.news_source}",
            f"Summary: {self.article_summary}",
            f"Technologies: {', '.join(self.key_technologies)}",
            f"Use Cases: {', '.join(self.use_cases)}",
            f"Industries: {', '.join(self.affected_industries)}",
            f"Key Insights: {', '.join(self.key_insights)}",
            f"Impact Level: {self.potential_impact.value if self.potential_impact else 'N/A'}",
        ]
        
        if self.limitations_mentioned:
            sections.append(f"Limitations: {', '.join(self.limitations_mentioned)}")
        
        return "\n".join(sections)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "headline": "OpenAI Announces GPT-5 with Advanced Reasoning Capabilities",
                "article_url": "https://techcrunch.com/2026/01/15/openai-gpt5-announcement",
                "news_source": "TechCrunch",
                "publication_date": "2026-01-15",
                "author": "Kyle Wiggers",
                "article_summary": "OpenAI has unveiled GPT-5, featuring enhanced reasoning capabilities and multimodal understanding. The model shows significant improvements in complex problem-solving and code generation.",
                "key_technologies": [
                    "GPT-5",
                    "Multimodal AI",
                    "Advanced Reasoning",
                    "Code Generation"
                ],
                "use_cases": [
                    "Software development assistance",
                    "Complex data analysis",
                    "Scientific research support",
                    "Educational tutoring"
                ],
                "affected_industries": [
                    "Software Development",
                    "Education",
                    "Research",
                    "Healthcare",
                    "Finance"
                ],
                "potential_impact": "High",
                "relevance_score": 0.95,
                "recommended_priority": 1,
                "key_insights": [
                    "GPT-5 outperforms GPT-4 on reasoning benchmarks by 40%",
                    "New safety measures implemented to prevent misuse",
                    "API access available to enterprise customers in Q2 2026"
                ],
                "limitations_mentioned": [
                    "Still prone to occasional hallucinations",
                    "High computational requirements",
                    "Limited availability at launch"
                ],
                "scrape_success": True,
                "error_message": None
            }
        }