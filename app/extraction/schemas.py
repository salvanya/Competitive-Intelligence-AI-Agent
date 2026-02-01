"""
Pydantic Models for Competitive Intelligence
Defines structured schemas for competitor data extraction and validation
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class PricingTier(BaseModel):
    """
    Individual pricing tier model.
    
    Represents a single pricing plan (e.g., Free, Pro, Enterprise)
    with normalization for "Contact Sales" pricing.
    
    Attributes:
        name: Tier name (e.g., 'Pro', 'Enterprise', 'Starter')
        price: Price string or 'Contact Sales' for custom pricing
        features: List of key features included in this tier
    
    Example:
        >>> tier = PricingTier(
        ...     name="Pro",
        ...     price="$99/month",
        ...     features=["Unlimited projects", "Priority support"]
        ... )
    """
    
    name: str = Field(
        description="Tier name (e.g., 'Pro', 'Enterprise', 'Starter')"
    )
    
    price: Optional[str] = Field(
        default=None,
        description="Price string (e.g., '$99/month') or 'Contact Sales'"
    )
    
    features: List[str] = Field(
        default_factory=list,
        description="Key features included in this tier"
    )
    
    @field_validator('price')
    @classmethod
    def normalize_price(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalize price field to handle custom/enterprise pricing.
        
        Converts variations like 'custom', 'enterprise', 'contact us'
        to standardized 'Contact Sales' for consistency.
        
        Args:
            v: Raw price value from extraction
        
        Returns:
            Optional[str]: Normalized price or None
        
        Example:
            >>> PricingTier.normalize_price("custom pricing")
            'Contact Sales'
            >>> PricingTier.normalize_price("$99")
            '$99'
        """
        if v is None:
            return None
        
        v = v.strip()
        
        # Normalize enterprise/custom pricing variations
        if v.lower() in ['contact sales', 'custom', 'enterprise', 'contact us', 
                         'get a quote', 'custom pricing', 'talk to sales']:
            return "Contact Sales"
        
        return v
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """
        Validate and clean feature list.
        
        Removes empty strings and trims whitespace.
        
        Args:
            v: List of features
        
        Returns:
            List[str]: Cleaned feature list
        """
        return [feature.strip() for feature in v if feature and feature.strip()]


class CompetitorProfile(BaseModel):
    """
    Complete competitor intelligence profile.
    
    Comprehensive structured representation of competitor analysis
    extracted from their website. Includes company information,
    product details, pricing, and technical stack.
    
    Attributes:
        company_name: Official company name
        website_url: Source URL that was scraped
        tagline: Company tagline or value proposition
        target_market: Target customer segment or market
        key_features: Top product/service features
        pricing_tiers: List of pricing plans
        unique_selling_points: Key differentiators and USPs
        technology_stack: Technologies, frameworks, certifications mentioned
        extraction_timestamp: ISO timestamp of when data was extracted
        scrape_success: Whether the scrape operation succeeded
        error_message: Error description if scrape failed
    
    Example:
        >>> profile = CompetitorProfile(
        ...     company_name="Acme Corp",
        ...     website_url="https://acme.com",
        ...     tagline="The best SaaS platform",
        ...     key_features=["Feature 1", "Feature 2"],
        ...     pricing_tiers=[PricingTier(name="Pro", price="$99")]
        ... )
    """
    
    company_name: str = Field(
        description="Official company name"
    )
    
    website_url: str = Field(
        description="Source URL that was scraped"
    )
    
    tagline: Optional[str] = Field(
        default=None,
        description="Company tagline or value proposition"
    )
    
    target_market: Optional[str] = Field(
        default=None,
        description="Target customer segment or market (e.g., 'SMBs', 'Enterprise', 'Developers')"
    )
    
    key_features: List[str] = Field(
        default_factory=list,
        description="Top product/service features (5-10 most important)"
    )
    
    pricing_tiers: List[PricingTier] = Field(
        default_factory=list,
        description="List of pricing plans/tiers"
    )
    
    unique_selling_points: List[str] = Field(
        default_factory=list,
        description="Key differentiators and unique selling propositions"
    )
    
    technology_stack: List[str] = Field(
        default_factory=list,
        description="Technologies, frameworks, cloud providers, certifications mentioned"
    )
    
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
    
    @field_validator('company_name')
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        """
        Validate company name is not empty.
        
        Args:
            v: Company name
        
        Returns:
            str: Validated company name
        
        Raises:
            ValueError: If company name is empty
        """
        v = v.strip()
        if not v:
            raise ValueError("Company name cannot be empty")
        return v
    
    @field_validator('website_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """
        Validate URL format.
        
        Args:
            v: Website URL
        
        Returns:
            str: Validated URL
        
        Raises:
            ValueError: If URL is invalid
        """
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    def get_summary(self) -> str:
        """
        Get a concise text summary of the competitor profile.
        
        Useful for logging, debugging, or quick review.
        
        Returns:
            str: Multi-line summary of the profile
        
        Example:
            >>> print(profile.get_summary())
            Company: Acme Corp
            URL: https://acme.com
            Features: 5
            Pricing Tiers: 3
            Status: ✅ Success
        """
        status = "✅ Success" if self.scrape_success else f"❌ Failed: {self.error_message}"
        
        summary_lines = [
            f"Company: {self.company_name}",
            f"URL: {self.website_url}",
            f"Tagline: {self.tagline or 'N/A'}",
            f"Target Market: {self.target_market or 'N/A'}",
            f"Features: {len(self.key_features)}",
            f"Pricing Tiers: {len(self.pricing_tiers)}",
            f"USPs: {len(self.unique_selling_points)}",
            f"Tech Stack: {len(self.technology_stack)}",
            f"Extracted: {self.extraction_timestamp}",
            f"Status: {status}"
        ]
        
        return "\n".join(summary_lines)
    
    def to_searchable_text(self) -> str:
        """
        Convert profile to searchable text for vector embeddings.
        
        Creates a comprehensive text representation that captures
        all important aspects of the competitor for semantic search.
        
        Returns:
            str: Formatted text suitable for embedding
        
        Example:
            >>> text = profile.to_searchable_text()
            >>> # Use for vector store ingestion
        """
        sections = [
            f"Company: {self.company_name}",
            f"Tagline: {self.tagline or 'N/A'}",
            f"Target Market: {self.target_market or 'N/A'}",
            f"Features: {', '.join(self.key_features)}",
            f"USPs: {', '.join(self.unique_selling_points)}",
            f"Tech Stack: {', '.join(self.technology_stack)}",
            f"Pricing: {', '.join([f'{t.name}: {t.price}' for t in self.pricing_tiers])}"
        ]
        
        return "\n".join(sections)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "company_name": "Acme SaaS",
                "website_url": "https://acmesaas.com",
                "tagline": "The all-in-one project management solution",
                "target_market": "Small to Medium Businesses",
                "key_features": [
                    "Real-time collaboration",
                    "Advanced analytics dashboard",
                    "Custom workflows",
                    "Mobile apps (iOS/Android)",
                    "API integrations"
                ],
                "pricing_tiers": [
                    {
                        "name": "Starter",
                        "price": "$29/month",
                        "features": ["5 projects", "10 team members", "Basic support"]
                    },
                    {
                        "name": "Pro",
                        "price": "$99/month",
                        "features": ["Unlimited projects", "50 team members", "Priority support"]
                    },
                    {
                        "name": "Enterprise",
                        "price": "Contact Sales",
                        "features": ["Custom limits", "Dedicated account manager", "SLA"]
                    }
                ],
                "unique_selling_points": [
                    "AI-powered task recommendations",
                    "99.9% uptime SLA",
                    "SOC 2 Type II certified"
                ],
                "technology_stack": [
                    "AWS",
                    "React",
                    "Python",
                    "PostgreSQL"
                ],
                "scrape_success": True,
                "error_message": None
            }
        }