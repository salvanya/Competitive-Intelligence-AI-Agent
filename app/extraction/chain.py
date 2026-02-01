"""
Deterministic Extraction Chain (Temperature=0)
LCEL chain for structured competitive intelligence extraction from web content
"""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.extraction.schemas import CompetitorProfile
from app.config import AppConfig


class ExtractionChain:
    """
    LCEL chain for structured data extraction from competitor websites.
    
    Uses deterministic extraction (temperature=0) with JSON mode enforcement
    to ensure reproducible, well-structured competitive intelligence data.
    
    Key Features:
    - Temperature=0 for deterministic output
    - JSON mode via response_mime_type
    - Pydantic validation for type safety
    - LCEL composition with pipe operator
    
    Attributes:
        config: Application configuration
        llm: Gemini LLM instance configured for extraction
        parser: Pydantic output parser
        prompt: Chat prompt template
        chain: Composed LCEL chain
    
    Example:
        >>> config = AppConfig(google_api_key="your-key")
        >>> extractor = ExtractionChain("your-key", config)
        >>> profile = await extractor.extract("https://example.com", content)
        >>> print(profile.company_name)
    """
    
    def __init__(self, api_key: str, config: AppConfig):
        """
        Initialize the extraction chain.
        
        Args:
            api_key: Google AI Studio API key
            config: Application configuration object
        
        Raises:
            ValueError: If API key is missing or invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.config = config
        
        # Initialize Gemini LLM with deterministic settings
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.EXTRACTION_TEMPERATURE,  # 0.0 for deterministic
            google_api_key=api_key,
            model_kwargs={
                "response_mime_type": "application/json"  # Enforce JSON output
            }
        )
        
        # Initialize Pydantic parser
        self.parser = PydanticOutputParser(pydantic_object=CompetitorProfile)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
        # Compose LCEL chain using pipe operator
        self.chain = (
            self.prompt.partial(
                format_instructions=self.parser.get_format_instructions()
            )
            | self.llm
            | self.parser
        )
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for extraction.
        
        Returns:
            str: System prompt with extraction instructions
        """
        return """You are a competitive intelligence extraction specialist with expertise in analyzing competitor websites.

Your task is to extract structured, factual information from website content with high precision and accuracy.

EXTRACTION RULES:
1. **Factual Only**: Extract ONLY information explicitly stated in the content
2. **No Inference**: Do not make assumptions or infer information not directly stated
3. **Null for Missing**: Use null/empty values for fields not found in content
4. **Exact Quotes**: For pricing, preserve exact format (e.g., "$99/month", "Contact Sales")
5. **Conciseness**: Keep features/USPs to 5-10 words each
6. **Technology Stack**: Extract mentioned frameworks, cloud providers, certifications (e.g., "AWS", "React", "SOC 2")

PRICING NORMALIZATION:
- If pricing says "Custom", "Enterprise", "Contact us", "Get a quote" → use "Contact Sales"
- If specific price is stated → preserve exact format including currency and period
- If no pricing found → leave pricing_tiers empty

FEATURE EXTRACTION:
- Focus on top 5-10 most important features
- Be specific and actionable (e.g., "Real-time collaboration" not "Collaboration")
- Avoid marketing fluff (e.g., skip "Best-in-class", "Industry-leading")

QUALITY CHECKS:
- Company name must not be empty or generic (e.g., not "Company", "Business")
- Tagline should be concise (under 150 characters)
- Target market should be specific (e.g., "Enterprise SaaS companies", not just "Businesses")

{format_instructions}

Remember: Precision and accuracy are critical. When in doubt, omit rather than guess."""
    
    def _get_human_prompt(self) -> str:
        """
        Get the human prompt template.
        
        Returns:
            str: Human prompt with input variables
        """
        return """Website URL: {url}

Website Content:
{content}

Extract competitive intelligence from this website following the system instructions.
Return valid JSON matching the CompetitorProfile schema."""
    
    async def extract(self, url: str, content: str) -> CompetitorProfile:
        """
        Extract structured competitor profile from scraped content.
        
        Invokes the LCEL chain asynchronously to extract structured data.
        Automatically sets the website_url field after extraction.
        
        Args:
            url: Website URL (used for reference, not extracted)
            content: Scraped website content (Markdown or text)
        
        Returns:
            CompetitorProfile: Validated competitor profile
        
        Raises:
            ValueError: If content is empty or too short
            Exception: If LLM extraction fails
        
        Example:
            >>> content = "# Acme Corp\\nThe best SaaS platform..."
            >>> profile = await extractor.extract("https://acme.com", content)
            >>> print(f"Extracted: {profile.company_name}")
        """
        # Validate inputs
        if not content or len(content.strip()) < 50:
            raise ValueError(
                f"Content too short for extraction: {len(content)} characters. "
                "Need at least 50 characters."
            )
        
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        try:
            # Invoke chain asynchronously
            result = await self.chain.ainvoke({
                "url": url,
                "content": content[:50000]  # Limit content to first 50k chars for safety
            })
            
            # Ensure URL is set correctly (in case LLM didn't extract it properly)
            result.website_url = url
            
            return result
        
        except Exception as e:
            # Create error profile if extraction fails
            error_profile = CompetitorProfile(
                company_name=url,  # Use URL as fallback name
                website_url=url,
                scrape_success=False,
                error_message=f"Extraction failed: {str(e)}"
            )
            return error_profile
    
    async def extract_batch(
        self, 
        url_content_pairs: list[tuple[str, str]]
    ) -> list[CompetitorProfile]:
        """
        Extract profiles from multiple URL/content pairs.
        
        Processes multiple extractions sequentially (to respect rate limits).
        Use in combination with RateLimiter for production use.
        
        Args:
            url_content_pairs: List of (url, content) tuples
        
        Returns:
            list[CompetitorProfile]: List of extracted profiles
        
        Example:
            >>> pairs = [
            ...     ("https://comp1.com", content1),
            ...     ("https://comp2.com", content2)
            ... ]
            >>> profiles = await extractor.extract_batch(pairs)
        """
        profiles = []
        
        for url, content in url_content_pairs:
            try:
                profile = await self.extract(url, content)
                profiles.append(profile)
            except Exception as e:
                # Create error profile and continue
                error_profile = CompetitorProfile(
                    company_name=url,
                    website_url=url,
                    scrape_success=False,
                    error_message=str(e)
                )
                profiles.append(error_profile)
        
        return profiles
    
    def get_chain_info(self) -> dict:
        """
        Get information about the extraction chain configuration.
        
        Useful for debugging and monitoring.
        
        Returns:
            dict: Chain configuration details
        
        Example:
            >>> info = extractor.get_chain_info()
            >>> print(f"Model: {info['model']}")
        """
        return {
            "model": self.config.GEMINI_MODEL,
            "temperature": self.config.EXTRACTION_TEMPERATURE,
            "schema": "CompetitorProfile",
            "json_mode": True,
            "deterministic": True
        }