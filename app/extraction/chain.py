"""
Deterministic Extraction Chain (Temperature=0)
LCEL chain for structured AI news analysis extraction from article content
"""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.extraction.schemas import NewsArticleProfile
from app.config import AppConfig


class ExtractionChain:
    """
    LCEL chain for structured data extraction from AI news articles.
    
    Uses deterministic extraction (temperature=0) with JSON mode enforcement
    to ensure reproducible, well-structured news analysis data.
    
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
        >>> profile = await extractor.extract("https://techcrunch.com/...", content)
        >>> print(profile.headline)
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
        self.parser = PydanticOutputParser(pydantic_object=NewsArticleProfile)
        
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
        return """You are an AI news analysis specialist with expertise in extracting structured insights from technology and AI news articles.

Your task is to extract comprehensive, structured information from news articles with high precision and analytical depth.

EXTRACTION RULES:

1. **Factual Accuracy**: Extract ONLY information explicitly stated in the article
2. **No Inference**: Do not make assumptions beyond what is written
3. **Null for Missing**: Use null/empty values for fields not found in the content
4. **Exact Quotes**: Preserve exact terminology for technologies and key concepts
5. **Comprehensive Coverage**: Extract all relevant technologies, use cases, and industries mentioned

FIELD-SPECIFIC GUIDELINES:

**headline**: Extract the exact article title/headline

**news_source**: Identify the publication (e.g., "TechCrunch", "VentureBeat", "The Verge")

**publication_date**: Extract the publication date if mentioned (ISO format preferred, e.g., "2026-01-15")

**author**: Extract author name(s) if mentioned

**article_summary**: Create a concise 2-3 sentence summary capturing:
- What happened (the main announcement/development)
- Why it matters (significance)
- Key implications

**key_technologies**: List ALL AI technologies, models, frameworks, or tools mentioned:
- Examples: "GPT-4", "LangChain", "RAG", "Claude", "Stable Diffusion", "Transformers"
- Include versions when specified (e.g., "GPT-4 Turbo")
- Include technical terms (e.g., "Fine-tuning", "Prompt Engineering")

**use_cases**: Extract practical applications and use cases described:
- Be specific (e.g., "Medical diagnosis assistance" not just "Healthcare")
- Focus on actionable applications
- Include examples mentioned in the article

**affected_industries**: List industries that could be impacted:
- Examples: "Healthcare", "Finance", "Education", "Software Development", "Legal"
- Base on article content, not generic speculation

**potential_impact**: Assess overall impact level:
- "Critical": Paradigm-shifting developments (e.g., AGI breakthrough, major regulatory change)
- "High": Significant advancement with broad implications
- "Medium": Important development with specific applications
- "Low": Incremental improvement or niche application

**relevance_score** (0.0 to 1.0): Rate how relevant this is to AI practitioners/businesses:
- 0.9-1.0: Must-know development (major model release, regulatory change)
- 0.7-0.8: Very important (significant capability advancement)
- 0.5-0.6: Important (useful tool/technique)
- 0.3-0.4: Moderately interesting (niche application)
- 0.0-0.2: Low relevance (minor update)

**recommended_priority** (1-5): Suggest investigation priority:
- 1: Investigate immediately (time-sensitive, high impact)
- 2: Investigate soon (important but not urgent)
- 3: Investigate when relevant (useful to know)
- 4: Investigate if interested (nice to have)
- 5: Optional (low priority)

**key_insights**: Extract 3-5 main takeaways:
- Specific facts, statistics, or claims
- Notable quotes from experts
- Surprising or counterintuitive findings

**limitations_mentioned**: Extract any limitations, concerns, or risks discussed:
- Technical limitations
- Ethical concerns
- Safety issues
- Cost/accessibility barriers
- Regulatory challenges

QUALITY CHECKS:
- Headline must not be empty or generic
- Article summary should be 2-3 complete sentences
- At least 1 technology should be identified (if it's an AI article)
- Relevance score should reflect actual importance, not hype
- Impact assessment should be evidence-based

{format_instructions}

Remember: Precision and analytical depth are critical. Extract meaningful insights, not just surface-level information."""
    
    def _get_human_prompt(self) -> str:
        """
        Get the human prompt template.
        
        Returns:
            str: Human prompt with input variables
        """
        return """Article URL: {url}

Article Content:
{content}

Extract comprehensive AI news analysis from this article following the system instructions.
Return valid JSON matching the NewsArticleProfile schema."""
    
    async def extract(self, url: str, content: str) -> NewsArticleProfile:
        """
        Extract structured news article profile from scraped content.
        
        Invokes the LCEL chain asynchronously to extract structured data.
        Automatically sets the article_url field after extraction.
        
        Args:
            url: Article URL (used for reference, not extracted)
            content: Scraped article content (Markdown or text)
        
        Returns:
            NewsArticleProfile: Validated news article profile
        
        Raises:
            ValueError: If content is empty or too short
            Exception: If LLM extraction fails
        
        Example:
            >>> content = "# OpenAI Releases GPT-5\\nOpenAI announced..."
            >>> profile = await extractor.extract("https://techcrunch.com/...", content)
            >>> print(f"Extracted: {profile.headline}")
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
            result.article_url = url
            
            return result
        
        except Exception as e:
            # Create error profile if extraction fails
            error_profile = NewsArticleProfile(
                headline=url,  # Use URL as fallback headline
                article_url=url,
                news_source="Unknown",
                article_summary=f"Extraction failed: {str(e)}",
                scrape_success=False,
                error_message=f"Extraction failed: {str(e)}"
            )
            return error_profile
    
    async def extract_batch(
        self, 
        url_content_pairs: list[tuple[str, str]]
    ) -> list[NewsArticleProfile]:
        """
        Extract profiles from multiple URL/content pairs.
        
        Processes multiple extractions sequentially (to respect rate limits).
        Use in combination with RateLimiter for production use.
        
        Args:
            url_content_pairs: List of (url, content) tuples
        
        Returns:
            list[NewsArticleProfile]: List of extracted profiles
        
        Example:
            >>> pairs = [
            ...     ("https://techcrunch.com/ai1", content1),
            ...     ("https://venturebeat.com/ai2", content2)
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
                error_profile = NewsArticleProfile(
                    headline=url,
                    article_url=url,
                    news_source="Unknown",
                    article_summary=f"Extraction error: {str(e)}",
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
            "schema": "NewsArticleProfile",
            "json_mode": True,
            "deterministic": True
        }