"""
Synthesis Agent with Streaming
Generates comprehensive AI news analysis reports using few-shot prompting
"""

from typing import List, AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import AppConfig
from app.extraction.schemas import NewsArticleProfile


class SynthesisAgent:
    """
    Generate AI news analysis reports with streaming output.
    
    Uses few-shot prompting with example reports to guide the LLM
    in producing high-quality, structured news intelligence analysis.
    Streams output token-by-token for real-time UI updates.
    
    Key Features:
    - Temperature=0.7 for creative reasoning
    - Few-shot examples for consistent structure
    - Streaming for real-time feedback
    - Comprehensive analysis sections
    
    Attributes:
        config: Application configuration
        llm: Gemini LLM configured for synthesis
        prompt: Chat prompt with few-shot examples
    
    Example:
        >>> agent = SynthesisAgent(api_key, config)
        >>> async for chunk in agent.generate_report_stream(profiles, objective):
        ...     print(chunk, end="")
    """
    
    def __init__(self, api_key: str, config: AppConfig):
        """
        Initialize the synthesis agent.
        
        Args:
            api_key: Google AI Studio API key
            config: Application configuration
        
        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.config = config
        
        # Initialize LLM with creative temperature for synthesis
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.SYNTHESIS_TEMPERATURE,  # 0.7 for creative insights
            google_api_key=api_key
        )
        
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create synthesis prompt with few-shot examples.
        
        Returns:
            ChatPromptTemplate: Configured prompt
        """
        system_prompt = """You are an expert AI technology analyst specializing in synthesizing news intelligence into actionable business insights.

Your task is to generate comprehensive AI news analysis reports that help decision-makers understand emerging trends, assess impact, and prioritize investigation.

**Report Structure:**

# Executive Summary
[2-3 paragraph overview of the most critical developments and key takeaways]

# Key AI Developments
[Detailed analysis of the most significant news items]

## Critical Developments (High/Critical Impact)
[Focus on paradigm-shifting announcements, major model releases, regulatory changes]

## Notable Developments (Medium Impact)
[Important advancements with specific applications]

## Emerging Trends
[Early-stage developments worth monitoring]

# Technology Trend Analysis
[Deep dive into technology patterns and trends]

## Most Mentioned Technologies
[Frequency analysis with context - why is this trending?]

## Emerging Technologies
[New or rarely mentioned technologies gaining traction]

## Technology Convergence Patterns
[How different technologies are being combined]

# Industry Impact Assessment
[Analysis of how AI developments affect different sectors]

## High-Impact Industries
[Industries facing significant transformation]

## Sector-Specific Insights
[Industry-by-industry breakdown of relevant developments]

## Cross-Industry Patterns
[Common themes across multiple industries]

# Use Case Opportunities
[Practical applications and business opportunities]

## Proven Use Cases
[Applications already showing results]

## Experimental Use Cases
[Novel applications in early stages]

## Gap Analysis
[Underserved use cases or market opportunities]

# Recommended Investigation Priority
[Actionable reading order with rationale]

## Priority 1: Investigate Immediately
[Time-sensitive or high-impact items requiring immediate attention]

## Priority 2-3: Investigate Soon
[Important but not urgent developments]

## Priority 4-5: Optional Reading
[Interesting but lower priority items]

# Strategic Insights
[Forward-looking analysis and recommendations]

## Key Takeaways
[3-5 critical insights that emerged from the analysis]

## Risk Factors & Limitations
[Potential concerns, technical limitations, ethical issues]

## Recommended Actions
[Specific next steps for different stakeholder types]

**Analysis Guidelines:**

1. **Be Specific**: Cite actual data from news articles (dates, names, metrics)
2. **Be Critical**: Distinguish hype from substance, note limitations
3. **Be Actionable**: Focus on insights that drive decisions
4. **Be Comprehensive**: Cover all key dimensions (technology, industry, use case, impact)
5. **Be Objective**: Present facts and analysis, not opinions
6. **Use Evidence**: Quote relevant statistics, expert opinions, or concrete examples
7. **Note Confidence**: Indicate when claims are speculative vs. well-established
8. **Connect Dots**: Identify patterns and relationships between different developments

**Example Analysis Style:**

Good: "Three articles mentioned GPT-5 capabilities, with TechCrunch reporting 40% improvement on reasoning benchmarks (Jan 15, 2026). This represents a significant advancement in chain-of-thought capabilities, particularly relevant for code generation and scientific research use cases."

Bad: "GPT-5 is really good and will change everything."

**Few-Shot Example:**

Given articles about:
- OpenAI GPT-5 release (High impact, 0.95 relevance)
- New RAG framework from Anthropic (Medium impact, 0.75 relevance)
- Healthcare AI regulation update (Critical impact, 0.90 relevance)

Analysis would include:
- Executive summary highlighting regulatory changes as most critical (affects all healthcare AI)
- GPT-5 as key development (broad applicability, proven vendor)
- RAG framework as notable tool improvement (specific use case)
- Priority ranking: Regulation (1), GPT-5 (1), RAG (2)
- Industry focus on Healthcare (regulatory) and Software Dev (tools)
- Recommendations: compliance review, GPT-5 evaluation, RAG experimentation

**Critical Instructions:**

- Start report immediately without preamble
- Use markdown formatting for readability
- Include specific article headlines when citing evidence
- Provide concrete numbers (relevance scores, impact levels, mention counts)
- Balance breadth (covering all articles) with depth (detailed analysis of key items)
- End with clear, actionable recommendations

Now generate the AI news analysis report:"""
        
        human_prompt = """**Analysis Objective:**
{objective}

**News Article Data:**

{article_data}

**Instructions:**
Generate a comprehensive AI news analysis report following the structure outlined in the system prompt. Focus on the stated objective while covering all key analytical dimensions. Be specific, cite evidence from the articles, and provide actionable insights.

Begin the report now:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _format_articles(self, profiles: List[NewsArticleProfile]) -> str:
        """
        Format news article profiles for LLM consumption.
        
        Creates a comprehensive text representation of all profiles
        that's easy for the LLM to analyze.
        
        Args:
            profiles: List of news article profiles
        
        Returns:
            str: Formatted article data
        """
        formatted_sections = []
        
        for i, profile in enumerate(profiles, 1):
            if not profile.scrape_success:
                # Include failed profiles with error info
                section = [
                    f"## Article {i}: {profile.headline}",
                    f"**Status:** ❌ Failed to scrape",
                    f"**Error:** {profile.error_message}",
                    f"**URL:** {profile.article_url}",
                    ""
                ]
                formatted_sections.append("\n".join(section))
                continue
            
            # Successful profile
            section = [
                f"## Article {i}: {profile.headline}",
                f"**Source:** {profile.news_source}",
                f"**URL:** {profile.article_url}",
            ]
            
            if profile.publication_date:
                section.append(f"**Published:** {profile.publication_date}")
            
            if profile.author:
                section.append(f"**Author:** {profile.author}")
            
            # Metadata scores
            metadata_parts = []
            if profile.potential_impact:
                metadata_parts.append(f"Impact: {profile.potential_impact.value}")
            if profile.relevance_score is not None:
                metadata_parts.append(f"Relevance: {profile.relevance_score:.2f}")
            if profile.recommended_priority is not None:
                metadata_parts.append(f"Priority: {profile.recommended_priority}")
            
            if metadata_parts:
                section.append(f"**Metadata:** {' | '.join(metadata_parts)}")
            
            # Summary
            section.append(f"\n**Summary:**")
            section.append(profile.article_summary)
            
            # Key Technologies
            if profile.key_technologies:
                section.append("\n**Technologies:**")
                section.append(", ".join(profile.key_technologies))
            
            # Use Cases
            if profile.use_cases:
                section.append("\n**Use Cases:**")
                for use_case in profile.use_cases[:5]:  # Top 5
                    section.append(f"- {use_case}")
                if len(profile.use_cases) > 5:
                    section.append(f"- (+{len(profile.use_cases) - 5} more)")
            
            # Affected Industries
            if profile.affected_industries:
                section.append(f"\n**Affected Industries:** {', '.join(profile.affected_industries)}")
            
            # Key Insights
            if profile.key_insights:
                section.append("\n**Key Insights:**")
                for insight in profile.key_insights[:3]:  # Top 3
                    section.append(f"- {insight}")
                if len(profile.key_insights) > 3:
                    section.append(f"- (+{len(profile.key_insights) - 3} more)")
            
            # Limitations
            if profile.limitations_mentioned:
                section.append("\n**Limitations/Concerns:**")
                for limitation in profile.limitations_mentioned[:3]:  # Top 3
                    section.append(f"- {limitation}")
            
            section.append("")  # Blank line between articles
            formatted_sections.append("\n".join(section))
        
        return "\n".join(formatted_sections)
    
    async def generate_report_stream(
        self,
        profiles: List[NewsArticleProfile],
        objective: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate report with streaming output.
        
        Streams the report token-by-token for real-time UI updates.
        Each chunk is a partial string that can be appended to the display.
        
        Args:
            profiles: List of news article profiles to analyze
            objective: Analysis objective/question from user
        
        Yields:
            str: Token chunks as they're generated
        
        Example:
            >>> async for chunk in agent.generate_report_stream(profiles, "priority"):
            ...     report_text += chunk
            ...     update_ui(report_text)
        
        Raises:
            ValueError: If no valid profiles provided
        """
        if not profiles:
            yield "Error: No news article profiles provided for analysis"
            return
        
        # Check if we have at least one valid profile
        valid_profiles = [p for p in profiles if p.scrape_success]
        if not valid_profiles:
            yield "Error: All article scrapes failed. No data available for analysis."
            return
        
        # Format article data
        article_data = self._format_articles(profiles)
        
        # Format messages
        messages = self.prompt.format_messages(
            objective=objective,
            article_data=article_data
        )
        
        # Stream synthesis
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        
        except Exception as e:
            yield f"\n\n---\n**Error during report generation:** {str(e)}"
    
    async def generate_report(
        self,
        profiles: List[NewsArticleProfile],
        objective: str
    ) -> str:
        """
        Generate complete report without streaming.
        
        Useful for batch processing or when streaming is not needed.
        
        Args:
            profiles: List of news article profiles to analyze
            objective: Analysis objective/question from user
        
        Returns:
            str: Complete report text
        
        Example:
            >>> report = await agent.generate_report(profiles, "technology trends")
            >>> print(report)
        """
        chunks = []
        async for chunk in self.generate_report_stream(profiles, objective):
            chunks.append(chunk)
        
        return "".join(chunks)
    
    def get_synthesis_info(self) -> dict:
        """
        Get information about the synthesis configuration.
        
        Returns:
            dict: Synthesis configuration details
        """
        return {
            "model": self.config.GEMINI_MODEL,
            "temperature": self.config.SYNTHESIS_TEMPERATURE,
            "streaming": True,
            "few_shot": True,
            "creative": True,
            "report_type": "AI News Analysis"
        }