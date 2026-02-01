"""
Synthesis Agent with Streaming
Generates comprehensive competitive analysis reports using few-shot prompting
"""

from typing import List, AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import AppConfig
from app.extraction.schemas import CompetitorProfile


class SynthesisAgent:
    """
    Generate competitive analysis reports with streaming output.
    
    Uses few-shot prompting with example reports to guide the LLM
    in producing high-quality, structured competitive intelligence
    analysis. Streams output token-by-token for real-time UI updates.
    
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
        system_prompt = """You are an expert competitive intelligence analyst specializing in market research and strategic analysis.

Your task is to generate comprehensive competitive analysis reports that provide actionable insights for business decision-making.

**Report Structure:**

# Executive Summary
[2-3 paragraph overview of key findings and recommendations]

# Market Positioning Analysis
[Analyze how competitors position themselves in the market]
- Target markets and customer segments
- Value propositions and messaging
- Market gaps and opportunities

# Pricing Strategy Comparison
[Detailed comparison of pricing models]
- Pricing tier structures
- Price points and value positioning
- Monetization strategies
- Pricing gaps and opportunities

# Feature Analysis
[Compare product/service features]
- Common features (table stakes)
- Unique differentiators
- Feature gaps and whitespace
- Technology stack insights

# Competitive Advantages & Weaknesses
[SWOT-style analysis]

**Strengths by Competitor:**
[What each competitor does well]

**Weaknesses & Gaps:**
[Where competitors fall short]

**Opportunities:**
[Market gaps and potential advantages]

**Threats:**
[Competitive pressures and risks]

# Strategic Recommendations
[Actionable insights based on analysis]
1. [Specific recommendation with rationale]
2. [Specific recommendation with rationale]
3. [Specific recommendation with rationale]

**Analysis Guidelines:**
- Be specific - cite actual data from competitor profiles
- Be objective - present facts, not opinions
- Be actionable - focus on insights that drive decisions
- Be comprehensive - cover all key competitive dimensions
- Use markdown formatting for readability
- Include relevant quotes or data points from profiles
- Highlight patterns and trends across competitors
- Note confidence levels when making inferences

**Example Output Style:**

"Company A positions itself in the enterprise segment with pricing starting at $99/month, while Company B targets SMBs with a $29/month entry point. This 3.4x price difference reflects distinct value propositions: Company A emphasizes 'Advanced analytics and dedicated support' while Company B focuses on 'Simple, affordable project management.'"

**Few-Shot Example:**

Given competitors:
- Acme SaaS: Enterprise project management, $149/month Pro tier, AI-powered features
- QuickTask: SMB task management, $29/month, mobile-first approach
- TeamFlow: Mid-market collaboration, $79/month, integrations focus

Analysis would include:
- Price positioning: 5x spread from $29 to $149 indicates segmentation by market size
- Feature differentiation: AI (Acme), Mobile (QuickTask), Integrations (TeamFlow)
- Market gaps: No affordable enterprise option, no AI for SMBs
- Recommendation: Position between $79-$99 with AI features for mid-market

Now analyze the provided competitors following this structure and style."""
        
        human_prompt = """**Analysis Objective:**
{objective}

**Competitor Data:**

{competitor_data}

**Instructions:**
Generate a comprehensive competitive analysis report following the structure outlined in the system prompt. Focus on the stated objective while covering all key competitive dimensions. Be specific, cite data, and provide actionable recommendations.

Begin the report now:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _format_profiles(self, profiles: List[CompetitorProfile]) -> str:
        """
        Format competitor profiles for LLM consumption.
        
        Creates a comprehensive text representation of all profiles
        that's easy for the LLM to analyze.
        
        Args:
            profiles: List of competitor profiles
        
        Returns:
            str: Formatted competitor data
        """
        formatted_sections = []
        
        for i, profile in enumerate(profiles, 1):
            if not profile.scrape_success:
                # Include failed profiles with error info
                section = [
                    f"## Competitor {i}: {profile.company_name}",
                    f"**Status:** ❌ Failed to scrape",
                    f"**Error:** {profile.error_message}",
                    f"**URL:** {profile.website_url}",
                    ""
                ]
                formatted_sections.append("\n".join(section))
                continue
            
            # Successful profile
            section = [
                f"## Competitor {i}: {profile.company_name}",
                f"**Website:** {profile.website_url}",
            ]
            
            if profile.tagline:
                section.append(f"**Tagline:** {profile.tagline}")
            
            if profile.target_market:
                section.append(f"**Target Market:** {profile.target_market}")
            
            # Key Features
            if profile.key_features:
                section.append("\n**Key Features:**")
                for feature in profile.key_features:
                    section.append(f"- {feature}")
            
            # Pricing Tiers
            if profile.pricing_tiers:
                section.append("\n**Pricing:**")
                for tier in profile.pricing_tiers:
                    tier_text = f"- **{tier.name}**: {tier.price or 'N/A'}"
                    if tier.features:
                        tier_text += f"\n  Features: {', '.join(tier.features[:5])}"
                        if len(tier.features) > 5:
                            tier_text += f" (+{len(tier.features) - 5} more)"
                    section.append(tier_text)
            
            # USPs
            if profile.unique_selling_points:
                section.append("\n**Unique Selling Points:**")
                for usp in profile.unique_selling_points:
                    section.append(f"- {usp}")
            
            # Technology Stack
            if profile.technology_stack:
                section.append(f"\n**Technology Stack:** {', '.join(profile.technology_stack)}")
            
            section.append("")  # Blank line between competitors
            formatted_sections.append("\n".join(section))
        
        return "\n".join(formatted_sections)
    
    async def generate_report_stream(
        self,
        profiles: List[CompetitorProfile],
        objective: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate report with streaming output.
        
        Streams the report token-by-token for real-time UI updates.
        Each chunk is a partial string that can be appended to the display.
        
        Args:
            profiles: List of competitor profiles to analyze
            objective: Analysis objective/question from user
        
        Yields:
            str: Token chunks as they're generated
        
        Example:
            >>> async for chunk in agent.generate_report_stream(profiles, "pricing"):
            ...     report_text += chunk
            ...     update_ui(report_text)
        
        Raises:
            ValueError: If no valid profiles provided
        """
        if not profiles:
            yield "Error: No competitor profiles provided for analysis"
            return
        
        # Check if we have at least one valid profile
        valid_profiles = [p for p in profiles if p.scrape_success]
        if not valid_profiles:
            yield "Error: All competitor scrapes failed. No data available for analysis."
            return
        
        # Format competitor data
        competitor_data = self._format_profiles(profiles)
        
        # Format messages
        messages = self.prompt.format_messages(
            objective=objective,
            competitor_data=competitor_data
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
        profiles: List[CompetitorProfile],
        objective: str
    ) -> str:
        """
        Generate complete report without streaming.
        
        Useful for batch processing or when streaming is not needed.
        
        Args:
            profiles: List of competitor profiles to analyze
            objective: Analysis objective/question from user
        
        Returns:
            str: Complete report text
        
        Example:
            >>> report = await agent.generate_report(profiles, "pricing")
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
            "creative": True
        }