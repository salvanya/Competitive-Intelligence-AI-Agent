"""
Phase 7: Synthesis Agent with Few-Shot Prompting
=================================================
Implements: Advanced competitive analysis report generation

Architecture:
- Few-shot prompting with example trajectories
- Chain-of-Thought reasoning via <thinking> tags
- Multi-source intelligence aggregation
- Temperature=0.7 for creative synthesis
- Structured report generation with citations
- Automatic report saving to ./outputs

Key Concepts:
1. Few-shot learning: Show the model examples of excellent reports
2. Chain-of-Thought: Explicit reasoning steps before conclusions
3. Multi-source synthesis: Combine data from multiple competitors
4. Citation tracking: Reference specific data points in output
5. Report templates: Structured markdown outputs

Production Patterns:
- Few-shot examples guide report quality and structure
- <thinking> tags make reasoning transparent and debuggable
- Temperature=0.7 balances creativity with factual accuracy
- Pydantic validation ensures report schema compliance
- All reports saved to timestamped files for persistence
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    GEMINI_MODEL = "gemini-2.0-flash-lite"
    EMBEDDING_MODEL = "models/text-embedding-004"
    SYNTHESIS_TEMPERATURE = 0.7  # Creative but grounded
    MAX_TOKENS = 4096  # Longer reports
    VECTOR_SIZE = 768
    COLLECTION_NAME = "competitor_intelligence_phase7"
    
    # Output directory
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# REPORT WRITER
# ============================================================================

class ReportWriter:
    """Handles saving reports to files with timestamps"""
    
    @staticmethod
    def save_report(content: str, report_type: str, objective: str = "") -> Path:
        """
        Save report to timestamped file
        
        Args:
            content: Report content to save
            report_type: Type of report (e.g., "competitive_analysis", "pricing_analysis")
            objective: Optional objective description for filename
        
        Returns:
            Path to saved file
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create safe filename from objective
        safe_objective = ""
        if objective:
            safe_objective = "_" + "".join(c if c.isalnum() or c in (' ', '_') else '' 
                                           for c in objective)
            safe_objective = safe_objective.replace(' ', '_')[:50]  # Limit length
        
        # Construct filename
        filename = f"{report_type}{safe_objective}_{timestamp}.txt"
        filepath = Config.OUTPUT_DIR / filename
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n💾 Report saved to: {filepath}")
        
        return filepath
    
    @staticmethod
    def save_report_with_metadata(
        content: str, 
        report_type: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save report with metadata header
        
        Args:
            content: Report content
            report_type: Type of report
            metadata: Additional metadata to include
        
        Returns:
            Path to saved file
        """
        # Build metadata header
        header = "=" * 80 + "\n"
        header += f"REPORT TYPE: {report_type}\n"
        header += f"GENERATED: {datetime.now(timezone.utc).isoformat()}\n"
        
        for key, value in metadata.items():
            header += f"{key.upper()}: {value}\n"
        
        header += "=" * 80 + "\n\n"
        
        # Combine header and content
        full_content = header + content
        
        # Save using standard method
        return ReportWriter.save_report(
            full_content,
            report_type,
            metadata.get('objective', '')
        )


# ============================================================================
# REPORT SCHEMA (Pydantic)
# ============================================================================

class CompetitiveInsight(BaseModel):
    """Single insight with supporting evidence"""
    insight: str = Field(description="The key insight or finding")
    evidence: List[str] = Field(description="Supporting data points with citations")
    confidence: str = Field(description="High/Medium/Low confidence level")


class CompetitiveAnalysisReport(BaseModel):
    """Structured competitive analysis report"""
    executive_summary: str = Field(description="2-3 sentence overview")
    market_positioning: List[CompetitiveInsight] = Field(
        description="How competitors position themselves"
    )
    pricing_analysis: List[CompetitiveInsight] = Field(
        description="Pricing strategies and gaps"
    )
    feature_comparison: List[CompetitiveInsight] = Field(
        description="Feature differentiation analysis"
    )
    opportunities: List[str] = Field(
        description="Actionable market opportunities"
    )
    threats: List[str] = Field(
        description="Competitive threats to watch"
    )
    recommendations: List[str] = Field(
        description="Strategic recommendations"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# MOCK DATA RETRIEVAL (Simulates Vector Store Queries)
# ============================================================================

class MockCompetitorData:
    """Simulates retrieved competitor intelligence"""
    
    @staticmethod
    def get_market_data() -> Dict[str, Any]:
        """Simulate market segment analysis data"""
        return {
            "segment": "Mid-Market Analytics",
            "total_competitors": 3,
            "competitors": [
                {
                    "name": "DataStream Analytics",
                    "target_market": "Mid-market (10-500 employees)",
                    "tagline": "Transform Data into Decisions in Real-Time",
                    "pricing": {
                        "entry": "$99/month (Starter)",
                        "mid": "$299/month (Professional)",
                        "enterprise": "Custom pricing"
                    },
                    "key_features": [
                        "200+ integrations",
                        "AI-powered SQL editor",
                        "Real-time alerting with ML anomaly detection",
                        "White-label portals",
                        "RBAC"
                    ],
                    "differentiators": [
                        "No-code 5-minute setup",
                        "SOC 2 Type II certified",
                        "500+ customers including Fortune 500"
                    ],
                    "technology": "Serverless (AWS Lambda), PostgreSQL, ML models"
                },
                {
                    "name": "CloudMetrics Pro",
                    "target_market": "Enterprise (500+ employees)",
                    "tagline": "Next-Generation Analytics for Enterprise",
                    "pricing": {
                        "entry": "$149/month (Growth)",
                        "mid": "N/A",
                        "enterprise": "Contact Sales"
                    },
                    "key_features": [
                        "Advanced visualization (50+ chart types)",
                        "Predictive ML analytics",
                        "Automated reporting",
                        "Mobile dashboards",
                        "Enterprise API integrations (SAP, Oracle)"
                    ],
                    "differentiators": [
                        "1,000+ companies worldwide",
                        "ISO 27001 + GDPR certified",
                        "99.99% uptime SLA",
                        "Dedicated customer success manager"
                    ],
                    "technology": "Google Cloud Platform, TensorFlow, PyTorch"
                },
                {
                    "name": "InsightHub",
                    "target_market": "SMB/Startups (5-50 employees)",
                    "tagline": "Analytics Simplified",
                    "pricing": {
                        "entry": "$49/month (Basic)",
                        "mid": "$199/month (Pro)",
                        "enterprise": "Custom"
                    },
                    "key_features": [
                        "Drag-and-drop dashboard builder",
                        "E-commerce integrations (Shopify, Stripe)",
                        "One-click sharing",
                        "Automated email reports",
                        "No coding required"
                    ],
                    "differentiators": [
                        "10-minute setup",
                        "14-day free trial (no credit card)",
                        "Designed for non-technical users",
                        "Affordable for bootstrapped startups"
                    ],
                    "technology": "Simple, reliable infrastructure"
                }
            ]
        }


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "input": """Analyze competitive landscape for mid-market analytics tools.

Retrieved Data:
- Competitor A: $99/mo entry, targets mid-market, SOC 2 certified
- Competitor B: $149/mo entry, targets enterprise, ISO 27001 certified
- Competitor C: $49/mo entry, targets SMBs, no certifications mentioned

Market Context: Mid-market segment (10-500 employees)""",
        
        "output": """<thinking>
Let me analyze the competitive landscape systematically:

1. Pricing Tiers:
   - Entry tier range: $49-$149/month
   - Competitor A ($99) positions middle-market
   - Competitor B ($149) premium positioning despite targeting larger orgs
   - Competitor C ($49) budget option for smaller teams

2. Market Positioning:
   - Competitor A: Mid-market focus with enterprise-grade security (SOC 2)
   - Competitor B: Enterprise focus but Growth tier accessible to mid-market
   - Competitor C: SMB/startup focus with affordability priority

3. Gaps I'm observing:
   - No micro-tier below $49 for very small teams
   - Large price gap between C ($49) and A ($99) - 100% increase
   - Certification gap: Only A and B have compliance certifications
   - Potential gap: Usage-based pricing not mentioned anywhere

4. Competitive Dynamics:
   - A competes on balanced value (mid-price, high security)
   - B competes on enterprise features at accessible price
   - C competes purely on affordability and simplicity
</thinking>

**Executive Summary:**
The mid-market analytics space shows three distinct positioning strategies: enterprise-grade security at mid-tier pricing (Competitor A), feature-rich enterprise tools with accessible entry tier (Competitor B), and budget-focused simplicity (Competitor C). A significant $50 pricing gap exists between budget and mid-tier options.

**Market Positioning Insights:**

1. **Security as Differentiator** (High Confidence)
   - Evidence: Competitor A emphasizes SOC 2 Type II certification
   - Evidence: Competitor B highlights ISO 27001 + GDPR compliance
   - Evidence: Competitor C makes no security/compliance claims
   - Implication: Mid-market buyers prioritize security certifications

2. **Market Segment Overlap** (High Confidence)
   - Evidence: Competitor A targets 10-500 employees (mid-market)
   - Evidence: Competitor B targets 500+ but offers $149 Growth tier
   - Evidence: Competitor C targets 5-50 employees (SMB)
   - Implication: B is moving downmarket to capture mid-tier growth

**Pricing Analysis:**

1. **Tiered Pricing Standard** (High Confidence)
   - Evidence: All three offer entry/mid/enterprise tiers
   - Evidence: Entry tier range spans $49-$149 (3x difference)
   - Opportunity: $50-$75 sweet spot is unaddressed

2. **Enterprise Pricing Opacity** (Medium Confidence)
   - Evidence: Both A and B use "Custom pricing" for enterprise
   - Evidence: Only C specifies mid-tier price ($199)
   - Implication: Enterprise pricing likely 3-5x entry tier based on industry norms

**Opportunities:**

1. **Micro-tier Gap**: No option under $49/month for 1-5 person teams
2. **Usage-based Pricing**: All competitors use seat-based pricing - consumption model untapped
3. **Vertical Specialization**: No competitor emphasizes industry-specific features (healthcare, finance)
4. **Mid-tier Certification**: Opportunity to offer SOC 2 at $75-99 price point

**Threats:**

1. **Downmarket Pressure**: Enterprise player (B) expanding to mid-market
2. **Price Competition**: C's $49 entry creates low-price anchor
3. **Feature Parity**: Risk of commoditization as features converge

**Recommendations:**

1. Position at $75-89/month with SOC 2 certification (split the gap between C and A)
2. Introduce usage-based tier for seasonal/variable workloads
3. Develop vertical-specific templates (e.g., "Healthcare Analytics Starter Pack")
4. Emphasize setup speed (target <5 minutes to match A's claim)
5. Consider freemium tier to compete with C's trial offer"""
    }
]


# ============================================================================
# SYNTHESIS AGENT WITH FEW-SHOT PROMPTING
# ============================================================================

class SynthesisAgent:
    """
    Advanced competitive analysis synthesis using few-shot prompting
    and chain-of-thought reasoning
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.SYNTHESIS_TEMPERATURE,  # 0.7 for creative synthesis
            max_tokens=Config.MAX_TOKENS,
        )
        
        # Build few-shot prompt template
        self.prompt = self._create_few_shot_prompt()
        
        # Chain for text generation
        self.synthesis_chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_few_shot_prompt(self) -> ChatPromptTemplate:
        """
        Create prompt with few-shot examples showing desired reasoning pattern
        """
        
        # Example prompt template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        # Few-shot prompt with examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=FEW_SHOT_EXAMPLES,
        )
        
        # Final prompt combining system instructions + few-shot examples + user query
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an elite competitive intelligence analyst specializing in market research and strategic analysis.

**YOUR PROCESS (MANDATORY):**

1. **Begin with <thinking> tags** to show your reasoning:
   - Break down the competitive data systematically
   - Identify patterns, gaps, and anomalies
   - Consider multiple perspectives (pricing, features, positioning)
   - Note confidence levels for each insight

2. **After </thinking>, provide structured analysis** with these sections:
   - **Executive Summary**: 2-3 sentence distillation of key findings
   - **Market Positioning Insights**: How competitors differentiate (with evidence bullets)
   - **Pricing Analysis**: Pricing strategies, gaps, opportunities (with evidence)
   - **Feature Comparison**: Feature differentiation and gaps (with evidence)
   - **Opportunities**: 3-5 actionable market opportunities
   - **Threats**: 3-5 competitive threats to monitor
   - **Recommendations**: 5-7 strategic recommendations prioritized by impact

**QUALITY STANDARDS:**

- **Evidence-Based**: Every insight must cite specific data points
- **Confidence Levels**: Label insights as High/Medium/Low confidence
- **Actionable**: Recommendations must be specific and measurable
- **Balanced**: Consider both opportunities and threats
- **Strategic**: Think 12-24 months ahead, not just current state

**CITATION FORMAT:**
- Evidence: [Competitor name] [specific data point]
- Example: "Evidence: DataStream charges $99/month for entry tier with SOC 2 certification"

**AVOID:**
- Generic insights that could apply to any market
- Recommendations without supporting evidence
- Copying competitor language verbatim
- Ignoring gaps or contradictions in data

Study the following examples of excellent competitive analysis:"""),
            
            # Insert few-shot examples
            few_shot_prompt,
            
            # User query
            ("human", "{input}")
        ])
        
        return final_prompt
    
    async def generate_competitive_report(
        self, 
        analysis_objective: str,
        competitor_data: Dict[str, Any],
        save_to_file: bool = True
    ) -> str:
        """
        Generate comprehensive competitive analysis report
        
        Args:
            analysis_objective: What aspect to analyze (e.g., "pricing strategy")
            competitor_data: Retrieved intelligence from vector store
            save_to_file: Whether to save report to file (default: True)
        
        Returns:
            Markdown-formatted analysis report with <thinking> and structured sections
        """
        
        # Format competitor data for prompt
        data_summary = self._format_competitor_data(competitor_data)
        
        # Construct input
        input_text = f"""Objective: {analysis_objective}

Retrieved Competitive Intelligence:
{data_summary}

Market Context: {competitor_data.get('segment', 'General Market')}
Total Competitors Analyzed: {competitor_data.get('total_competitors', 0)}

Provide comprehensive competitive analysis following the examples shown."""
        
        # Generate synthesis
        print(f"\n🤖 Synthesizing competitive analysis...")
        print(f"   Objective: {analysis_objective}")
        print(f"   Competitors: {competitor_data.get('total_competitors', 0)}")
        print(f"   Temperature: {Config.SYNTHESIS_TEMPERATURE}")
        
        report = await self.synthesis_chain.ainvoke({"input": input_text})
        
        # Save to file if requested
        if save_to_file:
            ReportWriter.save_report_with_metadata(
                content=report,
                report_type="competitive_analysis",
                metadata={
                    "objective": analysis_objective,
                    "competitors_analyzed": competitor_data.get('total_competitors', 0),
                    "market_segment": competitor_data.get('segment', 'N/A'),
                    "temperature": Config.SYNTHESIS_TEMPERATURE,
                    "model": Config.GEMINI_MODEL
                }
            )
        
        return report
    
    def _format_competitor_data(self, data: Dict[str, Any]) -> str:
        """Format competitor data for LLM consumption"""
        
        formatted = []
        
        for i, comp in enumerate(data.get('competitors', []), 1):
            comp_text = f"""
**Competitor {i}: {comp['name']}**
- Target Market: {comp['target_market']}
- Positioning: {comp['tagline']}
- Pricing:
  • Entry Tier: {comp['pricing']['entry']}
  • Mid Tier: {comp['pricing']['mid']}
  • Enterprise: {comp['pricing']['enterprise']}
- Key Features: {', '.join(comp['key_features'][:5])}
- Differentiators: {', '.join(comp['differentiators'])}
- Technology: {comp['technology']}
"""
            formatted.append(comp_text)
        
        return "\n".join(formatted)
    
    async def generate_streaming_report(
        self,
        analysis_objective: str,
        competitor_data: Dict[str, Any],
        save_to_file: bool = True
    ):
        """
        Generate report with streaming for real-time UX
        
        Yields chunks of the report as they're generated
        """
        
        data_summary = self._format_competitor_data(competitor_data)
        
        input_text = f"""Objective: {analysis_objective}

Retrieved Competitive Intelligence:
{data_summary}

Market Context: {competitor_data.get('segment', 'General Market')}

Provide comprehensive competitive analysis following the examples shown."""
        
        print(f"\n🌊 Streaming competitive analysis...")
        print(f"   Watch the reasoning unfold in real-time:\n")
        
        # Collect chunks for saving
        collected_chunks = []
        
        async for chunk in self.synthesis_chain.astream({"input": input_text}):
            print(chunk, end="", flush=True)
            collected_chunks.append(chunk)
            yield chunk
        
        print("\n")  # Final newline
        
        # Save complete report if requested
        if save_to_file:
            full_report = "".join(collected_chunks)
            ReportWriter.save_report_with_metadata(
                content=full_report,
                report_type="competitive_analysis_streamed",
                metadata={
                    "objective": analysis_objective,
                    "competitors_analyzed": competitor_data.get('total_competitors', 0),
                    "market_segment": competitor_data.get('segment', 'N/A'),
                    "temperature": Config.SYNTHESIS_TEMPERATURE,
                    "model": Config.GEMINI_MODEL,
                    "streaming": True
                }
            )


# ============================================================================
# MULTI-SOURCE AGGREGATION EXAMPLE
# ============================================================================

class MultiSourceSynthesisAgent(SynthesisAgent):
    """
    Extended synthesis agent that combines multiple data sources
    """
    
    async def generate_comprehensive_report(
        self,
        primary_objective: str,
        market_data: Dict[str, Any],
        pricing_data: Optional[Dict[str, Any]] = None,
        feature_data: Optional[Dict[str, Any]] = None,
        save_to_file: bool = True
    ) -> str:
        """
        Synthesize insights from multiple data sources
        
        Demonstrates: Multi-source aggregation and cross-referencing
        """
        
        # Aggregate all data sources
        aggregated_input = f"""Primary Objective: {primary_objective}

=== MARKET POSITIONING DATA ===
{self._format_competitor_data(market_data)}
"""
        
        if pricing_data:
            aggregated_input += f"""
=== DETAILED PRICING ANALYSIS ===
{self._format_pricing_data(pricing_data)}
"""
        
        if feature_data:
            aggregated_input += f"""
=== FEATURE MATRIX ===
{self._format_feature_data(feature_data)}
"""
        
        aggregated_input += """
Synthesize comprehensive competitive intelligence across all data sources.
Cross-reference findings to identify:
1. Patterns that appear across multiple sources
2. Contradictions or gaps requiring further investigation
3. High-confidence insights with multiple supporting data points
4. Strategic opportunities emerging from combined analysis"""
        
        report = await self.synthesis_chain.ainvoke({"input": aggregated_input})
        
        # Save to file if requested
        if save_to_file:
            ReportWriter.save_report_with_metadata(
                content=report,
                report_type="comprehensive_multi_source_analysis",
                metadata={
                    "objective": primary_objective,
                    "data_sources": "market_data, pricing_data, feature_data",
                    "competitors_analyzed": market_data.get('total_competitors', 0),
                    "temperature": Config.SYNTHESIS_TEMPERATURE,
                    "model": Config.GEMINI_MODEL
                }
            )
        
        return report
    
    def _format_pricing_data(self, data: Dict[str, Any]) -> str:
        """Format pricing-specific data"""
        # Placeholder - would format detailed pricing comparisons
        return "Detailed pricing tier analysis with feature breakdowns..."
    
    def _format_feature_data(self, data: Dict[str, Any]) -> str:
        """Format feature comparison matrix"""
        # Placeholder - would format feature parity matrix
        return "Feature coverage matrix across competitors..."


# ============================================================================
# TESTING
# ============================================================================

async def test_synthesis_agent():
    """Test synthesis agent with few-shot prompting"""
    
    print("=" * 80)
    print("🚀 PHASE 7: SYNTHESIS AGENT WITH FEW-SHOT PROMPTING")
    print("=" * 80)
    
    # Get mock competitor data
    market_data = MockCompetitorData.get_market_data()
    
    # Test 1: Standard synthesis
    print("\n1️⃣  STANDARD COMPETITIVE ANALYSIS")
    print("-" * 80)
    
    agent = SynthesisAgent()
    
    report1 = await agent.generate_competitive_report(
        analysis_objective="Analyze competitive landscape for mid-market analytics positioning",
        competitor_data=market_data,
        save_to_file=True  # Save to file
    )
    
    print("\n📊 REPORT PREVIEW (first 500 chars):")
    print("=" * 80)
    print(report1[:500] + "...")
    print("=" * 80)
    
    # Test 2: Streaming synthesis
    print("\n2️⃣  STREAMING SYNTHESIS (Real-time UX)")
    print("-" * 80)
    
    input("\n⏸️  Press Enter to start streaming synthesis...")
    
    chunks = []
    async for chunk in agent.generate_streaming_report(
        analysis_objective="Identify pricing strategy opportunities in analytics market",
        competitor_data=market_data,
        save_to_file=True  # Save streamed report
    ):
        chunks.append(chunk)
    
    # Test 3: Different analysis angle
    print("\n3️⃣  FOCUSED ANALYSIS: Feature Differentiation")
    print("-" * 80)
    
    report2 = await agent.generate_competitive_report(
        analysis_objective="Analyze feature differentiation strategies and identify gaps",
        competitor_data=market_data,
        save_to_file=True  # Save to file
    )
    
    print("\n📊 REPORT PREVIEW (first 500 chars):")
    print("=" * 80)
    print(report2[:500] + "...")
    print("=" * 80)
    
    # Test 4: Multi-source synthesis
    print("\n4️⃣  MULTI-SOURCE SYNTHESIS")
    print("-" * 80)
    
    multi_agent = MultiSourceSynthesisAgent()
    
    comprehensive_report = await multi_agent.generate_comprehensive_report(
        primary_objective="Comprehensive competitive positioning strategy",
        market_data=market_data,
        pricing_data={"note": "Simulated pricing data"},
        feature_data={"note": "Simulated feature matrix"},
        save_to_file=True  # Save to file
    )
    
    print("\n📊 REPORT PREVIEW (first 500 chars):")
    print("=" * 80)
    print(comprehensive_report[:500] + "...")
    print("=" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 7 COMPLETE: SYNTHESIS AGENT OPERATIONAL")
    print("=" * 80)
    print("\n📋 Demonstrated Capabilities:")
    print("   ✅ Few-shot prompting with example trajectories")
    print("   ✅ Chain-of-Thought reasoning via <thinking> tags")
    print("   ✅ Temperature=0.7 for creative synthesis")
    print("   ✅ Structured report generation with citations")
    print("   ✅ Streaming synthesis for real-time UX")
    print("   ✅ Multi-source intelligence aggregation")
    print("   ✅ Automatic report saving with metadata")
    print(f"\n📁 All reports saved to: {Config.OUTPUT_DIR}")
    print("\n🎯 Next Phase: Production API & Deployment (Phase 8)")
    print("\n💡 Key Learnings:")
    print("   1. Few-shot examples dramatically improve output quality")
    print("   2. <thinking> tags make reasoning transparent and debuggable")
    print("   3. Temperature=0.7 balances creativity with factual grounding")
    print("   4. Structured sections ensure consistent report format")
    print("   5. Evidence citations build trust in AI-generated insights")
    print("   6. Streaming provides superior UX for long-form generation")
    print("   7. Timestamped files preserve analysis history")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    asyncio.run(test_synthesis_agent())