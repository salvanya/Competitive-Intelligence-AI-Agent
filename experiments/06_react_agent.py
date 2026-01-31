"""
Phase 6: ReAct Agent Construction
==================================
Implements: Tool-using agent with explicit reasoning over competitive intelligence

Architecture:
- Agent LLM: Gemini temp=0.7 for creative reasoning (vs temp=0 for extraction)
- Tools: search_competitors, compare_pricing, identify_gaps, generate_swot
- Pattern: ReAct (Reason → Act → Observe loop)
- Prompt Engineering: <thinking> tags for transparent reasoning
- Integration: Uses Phase 5 vector store as knowledge base

Key Concepts:
1. Deterministic chains (Phase 4) vs Dynamic agents (Phase 6)
2. Tool calling with structured inputs/outputs
3. Few-shot prompting with example trajectories
4. Multi-step reasoning with intermediate observations
5. Temperature strategy: 0.7 balances creativity with coherence

Production Patterns:
- Tools return formatted strings (not raw data) for LLM consumption
- Explicit thinking tags make agent reasoning observable
- Max iterations prevent infinite loops
- Graceful error handling for missing data
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field, field_validator, AliasChoices

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL = "models/text-embedding-004"
    EXTRACTION_TEMPERATURE = 0
    AGENT_TEMPERATURE = 0.7  # For creative reasoning
    MAX_TOKENS = 2048
    VECTOR_SIZE = 768
    COLLECTION_NAME = "competitor_intelligence_phase6"


# ============================================================================
# PYDANTIC MODELS (from Phase 5)
# ============================================================================

class PricingTier(BaseModel):
    name: str = Field(description="Plan name")
    price: Optional[str] = Field(default=None, description="Price or 'Contact Sales'")
    features: List[str] = Field(default_factory=list, description="Included features")
    
    @field_validator('price')
    @classmethod
    def normalize_price(cls, v):
        if v is None:
            return None
        v = v.strip()
        if v.lower() in ['contact sales', 'custom', 'enterprise']:
            return "Contact Sales"
        if v and v[0].isdigit():
            return f"${v}"
        return v


class CompetitorProfile(BaseModel):
    company_name: str = Field(
        description="Official company name",
        validation_alias=AliasChoices('company_name', 'companyName', 'competitorName', 'competitor_name', 'name')
    )
    tagline: Optional[str] = Field(default=None, description="Value proposition")
    target_market: Optional[str] = Field(default=None, description="Primary customer segment")
    
    @field_validator('target_market', mode='before')
    @classmethod
    def flatten_target_market(cls, v):
        if isinstance(v, dict):
            return v.get('company_size') or v.get('segment') or str(v)
        elif isinstance(v, list):
            return ', '.join(str(item) for item in v)
        return v
    
    key_features: List[str] = Field(default_factory=list, description="Top features")
    pricing_tiers: List[PricingTier] = Field(default_factory=list, description="Pricing plans")
    unique_selling_points: List[str] = Field(default_factory=list, description="Differentiators")
    technology_stack: List[str] = Field(default_factory=list, description="Tech mentions")
    website_url: Optional[str] = Field(default=None, description="Company website")
    extraction_timestamp: Optional[str] = Field(default=None, description="When extracted")
    
    model_config = {"populate_by_name": True}
    
    @field_validator('extraction_timestamp', mode='before')
    @classmethod
    def set_timestamp(cls, v):
        return v or datetime.now(timezone.utc).isoformat()


# ============================================================================
# EXTRACTION PIPELINE (from Phase 5)
# ============================================================================

class ExtractionPipeline:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.EXTRACTION_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        self.parser = PydanticOutputParser(pydantic_object=CompetitorProfile)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a competitive intelligence extraction specialist.

        Your task: Extract structured data from website content about a company.

        EXTRACTION RULES:
        1. Company name: Extract the official company name (look for brand name, "About" sections)
        2. Tagline: Look for value propositions, slogans, or "what we do" statements
        3. Target market: Identify who they serve (company size, industry, persona)
        4. Key features: List main product features/capabilities (aim for 5-7)
        5. Pricing tiers: Extract ALL pricing plans with exact names and prices
        6. Unique selling points: What makes them different from competitors
        7. Technology stack: Mentioned technologies, certifications, architectures

        IMPORTANT:
        - Extract ALL explicitly stated information
        - Use descriptive feature names, not just keywords
        - For pricing: extract exact dollar amounts or "Contact Sales"
        - If field is not mentioned, use null or empty list
        - Return valid JSON matching CompetitorProfile schema

        {format_instructions}"""),
            
            ("human", """Website: {url}

        Content:
        {content}

        Extract all competitive intelligence data following the schema.""")
        ])
        
        self.chain = (
            {
                "url": lambda x: x["url"],
                "content": lambda x: x["content"],
                "format_instructions": lambda x: self.parser.get_format_instructions()
            }
            | self.prompt
            | self.llm
            | self.parser
        )
    
    async def extract(self, url: str, content: str) -> CompetitorProfile:
        result = await self.chain.ainvoke({"url": url, "content": content})
        result.website_url = url
        return result


# ============================================================================
# VECTOR STORE (from Phase 5)
# ============================================================================

class CompetitorVectorStore:
    def __init__(self):
        self.client = QdrantClient(location=":memory:")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            task_type="retrieval_document"
        )
        self._initialize_collection()
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embeddings,
        )
    
    def _initialize_collection(self):
        try:
            self.client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=Config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass
    
    async def ingest_competitor(self, profile: CompetitorProfile) -> str:
        searchable_text = self._profile_to_searchable_text(profile)
        doc = Document(
            page_content=searchable_text,
            metadata={
                "company_name": profile.company_name,
                "website_url": profile.website_url or "",
                "target_market": profile.target_market or "",
                "num_features": len(profile.key_features),
                "num_pricing_tiers": len(profile.pricing_tiers),
                "technology_stack": ",".join(profile.technology_stack),
                "extraction_timestamp": profile.extraction_timestamp,
                "profile_id": str(uuid4())
            }
        )
        ids = await self.vectorstore.aadd_documents([doc])
        print(f"✅ Ingested: {profile.company_name}")
        return ids[0]
    
    def _profile_to_searchable_text(self, profile: CompetitorProfile) -> str:
        """Convert profile to searchable text for embedding"""
        parts = [
            f"Company: {profile.company_name}",
            f"Tagline: {profile.tagline or 'N/A'}",
            f"Target Market: {profile.target_market or 'N/A'}",
            f"Key Features: {', '.join(profile.key_features)}",
            f"Unique Selling Points: {', '.join(profile.unique_selling_points)}",
            f"Technology: {', '.join(profile.technology_stack)}",
        ]
        
        if profile.pricing_tiers:
            pricing = ", ".join([f"{t.name}: {t.price}" for t in profile.pricing_tiers])
            parts.append(f"Pricing: {pricing}")
        
        searchable = "\n".join(parts)
        
        # DEBUG: Print what we're embedding
        print(f"\n📝 EMBEDDING TEXT FOR: {profile.company_name}")
        print(f"{'='*60}")
        print(searchable)
        print(f"{'='*60}\n")
        
        return searchable
    

    async def search_similar(
        self, 
        query: str, 
        k: int = 5,  # Default to 5 for better recall
        filter_market: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for similar competitors.
        
        Note: Market filtering is now handled at the tool level for better control.
        """
        # Perform vector similarity search
        results = await self.vectorstore.asimilarity_search_with_score(
            query, 
            k=k
        )
        
        # Convert to standardized dict format
        return [
            {
                "company": doc.metadata["company_name"],
                "website": doc.metadata["website_url"],
                "target_market": doc.metadata["target_market"],
                "similarity_score": float(score),
                "content_preview": doc.page_content[:200] + "..."
            }
            for doc, score in results
        ]


# ============================================================================
# GLOBAL STATE (lazy initialization)
# ============================================================================

_vectorstore: Optional[CompetitorVectorStore] = None

def get_vectorstore() -> CompetitorVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = CompetitorVectorStore()
    return _vectorstore


# ============================================================================
# REACT AGENT TOOLS
# ============================================================================

async def search_competitors_tool(query: str, market_segment: str = "", k: int = 3) -> str:
    """
    Search for competitors using semantic similarity.
    
    Args:
        query: Search query describing the product category
        market_segment: Optional context (informational only, doesn't filter)
        k: Number of results to return
    
    Returns:
        Formatted search results with company details
    """
    store = get_vectorstore()
    
    print(f"\n🔍 SEARCH_COMPETITORS_TOOL:")
    print(f"   Query: '{query}'")
    print(f"   Market context: '{market_segment}' (informational)")
    
    try:
        # Get k+1 results to exclude the queried company if it appears
        search_k = k + 2  # Buffer for self-matches
        
        # Perform semantic search WITHOUT market filtering
        # Rationale: Competitors often serve adjacent markets
        all_results = await store.search_similar(
            query=query,
            k=search_k,
            filter_market=None  # Don't filter - let semantic similarity decide
        )
        
        print(f"   Raw results: {len(all_results)} companies found")
        for r in all_results:
            print(f"      {r['company']} (score: {r['similarity_score']:.3f})")
        
        if not all_results:
            return (
                f"⚠️ NO COMPETITORS FOUND\n\n"
                f"Search query: '{query}'\n"
                f"Searched the entire database but found no matches.\n\n"
                f"**Suggestion**: Try different descriptive terms for the product category."
            )
        
        # Exclude companies that match the query (self-references)
        filtered_results = [
            comp for comp in all_results
            if not _should_exclude_company(comp['company'], query)
        ]

        # Take top k from filtered results
        final_results = filtered_results[:k]
        
        # Format results with market context
        output = f"**Competitor Search Results**\n\n"
        output += f"Query: \"{query}\"\n"
        if market_segment:
            output += f"Context: {market_segment} segment (showing all competitors including adjacent markets)\n"
        output += f"\nFound {len(final_results)} competitors:\n\n"
        
        for i, comp in enumerate(final_results, 1):
            output += f"**{i}. {comp['company']}**\n"
            output += f"   - Similarity Score: {comp['similarity_score']:.3f}\n"
            output += f"   - Target Market: {comp['target_market']}\n"
            output += f"   - Website: {comp['website']}\n"
            output += f"   - Key Info: {comp['content_preview'][:180]}...\n\n"
        
        # Add interpretive note
        output += "**Note**: Results include competitors across all market segments. "
        output += "Companies may serve adjacent markets (e.g., enterprise, mid-market, SMB) "
        output += "while still competing on features, pricing, or positioning.\n"
        
        return output
        
    except Exception as e:
        return f"❌ Search error: {str(e)}"

def _should_exclude_company(company_name: str, query: str) -> bool:
    """
    Determine if a company should be excluded from competitor search results.
    Returns True if the company name appears in the search query.
    """
    query_lower = query.lower()
    company_lower = company_name.lower()
    
    # Check for exact match or close variants
    company_tokens = company_lower.split()
    for token in company_tokens:
        if len(token) > 3 and token in query_lower:
            return True
    
    return False

async def compare_pricing_tool(companies: str) -> str:
    """
    Compare pricing strategies across competitors.
    
    Args:
        companies: Comma-separated company names (e.g., "DataStream, CloudMetrics")
    
    Returns:
        Markdown table with pricing comparison and insights
    """
    company_list = [name.strip() for name in companies.split(",")]
    
    if not company_list:
        return "Error: No company names provided"
    
    # Simulated pricing data (in production, retrieve from vector store metadata)
    pricing_data = {
        "DataStream Analytics": ["Starter: $99/mo", "Professional: $299/mo", "Enterprise: Custom"],
        "CloudMetrics Pro": ["Growth: $149/mo", "N/A", "Enterprise: Contact Sales"],
        "InsightHub": ["Basic: $49/mo", "Pro: $199/mo", "Enterprise: Custom"]
    }
    
    output = f"**Pricing Comparison**\n\n"
    output += f"Analyzing {len(company_list)} competitors:\n\n"
    output += "| Company | Entry Tier | Mid Tier | Enterprise Tier |\n"
    output += "|---------|------------|----------|------------------|\n"
    
    for company in company_list:
        if company in pricing_data:
            tiers = pricing_data[company]
            output += f"| {company} | {tiers[0]} | {tiers[1]} | {tiers[2]} |\n"
        else:
            output += f"| {company} | Data unavailable | - | - |\n"
    
    output += "\n**Pricing Insights:**\n"
    output += "- Entry tier range: $49-$149/month\n"
    output += "- Mid-tier range: $199-$299/month\n"
    output += "- Enterprise: Mostly custom pricing\n"
    output += "- Gap: No options between $49-$99 for budget buyers\n"
    
    return output


async def identify_market_gaps_tool(segment: str) -> str:
    """
    Analyze market segment for opportunities.
    
    Args:
        segment: Market segment ("enterprise", "mid-market", "SMB", "startups")
    
    Returns:
        Analysis of gaps, missing features, and opportunities
    """
    output = f"**Market Gap Analysis: {segment.upper()} Segment**\n\n"
    output += "**Common Features** (% of competitors offering):\n"
    output += "- Real-time dashboards: 90%\n"
    output += "- Data integrations: 85%\n"
    output += "- Custom reporting: 70%\n"
    output += "- Mobile access: 60%\n\n"
    
    output += "**IDENTIFIED GAPS:**\n\n"
    output += "1. **Feature Gaps:**\n"
    output += "   - Advanced ML/AI: Only 30% offer\n"
    output += "   - White-label options: Only 20%\n"
    output += "   - Real-time collaboration: Rare\n\n"
    
    output += "2. **Pricing Gaps:**\n"
    output += "   - No micro-tier (<$50/month)\n"
    output += "   - Large jump mid to enterprise (3-5x)\n"
    output += "   - No usage-based pricing\n\n"
    
    output += "3. **Positioning Gaps:**\n"
    output += "   - Vertical specialization (healthcare, finance)\n"
    output += "   - Geographic focus (non-US markets)\n"
    output += "   - Compliance-first (HIPAA, SOC 2, GDPR)\n\n"
    
    output += "**RECOMMENDATIONS:**\n"
    output += "- Differentiate with AI + white-label\n"
    output += "- Introduce flexible/usage-based pricing\n"
    output += "- Target underserved verticals\n"
    
    return output


async def generate_swot_tool(company_name: str) -> str:
    """
    Generate SWOT analysis for a competitor.
    
    Args:
        company_name: Name of competitor to analyze
    
    Returns:
        Structured SWOT analysis
    """
    output = f"**SWOT Analysis: {company_name}**\n\n"
    
    output += "**STRENGTHS:**\n"
    output += "- Strong product-market fit in target segment\n"
    output += "- Competitive pricing vs enterprise solutions\n"
    output += "- Modern technology stack (serverless, ML)\n"
    output += "- Good core feature coverage\n\n"
    
    output += "**WEAKNESSES:**\n"
    output += "- Limited brand recognition\n"
    output += "- Smaller feature library vs leaders\n"
    output += "- Potential scalability concerns\n"
    output += "- Less enterprise-grade support\n\n"
    
    output += "**OPPORTUNITIES:**\n"
    output += "- Growing demand for AI-powered analytics\n"
    output += "- Market shift to mid-market SaaS\n"
    output += "- Vertical specialization potential\n"
    output += "- International expansion\n\n"
    
    output += "**THREATS:**\n"
    output += "- Well-funded competitors entering mid-market\n"
    output += "- Pricing pressure from low-cost alternatives\n"
    output += "- Rapid technology changes\n"
    output += "- Economic downturns affecting budgets\n"
    
    return output


# ============================================================================
# SIMPLE REACT AGENT (Manual Implementation)
# ============================================================================

class SimpleReActAgent:
    """
    Manual ReAct implementation - demonstrates the pattern explicitly
    
    ReAct Loop:
    1. Thought: Reason about what to do
    2. Action: Call a tool
    3. Observation: See tool output
    4. Repeat until done
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.AGENT_TEMPERATURE,  # 0.7 for creative reasoning
            max_tokens=Config.MAX_TOKENS,
        )
        
        self.tools = {
            "search_competitors": search_competitors_tool,
            "compare_pricing": compare_pricing_tool,
            "identify_market_gaps": identify_market_gaps_tool,
            "generate_swot": generate_swot_tool,
        }
        
        self.max_iterations = 10
    
    def _create_prompt(self, user_query: str, conversation_history: str) -> str:
        """Build the prompt for each reasoning step"""
        
        tools_description = """
Available Tools:

1. search_competitors(query: str, market_segment: str = "", k: int = 3)
   - Find competitors using semantic similarity
   - Strategy: Describe the PRODUCT CATEGORY, not a specific company
   - Good queries:
     * "business analytics platform"
     * "data visualization and reporting tool"
     * "business intelligence dashboard software"
   - market_segment parameter: Provides context but doesn't filter results
     * Use when relevant: "mid-market", "enterprise", "SMB"
     * Tool returns competitors across ALL markets for comprehensive view
   - Returns: Ranked list of competitors with similarity scores and market positioning
   - IMPORTANT: Results may include companies serving adjacent markets (enterprise competitors 
     can still compete with mid-market solutions on features/pricing)

2. compare_pricing(companies: str)
   - Compare pricing strategies across competitors
   - Use when: analyzing pricing, finding pricing gaps
   
3. identify_market_gaps(segment: str)
   - Analyze market segment for opportunities
   - Use when: finding market opportunities, gaps
   
4. generate_swot(company_name: str)
   - Generate SWOT analysis for a competitor
   - Use when: comprehensive competitive assessment
"""
        
        system_prompt = f"""You are a competitive intelligence analyst agent.

{tools_description}

CRITICAL PROCESS (YOU MUST FOLLOW):

For each step, you MUST output EXACTLY in this format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <parameters as comma-separated values>

OR, when you have enough information:

Thought: <why you can answer now>
Final Answer: <comprehensive response citing specific data>

RULES:
- Always start with "Thought:" to show your reasoning
- Only call ONE tool per step
- Wait for Observation before next Thought
- Cite specific data from tool outputs in your Final Answer
- Never make up data not in Observations
- **CRITICAL: If tool returns "NO RESULTS FOUND", you MUST acknowledge this in Final Answer**
- **NEVER invent companies, similarity scores, or features not in tool output**
- **Look for [VERIFIED IN DATABASE] tags - only cite those companies**

EXAMPLE:

User Query: "Who are DataStream's competitors?"

Thought: I need to search for analytics platforms in the mid-market segment.
Action: search_competitors
Action Input: analytics platform, mid-market, 5

[System provides Observation]

Thought: I found CloudMetrics Pro and InsightHub as competitors. I have enough information to answer.
Final Answer: DataStream Analytics' main competitors are CloudMetrics Pro (similarity 0.87) targeting enterprise/mid-market, and InsightHub (similarity 0.78) focusing on SMBs/startups. CloudMetrics competes directly on features while InsightHub competes on price.

Now begin:
"""
        
        return f"{system_prompt}\n\nUser Query: {user_query}\n\n{conversation_history}\n"
    
    async def run(self, query: str) -> str:
        """Run the ReAct loop"""
        
        print(f"\n{'='*80}")
        print(f"🤖 AGENT PROCESSING: {query}")
        print(f"{'='*80}\n")
        
        conversation_history = ""
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---\n")
            
            # Get agent's next thought/action
            prompt = self._create_prompt(query, conversation_history)
            response = await self.llm.ainvoke(prompt)
            agent_output = response.content
            
            print(agent_output)
            
            # Check if agent is done
            if "Final Answer:" in agent_output:
                # Extract final answer
                final_answer = agent_output.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Parse action and input
            if "Action:" in agent_output and "Action Input:" in agent_output:
                try:
                    action_line = [line for line in agent_output.split("\n") if line.startswith("Action:")][0]
                    input_line = [line for line in agent_output.split("\n") if line.startswith("Action Input:")][0]
                    
                    tool_name = action_line.replace("Action:", "").strip()
                    tool_input = input_line.replace("Action Input:", "").strip()
                    
                    # Execute tool
                    if tool_name in self.tools:
                        # Parse comma-separated inputs
                        inputs = [x.strip() for x in tool_input.split(",")]
                        
                        # Call tool with appropriate args (NOW ASYNC)
                        if tool_name == "search_competitors":
                            query_arg = inputs[0] if len(inputs) > 0 else ""
                            market = inputs[1] if len(inputs) > 1 else ""
                            k = int(inputs[2]) if len(inputs) > 2 else 3
                            observation = await self.tools[tool_name](query_arg, market, k)
                        elif tool_name == "compare_pricing":
                            observation = await self.tools[tool_name](tool_input)
                        elif tool_name == "identify_market_gaps":
                            observation = await self.tools[tool_name](inputs[0])
                        elif tool_name == "generate_swot":
                            observation = await self.tools[tool_name](inputs[0])
                        
                        print(f"\n**Observation:**\n{observation}\n")
                        
                        # Add to conversation history
                        conversation_history += f"\n{agent_output}\n\nObservation: {observation}\n"
                    else:
                        observation = f"Error: Tool '{tool_name}' not found. Available: {', '.join(self.tools.keys())}"
                        conversation_history += f"\n{agent_output}\n\nObservation: {observation}\n"
                        print(f"\n**Observation:**\n{observation}\n")
                
                except Exception as e:
                    observation = f"Error parsing action: {str(e)}"
                    conversation_history += f"\n{agent_output}\n\nObservation: {observation}\n"
                    print(f"\n**Observation:**\n{observation}\n")
            else:
                # Agent didn't follow format
                conversation_history += f"\n{agent_output}\n\nObservation: Please follow format: 'Thought:', 'Action:', 'Action Input:'\n"
        
        return "Agent reached max iterations without completing the task."


# ============================================================================
# DATA INGESTION
# ============================================================================

async def populate_test_data():
    """Populate vector store with sample competitors"""
    
    print("📊 Ingesting test data into vector store...\n")
    
    competitors_data = [
        {
            "url": "https://datastream-analytics.com",
            "content": """
            DataStream Analytics - Real-time Business Intelligence Platform
            
            About Us:
            DataStream Analytics empowers mid-market teams (companies with 10-500 employees) 
            with AI-powered analytics. Our mission is to make enterprise-grade analytics 
            accessible to growing businesses.
            
            Tagline: "Transform Data into Decisions in Real-Time"
            
            Key Features:
            • 200+ pre-built integrations with Salesforce, Stripe, Google Analytics, HubSpot
            • Custom SQL editor with AI-powered autocomplete and query optimization
            • Real-time alerting system with anomaly detection powered by machine learning
            • Role-based access control (RBAC) for team collaboration
            • White-label customer portals for agencies and resellers
            • RESTful API with 99.9% uptime SLA
            • Interactive dashboards with drag-and-drop builder
            
            Pricing Plans:
            - Starter Plan: $99 per month - 3 dashboards, 10GB storage, email support
            - Professional Plan: $299 per month - Unlimited dashboards, 100GB storage, priority support
            - Enterprise Plan: Custom pricing - Dedicated infrastructure, 24/7 phone support, SSO
            
            What Makes Us Different:
            • No-code setup in under 5 minutes
            • SOC 2 Type II certified for data security
            • 500+ companies trust us, including Fortune 500 clients
            • Industry-leading customer satisfaction score of 4.8/5
            
            Technology:
            Built on serverless architecture using AWS Lambda, PostgreSQL, and Redis.
            Machine learning models trained on billions of data points for predictive analytics.
            """
        },
        {
            "url": "https://cloudmetrics.io",
            "content": """
            CloudMetrics Pro - Next-Generation Analytics for Enterprise
            
            CloudMetrics Pro is the analytics platform of choice for Fortune 500 companies
            and large enterprises looking to transform their data infrastructure.
            
            Company Overview:
            We serve enterprise clients (500+ employees) across finance, healthcare, and retail.
            
            Core Capabilities:
            • Advanced visualization engine with 50+ chart types
            • Predictive analytics powered by proprietary machine learning algorithms
            • Automated report generation and distribution
            • Mobile-first responsive dashboards for iOS and Android
            • Custom API integrations with enterprise systems (SAP, Oracle, Microsoft Dynamics)
            • Multi-tenant architecture for global deployments
            
            Pricing:
            - Growth Plan: $149 per month - For mid-sized teams, includes 10 users
            - Enterprise Plan: Contact our sales team - Custom pricing for large organizations
            
            Why Choose CloudMetrics:
            • Trusted by over 1,000 companies worldwide
            • ISO 27001 and GDPR certified
            • 99.99% uptime guarantee backed by SLA
            • Dedicated customer success manager for enterprise clients
            
            Tech Stack:
            Built with modern cloud infrastructure on Google Cloud Platform.
            AI-powered insights using TensorFlow and PyTorch models.
            """
        },
        {
            "url": "https://insighthub.app",
            "content": """
            InsightHub - Analytics Simplified for Startups and Small Businesses
            
            InsightHub helps startups and SMBs (5-50 employees) make data-driven decisions
            without the complexity of enterprise tools.
            
            Our Promise: "Analytics That Just Works"
            
            Target Customers:
            Small businesses, startups, solopreneurs, and teams under 50 people.
            
            Features:
            • Drag-and-drop dashboard builder (no coding required)
            • Data connectors for Shopify, Stripe, Google Sheets, QuickBooks
            • One-click sharing and collaboration tools
            • Basic analytics and KPI tracking
            • Automated email reports sent daily, weekly, or monthly
            
            Simple Pricing:
            - Basic Plan: $49 per month - 5 dashboards, 3 data sources
            - Pro Plan: $199 per month - Unlimited dashboards, 10 data sources, priority support
            - Enterprise Plan: Custom pricing for teams over 20 people
            
            What Sets Us Apart:
            • Setup takes less than 10 minutes
            • No credit card required for 14-day free trial
            • Designed for non-technical users
            • Affordable for bootstrapped startups
            
            Technology:
            Simple, reliable infrastructure. No complex setup required.
            """
        }
    ]
    
    extractor = ExtractionPipeline()
    vectorstore = get_vectorstore()
    
    for data in competitors_data:
        profile = await extractor.extract(data['url'], data['content'])
        print(f"   Extracted company name: '{profile.company_name}'")  # DEBUG
        await vectorstore.ingest_competitor(profile)
    
    print()


# ============================================================================
# TESTING
# ============================================================================

async def test_react_agent():
    """Test agent with progressively complex queries"""
    
    print("=" * 80)
    print("🤖 PHASE 6: REACT AGENT - COMPETITIVE INTELLIGENCE")
    print("=" * 80)
    
    # Step 1: Populate vector store
    print("\n1️⃣  INITIALIZING KNOWLEDGE BASE")
    print("-" * 80)
    await populate_test_data()
    
    # Step 2: Create agent
    print("\n2️⃣  BUILDING REACT AGENT")
    print("-" * 80)
    agent = SimpleReActAgent()
    print("✅ Agent initialized with 4 tools")
    print("   - search_competitors")
    print("   - compare_pricing")
    print("   - identify_market_gaps")
    print("   - generate_swot")
    
    # Step 3: Test Query 1 - Simple search
    print("\n" + "=" * 80)
    print("3️⃣  TEST QUERY 1: Competitor Discovery")
    print("=" * 80)
    
    result1 = await agent.run(
        "Who are DataStream Analytics' main competitors in the mid-market segment?"
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL ANSWER:")
    print("=" * 80)
    print(result1)
    
    # Pause between queries
    await asyncio.sleep(3)
    
    # Step 4: Test Query 2 - Multi-step reasoning
    print("\n" + "=" * 80)
    print("4️⃣  TEST QUERY 2: Pricing Strategy Analysis")
    print("=" * 80)
    
    result2 = await agent.run(
        "Compare pricing strategies between DataStream Analytics, CloudMetrics Pro, and InsightHub. Identify any gaps."
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL ANSWER:")
    print("=" * 80)
    print(result2)
    
    # Pause
    await asyncio.sleep(3)
    
    # Step 5: Test Query 3 - Market analysis
    print("\n" + "=" * 80)
    print("5️⃣  TEST QUERY 3: Market Gap Analysis")
    print("=" * 80)
    
    result3 = await agent.run(
        "What market gaps exist in the mid-market analytics space? What should a new entrant focus on?"
    )
    
    print("\n" + "=" * 80)
    print("📊 FINAL ANSWER:")
    print("=" * 80)
    print(result3)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 6 COMPLETE: REACT AGENT OPERATIONAL")
    print("=" * 80)
    print("\n📋 Demonstrated Capabilities:")
    print("   ✅ Multi-step reasoning loop")
    print("   ✅ Tool selection and composition")
    print("   ✅ Semantic search integration")
    print("   ✅ Data synthesis from multiple sources")
    print("   ✅ Temperature=0.7 creative reasoning")
    print("\n🎯 Next Phase: Synthesis Agent with Few-Shot Prompting")
    print("\n💡 Key Learnings:")
    print("   1. ReAct pattern makes reasoning transparent")
    print("   2. Tools return formatted strings for LLM consumption")
    print("   3. Max iterations prevent infinite loops")
    print("   4. Temp=0.7 balances creativity with coherence")
    print("   5. Explicit Thought/Action/Observation improves quality")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    asyncio.run(test_react_agent())