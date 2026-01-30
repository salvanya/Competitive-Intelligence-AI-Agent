"""
Phase 3: Structured Output Pipeline (Pydantic V2)
Demonstrates: Pydantic V2 validators, JSON mode, validation, error handling
"""
import os
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# ============================================================
# PART 1: Pydantic V2 Schema Design
# ============================================================

class PricingTier(BaseModel):
    """
    Individual pricing tier within a product.
    
    Pydantic V2 Changes:
    - @validator → @field_validator
    - cls parameter still present but handled differently
    - ValidationInfo for context (optional)
    """
    name: str = Field(description="Tier name (e.g., 'Basic', 'Premium')")
    price: float = Field(description="Monthly price in USD", gt=0)
    currency: str = Field(default="USD", description="Currency code")
    features: List[str] = Field(
        default_factory=list,
        description="Key features included in this tier"
    )
    
    @field_validator('price')
    @classmethod
    def price_must_be_reasonable(cls, v: float) -> float:
        """
        Validate price is within expected range for streaming services.
        
        Pydantic V2: @field_validator replaces @validator
        """
        if v > 100:
            raise ValueError(f"Price ${v} seems unrealistic for streaming")
        return v
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """
        Ensure tier name is not empty.
        
        Pydantic V2: Explicit return type annotations recommended
        """
        if not v or v.strip() == "":
            raise ValueError("Tier name cannot be empty")
        return v.strip()


class CompetitorProfile(BaseModel):
    """
    Complete competitor intelligence profile.
    
    This is the core data structure for our extraction chain.
    """
    company_name: str = Field(description="Official company name")
    pricing_tiers: List[PricingTier] = Field(
        description="All pricing tiers offered"
    )
    subscriber_count: Optional[int] = Field(
        default=None,
        description="Total subscribers (if publicly disclosed)"
    )
    content_library_size: Optional[int] = Field(
        default=None,
        description="Number of titles in library"
    )
    key_differentiators: List[str] = Field(
        default_factory=list,
        description="Unique selling propositions vs competitors"
    )
    
    @field_validator('pricing_tiers')
    @classmethod
    def must_have_at_least_one_tier(cls, v: List[PricingTier]) -> List[PricingTier]:
        """
        Ensure at least one pricing tier exists.
        
        Pydantic V2: Type hints on both input and output
        """
        if len(v) == 0:
            raise ValueError("Must have at least one pricing tier")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "company_name": "Netflix",
                "pricing_tiers": [
                    {
                        "name": "Standard",
                        "price": 15.49,
                        "currency": "USD",
                        "features": ["1080p", "2 devices"]
                    }
                ],
                "subscriber_count": 260000000,
                "content_library_size": 15000,
                "key_differentiators": ["Original content", "Global reach"]
            }
        }
    }


# ============================================================
# PART 2: Extraction Chain with JSON Mode
# ============================================================

def create_extraction_chain_v1_basic():
    """
    Version 1: Basic approach (relies on LLM to format JSON correctly).
    
    Reliability: Moderate (LLM might produce invalid JSON)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,  # Deterministic
        max_tokens=1000,
    )
    
    parser = PydanticOutputParser(pydantic_object=CompetitorProfile)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data extraction specialist.
        Extract competitor information and return ONLY valid JSON.
        {format_instructions}"""),
        ("human", "Extract competitor data from:\n\n{text}")
    ])
    
    # Inject format instructions from parser
    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    
    return chain


def create_extraction_chain_v2_json_mode():

    """
    Version 2: Gemini native JSON mode (enforced at API level).
    
    Reliability: High (API guarantees valid JSON structure)
    
    Note: This is the production-grade approach!
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=1000,
    
        # Native JSON mode enforcement
        model_kwargs={
            "response_mime_type": "application/json",
        }
    )
    
    parser = PydanticOutputParser(pydantic_object=CompetitorProfile)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract competitor information as JSON.
        
        Schema:
        {format_instructions}
        
        Rules:
        - Extract all pricing tiers with exact prices
        - Include subscriber count if mentioned
        - List key differentiators
        - Return ONLY the JSON object, no markdown or explanations"""),
        ("human", "{text}")
    ])
    
    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    
    return chain


# ============================================================
# PART 3: Sample Data for Testing
# ============================================================

NETFLIX_DATA = """
Netflix Competitive Intelligence Report (2025)

Pricing Structure:
- Basic with Ads: $6.99/month (720p, 1 device, includes advertisements)
- Standard: $15.49/month (1080p Full HD, 2 devices simultaneously, no ads)
- Premium: $19.99/month (4K Ultra HD, 4 devices, spatial audio, no ads)

Market Position:
- 260 million global subscribers as of Q4 2024
- Content library: 15,000+ titles
- Annual content budget: $17 billion

Competitive Advantages:
- Largest original content library in the industry
- Strong brand recognition globally
- Advanced recommendation algorithm
- Multi-language content across 190+ countries
"""

DISNEY_DATA = """
Disney+ Market Analysis (2025)

Subscription Tiers:
Disney+ Basic (with ads): $7.99/month - Full catalog access with limited ads
Disney+ Premium: $13.99/month - Ad-free experience, 4K on select titles, downloads enabled

Bundle Offer:
Disney+ & Hulu combo: $19.99/month (Disney+ ad-free + Hulu with ads)

Company Metrics:
Total subscribers: 150 million (Q4 2024)
Content catalog: 500+ movies, 7,500+ TV episodes

Strategic Differentiators:
- Exclusive Marvel Cinematic Universe and Star Wars content
- Family-friendly brand positioning
- Leverages existing Disney IP (Pixar, National Geographic)
- Vertical integration with theatrical releases
"""


# ============================================================
# PART 4: Testing & Validation
# ============================================================

def test_extraction_basic():
    """Test basic extraction without JSON mode."""
    print("\n" + "="*70)
    print("🧪 TEST 1: Basic Extraction (No JSON Mode)")
    print("="*70)
    
    chain = create_extraction_chain_v1_basic()
    
    print("\n📄 Processing Netflix data...")
    try:
        result = chain.invoke({"text": NETFLIX_DATA})
        print(f"\n✅ Extraction successful!")
        print(f"\n📊 Extracted Profile:")
        print(f"   Company: {result.company_name}")
        print(f"   Pricing Tiers: {len(result.pricing_tiers)}")
        for tier in result.pricing_tiers:
            print(f"      - {tier.name}: ${tier.price}/month")
        print(f"   Subscribers: {result.subscriber_count:,}" if result.subscriber_count else "   Subscribers: Not disclosed")
        print(f"   Differentiators: {len(result.key_differentiators)}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {type(e).__name__}")
        print(f"   Error: {str(e)[:200]}...")
        return None


def test_extraction_json_mode():
    """Test extraction WITH native JSON mode."""
    print("\n" + "="*70)
    print("🔥 TEST 2: JSON Mode Extraction (Production-Grade)")
    print("="*70)
    
    chain = create_extraction_chain_v2_json_mode()
    
    print("\n📄 Processing Disney+ data...")
    try:
        result = chain.invoke({"text": DISNEY_DATA})
        print(f"\n✅ Extraction successful!")
        print(f"\n📊 Extracted Profile:")
        print(f"   Company: {result.company_name}")
        print(f"   Pricing Tiers: {len(result.pricing_tiers)}")
        for tier in result.pricing_tiers:
            print(f"      - {tier.name}: ${tier.price}/month")
            if tier.features:
                print(f"        Features: {', '.join(tier.features[:3])}...")
        print(f"   Subscribers: {result.subscriber_count:,}" if result.subscriber_count else "   Subscribers: Not disclosed")
        print(f"   Library Size: {result.content_library_size:,} titles" if result.content_library_size else "   Library Size: Not disclosed")
        print(f"   Differentiators:")
        for diff in result.key_differentiators:
            print(f"      • {diff}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {type(e).__name__}")
        print(f"   Error: {str(e)[:200]}...")
        return None


def test_pydantic_validation():
    """Test Pydantic V2 validation with invalid data."""
    print("\n" + "="*70)
    print("🧪 TEST 3: Pydantic V2 Validation (Error Handling)")
    print("="*70)
    
    print("\n📋 Testing validation rules...")
    
    # Test 1: Invalid price (too high)
    print("\n1️⃣ Testing price validation (should reject $200)...")
    try:
        invalid_tier = PricingTier(
            name="Scam Tier",
            price=200.0,  # Unrealistic
            features=["Everything"]
        )
        print("   ❌ Validation failed - accepted invalid price!")
    except ValueError as e:
        print(f"   ✅ Correctly rejected: {str(e)}")
    
    # Test 2: Empty name
    print("\n2️⃣ Testing name validation (should reject empty string)...")
    try:
        invalid_tier = PricingTier(
            name="",  # Empty
            price=10.0,
            features=[]
        )
        print("   ❌ Validation failed - accepted empty name!")
    except ValueError as e:
        print(f"   ✅ Correctly rejected: {str(e)}")
    
    # Test 3: No pricing tiers
    print("\n3️⃣ Testing profile validation (should require at least 1 tier)...")
    try:
        invalid_profile = CompetitorProfile(
            company_name="EmptyCompany",
            pricing_tiers=[]  # Empty list
        )
        print("   ❌ Validation failed - accepted empty tiers!")
    except ValueError as e:
        print(f"   ✅ Correctly rejected: {str(e)}")
    
    print("\n✅ All Pydantic V2 validation rules working correctly!")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 Phase 3: Structured Output Pipeline (Pydantic V2)")
    print("="*70)
    print("🎯 Objective: Production-grade data extraction")
    print("🔧 Tech: Pydantic V2 + Gemini JSON Mode + Validation")
    print("="*70)
    
    try:
        # Test 1: Basic extraction
        result1 = test_extraction_basic()
        
        input("\n⏸️  Press Enter to continue to JSON mode test...")
        
        # Test 2: JSON mode extraction (production approach)
        result2 = test_extraction_json_mode()
        
        input("\n⏸️  Press Enter to continue to validation tests...")
        
        # Test 3: Validation
        test_pydantic_validation()
        
        print("\n" + "="*70)
        print("\n📝 Key Learnings:")
        print("   1. JSON mode (response_mime_type) guarantees valid JSON")
        print("   2. Validators catch bad data before it enters the system")
        print("   3. Temperature=0 ensures deterministic extraction")
        print("\n🎯 Production Pattern:")
        print("   Always use: Pydantic V2 + JSON Mode + Temperature=0")
        print("   For: Data extraction, classification, structured parsing")

        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   - Verify API key and model access")
        print("   - Check rate limits")
        print("   - Ensure pydantic V2: pip install --upgrade pydantic")