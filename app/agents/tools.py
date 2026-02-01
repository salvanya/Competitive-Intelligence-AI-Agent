"""
Tools for ReAct Agent
Implements competitive intelligence analysis tools for agent reasoning
"""

from typing import List, Dict, Any, Optional
from app.extraction.schemas import CompetitorProfile, PricingTier
from app.vectorstore.store import CompetitorVectorStore


class CompetitorTools:
    """
    Tools for competitive intelligence reasoning and analysis.
    
    Provides a suite of specialized tools that the ReAct agent can use
    to analyze competitor data, compare features, identify gaps, and
    generate insights.
    
    Attributes:
        vectorstore: Vector store for semantic search
        profiles: List of all competitor profiles
    
    Example:
        >>> tools = CompetitorTools(vectorstore, profiles)
        >>> result = await tools.search_competitors("enterprise SaaS pricing")
        >>> print(result)
    """
    
    def __init__(
        self, 
        vectorstore: CompetitorVectorStore,
        profiles: List[CompetitorProfile]
    ):
        """
        Initialize competitor tools.
        
        Args:
            vectorstore: Initialized vector store with ingested profiles
            profiles: List of all competitor profiles
        """
        self.vectorstore = vectorstore
        self.profiles = profiles
        
        # Filter out failed scrapes for analysis
        self.valid_profiles = [
            p for p in profiles if p.scrape_success
        ]
    
    async def search_competitors(self, query: str, k: int = 3) -> str:
        """
        Search for competitors matching a semantic query.
        
        Uses vector search to find competitors most relevant to the query.
        Returns a formatted string suitable for LLM consumption.
        
        Args:
            query: Semantic search query (e.g., "enterprise pricing models")
            k: Number of results to return (default: 3)
        
        Returns:
            str: Formatted search results
        
        Example:
            >>> result = await tools.search_competitors("AI-powered features", k=2)
        """
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        results = await self.vectorstore.search_similar(query, k=k)
        
        if not results:
            return f"No competitors found matching query: '{query}'"
        
        output = [f"**Found {len(results)} competitor(s) matching '{query}':**\n"]
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. **{result['company']}** (Similarity: {result['similarity_score']:.3f})")
            output.append(f"   URL: {result['website_url']}")
            output.append(f"   Target Market: {result['target_market']}")
            output.append(f"   Features: {result['num_features']}")
            
            # Include snippet of content
            content_preview = result['content'][:300].replace('\n', ' ')
            output.append(f"   Preview: {content_preview}...")
            output.append("")
        
        return "\n".join(output)
    
    def compare_pricing(self, companies: Optional[List[str]] = None) -> str:
        """
        Compare pricing strategies across competitors.
        
        Analyzes pricing tiers, identifies patterns, and highlights differences.
        
        Args:
            companies: Optional list of company names to compare
                      (if None, compares all valid profiles)
        
        Returns:
            str: Formatted pricing comparison analysis
        
        Example:
            >>> result = tools.compare_pricing(["Acme Corp", "TechCo"])
        """
        # Filter profiles if specific companies requested
        if companies:
            profiles_to_compare = [
                p for p in self.valid_profiles 
                if p.company_name in companies
            ]
        else:
            profiles_to_compare = self.valid_profiles
        
        if not profiles_to_compare:
            return "No valid profiles available for pricing comparison"
        
        output = ["**Pricing Strategy Comparison:**\n"]
        
        for profile in profiles_to_compare:
            output.append(f"**{profile.company_name}**")
            
            if not profile.pricing_tiers:
                output.append("  - No pricing information available")
            else:
                output.append(f"  - Number of tiers: {len(profile.pricing_tiers)}")
                
                for tier in profile.pricing_tiers:
                    tier_info = f"  - {tier.name}: {tier.price or 'N/A'}"
                    if tier.features:
                        tier_info += f" ({len(tier.features)} features)"
                    output.append(tier_info)
            
            output.append("")
        
        # Analysis section
        output.append("**Pricing Insights:**")
        
        # Count tiers across all competitors
        tier_counts = [len(p.pricing_tiers) for p in profiles_to_compare if p.pricing_tiers]
        if tier_counts:
            avg_tiers = sum(tier_counts) / len(tier_counts)
            output.append(f"- Average tiers per competitor: {avg_tiers:.1f}")
        
        # Identify "Contact Sales" models
        contact_sales_count = sum(
            1 for p in profiles_to_compare 
            for tier in p.pricing_tiers 
            if tier.price and "Contact Sales" in tier.price
        )
        if contact_sales_count > 0:
            output.append(f"- {contact_sales_count} tier(s) use custom/enterprise pricing")
        
        return "\n".join(output)
    
    def identify_feature_gaps(self, reference_company: Optional[str] = None) -> str:
        """
        Identify feature gaps and unique offerings across competitors.
        
        Analyzes which features are common vs. unique, and identifies
        potential market gaps.
        
        Args:
            reference_company: Optional company to use as reference point
        
        Returns:
            str: Formatted feature gap analysis
        
        Example:
            >>> result = tools.identify_feature_gaps("Acme Corp")
        """
        if not self.valid_profiles:
            return "No valid profiles available for feature analysis"
        
        output = ["**Feature Gap Analysis:**\n"]
        
        # Collect all features across competitors
        all_features = {}
        for profile in self.valid_profiles:
            for feature in profile.key_features:
                feature_lower = feature.lower()
                if feature_lower not in all_features:
                    all_features[feature_lower] = {
                        "original": feature,
                        "companies": []
                    }
                all_features[feature_lower]["companies"].append(profile.company_name)
        
        # Identify common features (in >50% of competitors)
        threshold = len(self.valid_profiles) / 2
        common_features = {
            k: v for k, v in all_features.items() 
            if len(v["companies"]) >= threshold
        }
        
        # Identify unique features (only 1 competitor)
        unique_features = {
            k: v for k, v in all_features.items() 
            if len(v["companies"]) == 1
        }
        
        # Common features
        if common_features:
            output.append(f"**Common Features** (in ≥{int(threshold)} competitor(s)):")
            for feature, data in sorted(common_features.items(), 
                                       key=lambda x: len(x[1]["companies"]), 
                                       reverse=True):
                output.append(f"- {data['original']} ({len(data['companies'])} competitors)")
            output.append("")
        
        # Unique features
        if unique_features:
            output.append("**Unique Features** (competitive differentiators):")
            for feature, data in unique_features.items():
                output.append(f"- {data['original']} (only {data['companies'][0]})")
            output.append("")
        
        # Reference company analysis
        if reference_company:
            ref_profile = next(
                (p for p in self.valid_profiles if p.company_name == reference_company),
                None
            )
            
            if ref_profile:
                output.append(f"**{reference_company} vs. Market:**")
                
                # Features they have that others don't
                their_unique = [
                    f for f in ref_profile.key_features
                    if f.lower() in unique_features
                ]
                if their_unique:
                    output.append(f"- Unique advantages: {', '.join(their_unique)}")
                
                # Common features they're missing
                missing_common = [
                    data['original'] for f, data in common_features.items()
                    if not any(kf.lower() == f for kf in ref_profile.key_features)
                ]
                if missing_common:
                    output.append(f"- Missing common features: {', '.join(missing_common)}")
        
        return "\n".join(output)
    
    def analyze_target_markets(self) -> str:
        """
        Analyze target market positioning across competitors.
        
        Returns:
            str: Formatted target market analysis
        
        Example:
            >>> result = tools.analyze_target_markets()
        """
        if not self.valid_profiles:
            return "No valid profiles available for market analysis"
        
        output = ["**Target Market Analysis:**\n"]
        
        # Group by target market
        market_groups: Dict[str, List[str]] = {}
        no_market_specified = []
        
        for profile in self.valid_profiles:
            if profile.target_market:
                market = profile.target_market
                if market not in market_groups:
                    market_groups[market] = []
                market_groups[market].append(profile.company_name)
            else:
                no_market_specified.append(profile.company_name)
        
        # Display market segments
        if market_groups:
            for market, companies in sorted(market_groups.items()):
                output.append(f"**{market}:**")
                for company in companies:
                    output.append(f"  - {company}")
                output.append("")
        
        if no_market_specified:
            output.append("**Unspecified Target Market:**")
            for company in no_market_specified:
                output.append(f"  - {company}")
            output.append("")
        
        # Insights
        output.append("**Market Insights:**")
        output.append(f"- Total market segments identified: {len(market_groups)}")
        
        if market_groups:
            most_crowded = max(market_groups.items(), key=lambda x: len(x[1]))
            output.append(f"- Most competitive segment: {most_crowded[0]} ({len(most_crowded[1])} competitors)")
        
        return "\n".join(output)
    
    def get_technology_overview(self) -> str:
        """
        Analyze technology stack trends across competitors.
        
        Returns:
            str: Formatted technology analysis
        
        Example:
            >>> result = tools.get_technology_overview()
        """
        if not self.valid_profiles:
            return "No valid profiles available for technology analysis"
        
        output = ["**Technology Stack Overview:**\n"]
        
        # Collect all technologies
        tech_usage: Dict[str, List[str]] = {}
        
        for profile in self.valid_profiles:
            for tech in profile.technology_stack:
                if tech not in tech_usage:
                    tech_usage[tech] = []
                tech_usage[tech].append(profile.company_name)
        
        if not tech_usage:
            return "No technology stack information available"
        
        # Sort by popularity
        sorted_tech = sorted(
            tech_usage.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        output.append("**Most Common Technologies:**")
        for tech, companies in sorted_tech[:10]:  # Top 10
            output.append(f"- {tech} ({len(companies)} competitor(s))")
            if len(companies) <= 3:
                output.append(f"  Used by: {', '.join(companies)}")
        
        return "\n".join(output)
    
    def get_comprehensive_summary(self) -> str:
        """
        Generate a comprehensive summary of all competitors.
        
        Returns:
            str: Formatted comprehensive summary
        
        Example:
            >>> result = tools.get_comprehensive_summary()
        """
        output = ["**Competitive Intelligence Summary:**\n"]
        
        output.append(f"**Overview:**")
        output.append(f"- Total competitors analyzed: {len(self.profiles)}")
        output.append(f"- Successful extractions: {len(self.valid_profiles)}")
        output.append(f"- Failed scrapes: {len(self.profiles) - len(self.valid_profiles)}")
        output.append("")
        
        if self.valid_profiles:
            output.append("**Competitors:**")
            for profile in self.valid_profiles:
                output.append(f"- {profile.company_name}")
                if profile.tagline:
                    output.append(f"  Tagline: {profile.tagline}")
                output.append(f"  Features: {len(profile.key_features)}")
                output.append(f"  Pricing Tiers: {len(profile.pricing_tiers)}")
                output.append("")
        
        return "\n".join(output)