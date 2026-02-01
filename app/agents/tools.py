"""
Tools for ReAct Agent
Implements AI news analysis tools for agent reasoning
"""

from typing import List, Dict, Any, Optional
from app.extraction.schemas import NewsArticleProfile, ImpactLevel
from app.vectorstore.store import NewsVectorStore


class NewsAnalysisTools:
    """
    Tools for AI news intelligence reasoning and analysis.
    
    Provides a suite of specialized tools that the ReAct agent can use
    to analyze news articles, identify trends, assess impact, and
    generate prioritized insights.
    
    Attributes:
        vectorstore: Vector store for semantic search
        profiles: List of all news article profiles
    
    Example:
        >>> tools = NewsAnalysisTools(vectorstore, profiles)
        >>> result = await tools.search_articles("GPT-5 reasoning capabilities")
        >>> print(result)
    """
    
    def __init__(
        self, 
        vectorstore: NewsVectorStore,
        profiles: List[NewsArticleProfile]
    ):
        """
        Initialize news analysis tools.
        
        Args:
            vectorstore: Initialized vector store with ingested articles
            profiles: List of all news article profiles
        """
        self.vectorstore = vectorstore
        self.profiles = profiles
        
        # Filter out failed scrapes for analysis
        self.valid_profiles = [
            p for p in profiles if p.scrape_success
        ]
    
    async def search_articles(self, query: str, k: int = 3) -> str:
        """
        Search for news articles matching a semantic query.
        
        Uses vector search to find articles most relevant to the query.
        Returns a formatted string suitable for LLM consumption.
        
        Args:
            query: Semantic search query (e.g., "LLM reasoning improvements")
            k: Number of results to return (default: 3)
        
        Returns:
            str: Formatted search results
        
        Example:
            >>> result = await tools.search_articles("multimodal AI", k=2)
        """
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        results = await self.vectorstore.search_similar(query, k=k)
        
        if not results:
            return f"No articles found matching query: '{query}'"
        
        output = [f"**Found {len(results)} article(s) matching '{query}':**\n"]
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. **{result['headline']}**")
            output.append(f"   Source: {result['news_source']}")
            output.append(f"   URL: {result['article_url']}")
            output.append(f"   Date: {result['publication_date']}")
            output.append(f"   Similarity: {result['similarity_score']:.3f}")
            output.append(f"   Impact: {result['potential_impact']}")
            output.append(f"   Relevance: {result['relevance_score']:.2f}")
            output.append(f"   Priority: {result['recommended_priority']}")
            
            # Include snippet of content
            content_preview = result['content'][:250].replace('\n', ' ')
            output.append(f"   Preview: {content_preview}...")
            output.append("")
        
        return "\n".join(output)
    
    def analyze_relevance(self) -> str:
        """
        Analyze relevance scores across all articles.
        
        Provides statistical analysis of article relevance and
        identifies highly relevant developments.
        
        Returns:
            str: Formatted relevance analysis
        
        Example:
            >>> result = tools.analyze_relevance()
        """
        if not self.valid_profiles:
            return "No valid articles available for relevance analysis"
        
        output = ["**Relevance Score Analysis:**\n"]
        
        # Filter articles with relevance scores
        scored_articles = [
            p for p in self.valid_profiles 
            if p.relevance_score is not None
        ]
        
        if not scored_articles:
            return "No articles have relevance scores assigned"
        
        # Calculate statistics
        scores = [p.relevance_score for p in scored_articles]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        output.append(f"**Overall Statistics:**")
        output.append(f"- Total articles analyzed: {len(scored_articles)}")
        output.append(f"- Average relevance: {avg_score:.2f}")
        output.append(f"- Highest relevance: {max_score:.2f}")
        output.append(f"- Lowest relevance: {min_score:.2f}")
        output.append("")
        
        # High relevance articles (>= 0.7)
        high_relevance = [p for p in scored_articles if p.relevance_score >= 0.7]
        if high_relevance:
            output.append(f"**High Relevance Articles** (score ≥ 0.7): {len(high_relevance)}")
            for article in sorted(high_relevance, key=lambda x: x.relevance_score, reverse=True)[:5]:
                output.append(f"- {article.headline} ({article.relevance_score:.2f})")
            output.append("")
        
        # Medium relevance (0.4 - 0.69)
        medium_relevance = [p for p in scored_articles if 0.4 <= p.relevance_score < 0.7]
        if medium_relevance:
            output.append(f"**Medium Relevance Articles** (0.4-0.69): {len(medium_relevance)}")
            output.append("")
        
        # Low relevance (< 0.4)
        low_relevance = [p for p in scored_articles if p.relevance_score < 0.4]
        if low_relevance:
            output.append(f"**Low Relevance Articles** (< 0.4): {len(low_relevance)}")
        
        return "\n".join(output)
    
    def identify_technology_trends(self) -> str:
        """
        Identify emerging technology trends across articles.
        
        Analyzes which technologies are most frequently mentioned
        and identifies emerging patterns.
        
        Returns:
            str: Formatted technology trend analysis
        
        Example:
            >>> result = tools.identify_technology_trends()
        """
        if not self.valid_profiles:
            return "No valid articles available for technology analysis"
        
        output = ["**Technology Trend Analysis:**\n"]
        
        # Collect all technologies
        tech_mentions: Dict[str, List[str]] = {}
        
        for profile in self.valid_profiles:
            for tech in profile.key_technologies:
                tech_lower = tech.lower()
                if tech_lower not in tech_mentions:
                    tech_mentions[tech_lower] = {
                        "original": tech,
                        "articles": []
                    }
                tech_mentions[tech_lower]["articles"].append(profile.headline)
        
        if not tech_mentions:
            return "No technologies identified in articles"
        
        # Sort by frequency
        sorted_tech = sorted(
            tech_mentions.items(),
            key=lambda x: len(x[1]["articles"]),
            reverse=True
        )
        
        output.append(f"**Most Mentioned Technologies:**")
        for tech, data in sorted_tech[:15]:  # Top 15
            count = len(data["articles"])
            output.append(f"- {data['original']} ({count} article{'s' if count > 1 else ''})")
            if count <= 3:
                output.append(f"  Mentioned in: {', '.join(data['articles'][:3])}")
        
        output.append("")
        
        # Identify technologies mentioned in high-impact articles
        high_impact_articles = [
            p for p in self.valid_profiles
            if p.potential_impact in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]
        ]
        
        if high_impact_articles:
            output.append(f"**Technologies in High-Impact News:**")
            high_impact_tech = set()
            for article in high_impact_articles:
                high_impact_tech.update(article.key_technologies)
            
            for tech in sorted(high_impact_tech):
                output.append(f"- {tech}")
        
        return "\n".join(output)
    
    def analyze_industry_impact(self) -> str:
        """
        Analyze industry impact across all articles.
        
        Identifies which industries are most affected by
        AI developments covered in the news.
        
        Returns:
            str: Formatted industry impact analysis
        
        Example:
            >>> result = tools.analyze_industry_impact()
        """
        if not self.valid_profiles:
            return "No valid articles available for industry analysis"
        
        output = ["**Industry Impact Analysis:**\n"]
        
        # Collect all industries
        industry_mentions: Dict[str, List[str]] = {}
        
        for profile in self.valid_profiles:
            for industry in profile.affected_industries:
                if industry not in industry_mentions:
                    industry_mentions[industry] = []
                industry_mentions[industry].append(profile.headline)
        
        if not industry_mentions:
            return "No industry impact information available"
        
        # Sort by frequency
        sorted_industries = sorted(
            industry_mentions.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        output.append(f"**Most Affected Industries:**")
        for industry, articles in sorted_industries[:10]:  # Top 10
            count = len(articles)
            output.append(f"- **{industry}** ({count} article{'s' if count > 1 else ''})")
            if count <= 2:
                for article in articles:
                    output.append(f"  • {article}")
        
        output.append("")
        
        # Industry impact by impact level
        output.append("**Industry Impact by Development Severity:**")
        
        for impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH, ImpactLevel.MEDIUM]:
            relevant_articles = [
                p for p in self.valid_profiles
                if p.potential_impact == impact_level
            ]
            
            if relevant_articles:
                industries = set()
                for article in relevant_articles:
                    industries.update(article.affected_industries)
                
                if industries:
                    output.append(f"\n**{impact_level.value} Impact:**")
                    output.append(f"Industries: {', '.join(sorted(industries))}")
        
        return "\n".join(output)
    
    def prioritize_articles(self) -> str:
        """
        Rank articles by recommended investigation priority.
        
        Organizes articles by priority level and provides
        actionable reading order.
        
        Returns:
            str: Formatted priority analysis
        
        Example:
            >>> result = tools.prioritize_articles()
        """
        if not self.valid_profiles:
            return "No valid articles available for prioritization"
        
        output = ["**Article Priority Rankings:**\n"]
        
        # Filter articles with priority assigned
        prioritized = [
            p for p in self.valid_profiles
            if p.recommended_priority is not None
        ]
        
        if not prioritized:
            return "No articles have priority rankings assigned"
        
        # Group by priority level
        priority_groups = {i: [] for i in range(1, 6)}
        
        for profile in prioritized:
            priority_groups[profile.recommended_priority].append(profile)
        
        # Priority labels
        priority_labels = {
            1: "🔴 URGENT - Investigate Immediately",
            2: "🟠 HIGH - Investigate Soon",
            3: "🟡 MEDIUM - Investigate When Relevant",
            4: "🟢 LOW - Investigate If Interested",
            5: "⚪ OPTIONAL - Low Priority"
        }
        
        for priority in range(1, 6):
            articles = priority_groups[priority]
            if articles:
                output.append(f"**Priority {priority}: {priority_labels[priority]}**")
                output.append(f"Count: {len(articles)} article{'s' if len(articles) > 1 else ''}")
                output.append("")
                
                # Sort by relevance score within priority
                sorted_articles = sorted(
                    articles,
                    key=lambda x: x.relevance_score or 0.0,
                    reverse=True
                )
                
                for article in sorted_articles:
                    impact = article.potential_impact.value if article.potential_impact else "N/A"
                    relevance = f"{article.relevance_score:.2f}" if article.relevance_score else "N/A"
                    
                    output.append(f"• **{article.headline}**")
                    output.append(f"  Source: {article.news_source} | Impact: {impact} | Relevance: {relevance}")
                    output.append("")
        
        return "\n".join(output)
    
    def identify_use_cases(self) -> str:
        """
        Identify and categorize use cases across articles.
        
        Returns:
            str: Formatted use case analysis
        
        Example:
            >>> result = tools.identify_use_cases()
        """
        if not self.valid_profiles:
            return "No valid articles available for use case analysis"
        
        output = ["**Use Case Analysis:**\n"]
        
        # Collect all use cases
        use_case_mentions: Dict[str, List[str]] = {}
        
        for profile in self.valid_profiles:
            for use_case in profile.use_cases:
                use_case_lower = use_case.lower()
                if use_case_lower not in use_case_mentions:
                    use_case_mentions[use_case_lower] = {
                        "original": use_case,
                        "articles": []
                    }
                use_case_mentions[use_case_lower]["articles"].append(profile.headline)
        
        if not use_case_mentions:
            return "No use cases identified in articles"
        
        # Sort by frequency
        sorted_use_cases = sorted(
            use_case_mentions.items(),
            key=lambda x: len(x[1]["articles"]),
            reverse=True
        )
        
        output.append(f"**Most Common Use Cases:**")
        for use_case, data in sorted_use_cases[:10]:  # Top 10
            count = len(data["articles"])
            output.append(f"- {data['original']} ({count} article{'s' if count > 1 else ''})")
        
        return "\n".join(output)
    
    def get_comprehensive_summary(self) -> str:
        """
        Generate a comprehensive summary of all news articles.
        
        Returns:
            str: Formatted comprehensive summary
        
        Example:
            >>> result = tools.get_comprehensive_summary()
        """
        output = ["**AI News Intelligence Summary:**\n"]
        
        output.append(f"**Overview:**")
        output.append(f"- Total articles analyzed: {len(self.profiles)}")
        output.append(f"- Successful extractions: {len(self.valid_profiles)}")
        output.append(f"- Failed scrapes: {len(self.profiles) - len(self.valid_profiles)}")
        output.append("")
        
        if self.valid_profiles:
            # Impact distribution
            impact_counts = {}
            for profile in self.valid_profiles:
                if profile.potential_impact:
                    level = profile.potential_impact.value
                    impact_counts[level] = impact_counts.get(level, 0) + 1
            
            if impact_counts:
                output.append("**Impact Distribution:**")
                for level in ["Critical", "High", "Medium", "Low"]:
                    count = impact_counts.get(level, 0)
                    if count > 0:
                        output.append(f"- {level}: {count} article{'s' if count > 1 else ''}")
                output.append("")
            
            # News sources
            sources = {}
            for profile in self.valid_profiles:
                source = profile.news_source
                sources[source] = sources.get(source, 0) + 1
            
            if sources:
                output.append("**News Sources:**")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    output.append(f"- {source}: {count} article{'s' if count > 1 else ''}")
                output.append("")
            
            # Recent articles (if dates available)
            dated_articles = [p for p in self.valid_profiles if p.publication_date]
            if dated_articles:
                output.append("**Recent Articles:**")
                sorted_by_date = sorted(
                    dated_articles,
                    key=lambda x: x.publication_date or "",
                    reverse=True
                )[:5]
                
                for article in sorted_by_date:
                    output.append(f"- {article.headline} ({article.publication_date})")
                output.append("")
        
        return "\n".join(output)