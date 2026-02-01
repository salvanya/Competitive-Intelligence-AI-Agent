"""
ReAct Agent Implementation
Reason + Act pattern for AI news intelligence analysis
"""

from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import AppConfig
from app.agents.tools import NewsAnalysisTools
from app.extraction.schemas import NewsArticleProfile


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for AI news analysis.
    
    Implements the ReAct pattern where the agent:
    1. Thinks (reasons about what to do)
    2. Acts (calls tools to gather information)
    3. Observes (processes tool outputs)
    4. Repeats until it can provide a final answer
    
    Uses explicit <thinking> tags for transparent reasoning and
    chain-of-thought prompting for higher quality outputs.
    
    Attributes:
        config: Application configuration
        llm: Gemini LLM configured for reasoning (temp=0.7)
        tools: NewsAnalysisTools instance for analysis
        max_iterations: Maximum reasoning loops (default: 10)
    
    Example:
        >>> agent = ReActAgent(api_key, config, tools)
        >>> result = await agent.run("Identify the most critical AI developments")
        >>> print(result)
    """
    
    def __init__(
        self, 
        api_key: str, 
        config: AppConfig, 
        tools: NewsAnalysisTools,
        max_iterations: int = 10
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            api_key: Google AI Studio API key
            config: Application configuration
            tools: NewsAnalysisTools instance with loaded articles
            max_iterations: Maximum reasoning iterations (default: 10)
        
        Raises:
            ValueError: If API key is invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.config = config
        self.tools = tools
        self.max_iterations = max_iterations
        
        # Initialize LLM with creative temperature for reasoning
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.SYNTHESIS_TEMPERATURE,  # 0.7 for creative reasoning
            google_api_key=api_key
        )
        
        # Define available tool mappings
        self.tool_map = {
            "search_articles": self.tools.search_articles,
            "analyze_relevance": self.tools.analyze_relevance,
            "identify_technology_trends": self.tools.identify_technology_trends,
            "analyze_industry_impact": self.tools.analyze_industry_impact,
            "prioritize_articles": self.tools.prioritize_articles,
            "identify_use_cases": self.tools.identify_use_cases,
            "get_comprehensive_summary": self.tools.get_comprehensive_summary,
        }
        
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create ReAct prompt template with tool descriptions.
        
        Returns:
            ChatPromptTemplate: Configured prompt
        """
        system_prompt = """You are an AI news intelligence analyst using the ReAct (Reasoning + Acting) framework.

Your goal is to answer questions about AI news developments by:
1. **Thinking** about what information you need
2. **Acting** by calling appropriate tools
3. **Observing** the tool outputs
4. **Reasoning** about the observations
5. Repeating until you have enough information to provide a comprehensive answer

**Available Tools:**

1. **search_articles(query: str, k: int = 3)**
   - Semantic search for news articles matching a query
   - Example: search_articles("GPT-5 reasoning capabilities", k=2)

2. **analyze_relevance()**
   - Analyze relevance scores across all articles
   - Identifies high, medium, and low relevance developments
   - Example: analyze_relevance()

3. **identify_technology_trends()**
   - Identify emerging technology trends and patterns
   - Shows most mentioned technologies and high-impact tech
   - Example: identify_technology_trends()

4. **analyze_industry_impact()**
   - Analyze which industries are most affected
   - Groups impact by severity level
   - Example: analyze_industry_impact()

5. **prioritize_articles()**
   - Rank articles by recommended investigation priority (1-5)
   - Groups articles by urgency
   - Example: prioritize_articles()

6. **identify_use_cases()**
   - Identify and categorize practical use cases
   - Shows most common applications
   - Example: identify_use_cases()

7. **get_comprehensive_summary()**
   - Get overview of all news articles
   - Shows impact distribution, sources, recent articles
   - Example: get_comprehensive_summary()

**Response Format:**

For each iteration, use this structure:

<thinking>
[Your reasoning about what to do next]
- What information do I need?
- Which tool would help?
- What have I learned so far?
</thinking>

Action: [tool_name]
Action Input: [tool arguments]

Observation: [tool output will appear here]

<thinking>
[Your reasoning about the observation]
- What did I learn?
- Is this sufficient?
- Do I need more information?
</thinking>

When you have sufficient information:

Final Answer: [Your comprehensive analysis]

**Guidelines:**
- Think step-by-step
- Use multiple tools if needed to get comprehensive view
- Be specific in your analysis
- Cite evidence from tool outputs
- Provide actionable insights
- Focus on most critical/relevant developments
- Note patterns and connections between articles
"""
        
        human_prompt = """Query: {query}

Available Articles Overview:
{article_list}

Begin your analysis using the ReAct framework."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def run(
        self, 
        query: str, 
        profiles: Optional[List[NewsArticleProfile]] = None
    ) -> str:
        """
        Execute the ReAct reasoning loop.
        
        Iteratively calls LLM to reason and act until a final answer
        is reached or max iterations is hit.
        
        Args:
            query: User's analysis question
            profiles: Optional list of profiles (uses tools.profiles if None)
        
        Returns:
            str: Final analysis result
        
        Example:
            >>> result = await agent.run(
            ...     "What are the most important AI developments to investigate?"
            ... )
        """
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        # Use profiles from tools if not provided
        if profiles is None:
            profiles = self.tools.valid_profiles
        
        if not profiles:
            return "No valid news article profiles available for analysis"
        
        # Create article list summary
        article_list = "\n".join([
            f"- {p.headline} ({p.news_source}) - Impact: {p.potential_impact.value if p.potential_impact else 'N/A'}, Relevance: {p.relevance_score if p.relevance_score else 'N/A'}"
            for p in profiles[:10]  # Show first 10
        ])
        
        if len(profiles) > 10:
            article_list += f"\n... and {len(profiles) - 10} more articles"
        
        # Initialize conversation history
        conversation_history = []
        
        # Start ReAct loop
        for iteration in range(self.max_iterations):
            # Build current prompt
            if iteration == 0:
                # First iteration - use initial prompt
                messages = self.prompt.format_messages(
                    query=query,
                    article_list=article_list
                )
            else:
                # Subsequent iterations - append history
                messages = conversation_history
            
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            response_text = response.content
            
            # Check if we have a final answer
            if "Final Answer:" in response_text:
                # Extract final answer
                final_answer = response_text.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Parse action and action input
            action_result = self._parse_action(response_text)
            
            if action_result is None:
                # LLM didn't produce valid action - provide guidance
                observation = "Error: Please use the format 'Action: [tool_name]' and 'Action Input: [arguments]'"
            else:
                tool_name, tool_input = action_result
                
                # Execute tool
                observation = await self._execute_tool(tool_name, tool_input)
            
            # Add to conversation history
            conversation_history.append(("assistant", response_text))
            conversation_history.append(("human", f"Observation: {observation}\n\nContinue your analysis."))
        
        # Max iterations reached
        return f"Analysis incomplete after {self.max_iterations} iterations. Partial results:\n\n{response_text}"
    
    def _parse_action(self, response_text: str) -> Optional[tuple[str, str]]:
        """
        Parse action and action input from LLM response.
        
        Args:
            response_text: LLM response text
        
        Returns:
            Optional[tuple]: (tool_name, tool_input) or None if parsing failed
        """
        lines = response_text.split('\n')
        
        action_line = None
        input_line = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                action_line = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                input_line = line.replace("Action Input:", "").strip()
        
        if action_line and input_line:
            return (action_line, input_line)
        
        return None
    
    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input arguments for the tool
        
        Returns:
            str: Tool output or error message
        """
        # Check if tool exists
        if tool_name not in self.tool_map:
            available_tools = ", ".join(self.tool_map.keys())
            return f"Error: Tool '{tool_name}' not found. Available tools: {available_tools}"
        
        tool_func = self.tool_map[tool_name]
        
        try:
            # Parse tool input
            if tool_input.lower() in ['none', 'null', '', 'n/a']:
                # Tool requires no arguments
                result = await tool_func() if tool_name == "search_articles" else tool_func()
            else:
                # For search_articles, call with query
                if tool_name == "search_articles":
                    # Extract query from input (remove quotes if present)
                    query = tool_input.strip('"\'')
                    
                    # Check for k parameter
                    k = 3  # default
                    if ',' in query:
                        parts = query.split(',')
                        query = parts[0].strip()
                        # Try to extract k value
                        for part in parts[1:]:
                            if 'k=' in part.lower():
                                try:
                                    k = int(part.split('=')[1].strip())
                                except:
                                    pass
                    
                    result = await tool_func(query, k=k)
                else:
                    # For other tools, call without arguments
                    result = tool_func()
            
            return result
        
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List[str]: Tool names
        """
        return list(self.tool_map.keys())