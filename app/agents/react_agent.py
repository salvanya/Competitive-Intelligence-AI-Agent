"""
ReAct Agent Implementation
Reason + Act pattern for competitive intelligence analysis
"""

from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import AppConfig
from app.agents.tools import CompetitorTools
from app.extraction.schemas import CompetitorProfile


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for competitive analysis.
    
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
        tools: CompetitorTools instance for analysis
        max_iterations: Maximum reasoning loops (default: 10)
    
    Example:
        >>> agent = ReActAgent(api_key, config, tools)
        >>> result = await agent.run("Compare pricing strategies")
        >>> print(result)
    """
    
    def __init__(
        self, 
        api_key: str, 
        config: AppConfig, 
        tools: CompetitorTools,
        max_iterations: int = 10
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            api_key: Google AI Studio API key
            config: Application configuration
            tools: CompetitorTools instance with loaded profiles
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
            "search_competitors": self.tools.search_competitors,
            "compare_pricing": self.tools.compare_pricing,
            "identify_feature_gaps": self.tools.identify_feature_gaps,
            "analyze_target_markets": self.tools.analyze_target_markets,
            "get_technology_overview": self.tools.get_technology_overview,
            "get_comprehensive_summary": self.tools.get_comprehensive_summary,
        }
        
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create ReAct prompt template with tool descriptions.
        
        Returns:
            ChatPromptTemplate: Configured prompt
        """
        system_prompt = """You are a competitive intelligence analyst using the ReAct (Reasoning + Acting) framework.

Your goal is to answer questions about competitors by:
1. **Thinking** about what information you need
2. **Acting** by calling appropriate tools
3. **Observing** the tool outputs
4. **Reasoning** about the observations
5. Repeating until you have enough information to provide a comprehensive answer

**Available Tools:**

1. **search_competitors(query: str, k: int = 3)**
   - Semantic search for competitors matching a query
   - Example: search_competitors("enterprise pricing models", k=2)

2. **compare_pricing(companies: List[str] = None)**
   - Compare pricing strategies across competitors
   - Example: compare_pricing(["Company A", "Company B"])

3. **identify_feature_gaps(reference_company: str = None)**
   - Identify feature gaps and unique offerings
   - Example: identify_feature_gaps("Company A")

4. **analyze_target_markets()**
   - Analyze target market positioning
   - Example: analyze_target_markets()

5. **get_technology_overview()**
   - Analyze technology stack trends
   - Example: get_technology_overview()

6. **get_comprehensive_summary()**
   - Get overview of all competitors
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
- Use multiple tools if needed
- Be specific in your analysis
- Cite evidence from tool outputs
- Provide actionable insights
"""
        
        human_prompt = """Query: {query}

Available Competitors:
{competitor_list}

Begin your analysis using the ReAct framework."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def run(
        self, 
        query: str, 
        profiles: Optional[List[CompetitorProfile]] = None
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
            ...     "What are the main pricing differences between competitors?"
            ... )
        """
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        # Use profiles from tools if not provided
        if profiles is None:
            profiles = self.tools.valid_profiles
        
        if not profiles:
            return "No valid competitor profiles available for analysis"
        
        # Create competitor list summary
        competitor_list = "\n".join([
            f"- {p.company_name} ({p.website_url})"
            for p in profiles
        ])
        
        # Initialize conversation history
        conversation_history = []
        
        # Start ReAct loop
        for iteration in range(self.max_iterations):
            # Build current prompt
            if iteration == 0:
                # First iteration - use initial prompt
                messages = self.prompt.format_messages(
                    query=query,
                    competitor_list=competitor_list
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
            # Simple parsing - in production, would use more robust parsing
            if tool_input.lower() in ['none', 'null', '']:
                result = await tool_func() if tool_name == "search_competitors" else tool_func()
            else:
                # For search_competitors, call with query
                if tool_name == "search_competitors":
                    # Extract query from input
                    query = tool_input.strip('"\'')
                    result = await tool_func(query)
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