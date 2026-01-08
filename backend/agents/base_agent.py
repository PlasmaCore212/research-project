# backend/agents/base_agent.py
"""
Base Agent with ReAct (Reasoning + Acting) Pattern

This module implements the core agentic architecture based on:
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
- BDI Architecture (Rao & Georgeff, 1995)

The ReAct pattern enables agents to:
1. THINK: Reason about the current situation
2. ACT: Execute a chosen tool/action
3. OBSERVE: Process the result
4. REPEAT: Until goal is achieved or max iterations reached
"""

from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json


@dataclass
class AgentBelief:
    """Represents what the agent knows/believes about the world (BDI model)"""
    key: str
    value: Any
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentAction:
    """Represents an action the agent can take"""
    name: str
    description: str
    parameters: Dict[str, str]
    function: Optional[Callable] = None


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning chain"""
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class AgentState:
    """
    BDI-inspired agent state that persists across reasoning steps.
    
    Beliefs: What the agent knows about the world
    Desires: Goals the agent wants to achieve
    Intentions: Committed plans/actions
    """
    beliefs: Dict[str, AgentBelief] = field(default_factory=dict)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    reasoning_trace: List[ReActStep] = field(default_factory=list)
    
    def add_belief(self, key: str, value: Any, confidence: float = 1.0):
        """Update agent's beliefs based on new observations"""
        self.beliefs[key] = AgentBelief(key=key, value=value, confidence=confidence)
    
    def get_belief(self, key: str, default: Any = None) -> Any:
        """Retrieve a belief value"""
        belief = self.beliefs.get(key)
        return belief.value if belief else default
    
    def add_desire(self, goal: str):
        """Add a goal the agent wants to achieve"""
        if goal not in self.desires:
            self.desires.append(goal)
    
    def add_intention(self, plan: str):
        """Commit to a plan of action"""
        if plan not in self.intentions:
            self.intentions.append(plan)
    
    def get_context_summary(self) -> str:
        """Get a summary of current beliefs for context"""
        if not self.beliefs:
            return "No prior observations."
        
        summary_parts = []
        for key, belief in self.beliefs.items():
            summary_parts.append(f"- {key}: {belief.value}")
        return "\n".join(summary_parts)


class BaseReActAgent(ABC):
    """
    Abstract base class implementing the ReAct pattern for all agents.
    
    Each specialized agent (Flight, Hotel, Policy, Time) extends this class
    and defines its own tools and domain-specific logic.
    """
    
    def __init__(
        self, 
        agent_name: str,
        agent_role: str,
        model_name: str = "llama3.1:8b",
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        
        # Initialize agent state (BDI model)
        self.state = AgentState()
        
        # Tools available to this agent (defined by subclasses)
        self.tools: Dict[str, AgentAction] = {}
        
        # Message log for inter-agent communication tracking
        self.message_log: List[Dict] = []
    
    @abstractmethod
    def _register_tools(self) -> Dict[str, AgentAction]:
        """
        Register tools available to this agent.
        Must be implemented by each specialized agent.
        """
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the domain-specific system prompt for this agent.
        Must be implemented by each specialized agent.
        """
        pass
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the ReAct prompt"""
        tools_desc = []
        for name, action in self.tools.items():
            params = ", ".join([f"{k}: {v}" for k, v in action.parameters.items()])
            tools_desc.append(f"- {name}({params}): {action.description}")
        return "\n".join(tools_desc)
    
    def _create_react_prompt(self, goal: str, previous_steps: List[ReActStep]) -> str:
        """
        Create a ReAct prompt for the next reasoning step.
        
        This implements the Thought -> Action -> Observation loop from:
        "ReAct: Synergizing Reasoning and Acting in Language Models"
        """
        
        # Format previous reasoning steps
        history = ""
        if previous_steps:
            for step in previous_steps:
                history += f"""
Step {step.step_number}:
Thought: {step.thought}
Action: {step.action}
Action Input: {json.dumps(step.action_input)}
Observation: {step.observation}
"""
        
        # Get current beliefs/context
        context = self.state.get_context_summary()
        
        return f"""{self._get_system_prompt()}

CURRENT GOAL: {goal}

AVAILABLE TOOLS:
{self._format_tools_for_prompt()}
- finish(result): Complete the task and return the final result

CURRENT KNOWLEDGE/BELIEFS:
{context}

PREVIOUS REASONING STEPS:
{history if history else "None - this is the first step."}

Based on the goal and your observations so far, determine your next step.

INSTRUCTIONS:
1. Think step-by-step about what you know and what you need to find out
2. Choose the most appropriate action to take
3. If you have enough information to complete the goal, use the 'finish' action

You MUST respond with ONLY this JSON format (no other text):
{{
    "thought": "Your detailed reasoning about the current situation, what you know, what you need to find out, and why you're choosing this action",
    "action": "tool_name",
    "action_input": {{"param1": "value1", "param2": "value2"}}
}}

IMPORTANT: 
- The "thought" field should contain Chain-of-Thought reasoning
- Be specific and detailed in your reasoning
- If using 'finish', action_input should contain {{"result": "your final answer"}}"""

    def _execute_tool(self, action_name: str, action_input: Dict) -> str:
        """Execute a tool and return the observation"""
        
        if action_name == "finish":
            return f"TASK COMPLETE: {action_input.get('result', 'No result provided')}"
        
        if action_name not in self.tools:
            return f"ERROR: Unknown tool '{action_name}'. Available tools: {list(self.tools.keys())}"
        
        tool = self.tools[action_name]
        
        try:
            if tool.function:
                result = tool.function(**action_input)
                return str(result)
            else:
                return f"ERROR: Tool '{action_name}' has no implementation"
        except Exception as e:
            return f"ERROR executing {action_name}: {str(e)}"
    
    def run(self, goal: str) -> Dict[str, Any]:
        """
        Execute the ReAct loop to achieve a goal.
        
        Returns:
            Dict containing:
                - success: bool
                - result: The final result
                - reasoning_trace: List of all reasoning steps
                - iterations: Number of iterations taken
        """
        
        # Register tools if not already done
        if not self.tools:
            self.tools = self._register_tools()
        
        # Initialize desire (goal)
        self.state.add_desire(goal)
        
        previous_steps: List[ReActStep] = []
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[{self.agent_name}] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Create ReAct prompt
            prompt = self._create_react_prompt(goal, previous_steps)
            
            try:
                # Get LLM response
                response = self.llm.invoke(prompt)
                parsed = json.loads(response)
                
                thought = parsed.get("thought", "No thought provided")
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})
                
                if self.verbose:
                    print(f"  Thought: {thought[:100]}...")
                    print(f"  Action: {action}")
                
                # Execute the action
                observation = self._execute_tool(action, action_input)
                
                if self.verbose:
                    print(f"  Observation: {observation[:100]}...")
                
                # Record this step
                step = ReActStep(
                    step_number=iteration + 1,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation
                )
                previous_steps.append(step)
                self.state.reasoning_trace.append(step)
                
                # Update beliefs based on observation
                self.state.add_belief(
                    f"step_{iteration + 1}_result",
                    {"action": action, "observation": observation}
                )
                
                # Check if task is complete
                if action == "finish":
                    return {
                        "success": True,
                        "result": action_input.get("result"),
                        "reasoning_trace": previous_steps,
                        "iterations": iteration + 1
                    }
                    
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"  ERROR: Failed to parse LLM response: {e}")
                
                # Record error step
                step = ReActStep(
                    step_number=iteration + 1,
                    thought="Failed to parse response",
                    action="error",
                    action_input={},
                    observation=f"JSON parse error: {str(e)}"
                )
                previous_steps.append(step)
                
            except Exception as e:
                if self.verbose:
                    print(f"  ERROR: {e}")
                
                step = ReActStep(
                    step_number=iteration + 1,
                    thought="Unexpected error occurred",
                    action="error",
                    action_input={},
                    observation=str(e)
                )
                previous_steps.append(step)
        
        # Max iterations reached without finishing
        return {
            "success": False,
            "result": None,
            "reasoning_trace": previous_steps,
            "iterations": self.max_iterations,
            "error": "Max iterations reached without completing goal"
        }
    
    def log_message(self, to_agent: str, content: str, msg_type: str = "info"):
        """Log inter-agent communication for metrics tracking"""
        message = {
            "from": self.agent_name,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now().isoformat()
        }
        self.message_log.append(message)
        return message
    
    def get_reasoning_summary(self) -> str:
        """Get a human-readable summary of the agent's reasoning"""
        if not self.state.reasoning_trace:
            return "No reasoning steps recorded."
        
        summary_parts = []
        for step in self.state.reasoning_trace:
            summary_parts.append(f"""
**Step {step.step_number}**
- Thought: {step.thought}
- Action: {step.action}({json.dumps(step.action_input)})
- Observation: {step.observation}
""")
        
        return "\n".join(summary_parts)
    
    def reset_state(self):
        """Reset agent state for a new task"""
        self.state = AgentState()
        self.message_log = []
