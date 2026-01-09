# backend/agents/base_agent.py
"""
Base Agent with ReAct (Reasoning + Acting) Pattern
Based on: ReAct (Yao et al., 2023), Chain-of-Thought (Wei et al., 2022), BDI (Rao & Georgeff, 1995)
"""

from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json


@dataclass
class AgentBelief:
    """Agent's belief about the world (BDI model)"""
    key: str
    value: Any
    confidence: float = 1.0
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
    """BDI-inspired agent state: Beliefs, Desires, Intentions"""
    beliefs: Dict[str, AgentBelief] = field(default_factory=dict)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    reasoning_trace: List[ReActStep] = field(default_factory=list)
    
    def add_belief(self, key: str, value: Any, confidence: float = 1.0):
        self.beliefs[key] = AgentBelief(key=key, value=value, confidence=confidence)
    
    def get_belief(self, key: str, default: Any = None) -> Any:
        belief = self.beliefs.get(key)
        return belief.value if belief else default
    
    def add_desire(self, goal: str):
        if goal not in self.desires:
            self.desires.append(goal)
    
    def add_intention(self, plan: str):
        if plan not in self.intentions:
            self.intentions.append(plan)
    
    def get_context_summary(self) -> str:
        if not self.beliefs:
            return "No prior observations."
        return "\n".join(f"- {k}: {b.value}" for k, b in self.beliefs.items())


class BaseReActAgent(ABC):
    """Abstract base class implementing the ReAct pattern for all agents."""
    
    def __init__(
        self, 
        agent_name: str,
        agent_role: str,
        model_name: str = "llama3.1:8b",
        max_iterations: int = 5,
        min_iterations: int = 2,
        verbose: bool = True
    ):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.verbose = verbose
        self._last_action = None
        self._action_repeat_count = 0
        
        self.llm = OllamaLLM(model=model_name, temperature=0.0, format="json")
        self.state = AgentState()
        self.tools: Dict[str, AgentAction] = {}
        self.message_log: List[Dict] = []
    
    @abstractmethod
    def _register_tools(self) -> Dict[str, AgentAction]:
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass
    
    def _format_tools_for_prompt(self) -> str:
        return "\n".join(
            f"- {name}({', '.join(f'{k}: {v}' for k, v in a.parameters.items())}): {a.description}"
            for name, a in self.tools.items()
        )
    
    def _create_react_prompt(self, goal: str, previous_steps: List[ReActStep]) -> str:
        history = ""
        if previous_steps:
            history = "\n".join(
                f"Step {s.step_number}:\nThought: {s.thought}\nAction: {s.action}\n"
                f"Action Input: {json.dumps(s.action_input)}\nObservation: {s.observation}"
                for s in previous_steps
            )
        
        return f"""{self._get_system_prompt()}

CURRENT GOAL: {goal}

AVAILABLE TOOLS:
{self._format_tools_for_prompt()}
- finish(result): Complete the task and return the final result

CURRENT KNOWLEDGE: {self.state.get_context_summary()}

PREVIOUS STEPS:
{history if history else "None - first step."}

Respond with ONLY this JSON:
{{"thought": "reasoning about situation", "action": "tool_name", "action_input": {{"param": "value"}}}}

If using 'finish', action_input should be {{"result": "your final answer"}}"""

    def _execute_tool(self, action_name: str, action_input: Dict) -> str:
        if action_name == "finish":
            return f"TASK COMPLETE: {action_input.get('result', 'No result')}"
        
        if action_name not in self.tools:
            return f"ERROR: Unknown tool '{action_name}'. Available: {list(self.tools.keys())}"
        
        tool = self.tools[action_name]
        try:
            return str(tool.function(**action_input)) if tool.function else f"ERROR: No implementation for '{action_name}'"
        except Exception as e:
            return f"ERROR executing {action_name}: {e}"
    
    def run(self, goal: str) -> Dict[str, Any]:
        """Execute the ReAct loop to achieve a goal."""
        if not self.tools:
            self.tools = self._register_tools()
        
        self.state.add_desire(goal)
        previous_steps: List[ReActStep] = []
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[{self.agent_name}] Iteration {iteration + 1}/{self.max_iterations}")
            
            try:
                response = self.llm.invoke(self._create_react_prompt(goal, previous_steps))
                parsed = json.loads(response)
                
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})
                
                # Infer thought from action if not provided (LLM sometimes omits it when repeating)
                raw_thought = parsed.get("thought", "")
                if raw_thought and raw_thought.strip():
                    thought = raw_thought
                else:
                    # Generate a contextual thought based on the action being taken
                    thought = f"Continuing analysis with {action} to refine selection based on previous observations"
                
                if self.verbose:
                    print(f"  Thought: {thought[:100]}...")
                    print(f"  Action: {action}")
                
                observation = self._execute_tool(action, action_input)
                
                if self.verbose:
                    print(f"  Observation: {observation[:100]}...")
                
                step = ReActStep(
                    step_number=iteration + 1, thought=thought, action=action,
                    action_input=action_input, observation=observation
                )
                previous_steps.append(step)
                self.state.reasoning_trace.append(step)
                self.state.add_belief(f"step_{iteration + 1}_result", {"action": action, "observation": observation})
                
                # Check completion conditions
                if action == "finish":
                    if self.verbose:
                        print(f"  ✓ Task completed at iteration {iteration + 1}")
                    return {"success": True, "result": action_input.get("result"), 
                            "reasoning_trace": previous_steps, "iterations": iteration + 1, 
                            "early_stop_reason": "finish_action"}
                
                # Check for repeated actions
                if action == self._last_action:
                    self._action_repeat_count += 1
                else:
                    self._action_repeat_count = 0
                    self._last_action = action
                
                if self._action_repeat_count >= 2 and iteration + 1 >= self.min_iterations:
                    if self.verbose:
                        print(f"  ✓ Early stop: repeated action '{action}'")
                    return {"success": True, "result": self._extract_best_result_from_state(),
                            "reasoning_trace": previous_steps, "iterations": iteration + 1,
                            "early_stop_reason": "repeated_action"}
                
                if iteration + 1 >= self.min_iterations and self._should_stop_early(observation):
                    if self.verbose:
                        print(f"  ✓ Early stop: task completed")
                    return {"success": True, "result": self._extract_best_result_from_state(),
                            "reasoning_trace": previous_steps, "iterations": iteration + 1,
                            "early_stop_reason": "task_complete"}
                    
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"  ERROR: Failed to parse LLM response: {e}")
                previous_steps.append(ReActStep(
                    step_number=iteration + 1, thought="Failed to parse", action="error",
                    action_input={}, observation=f"JSON parse error: {e}"
                ))
            except Exception as e:
                if self.verbose:
                    print(f"  ERROR: {e}")
                previous_steps.append(ReActStep(
                    step_number=iteration + 1, thought="Unexpected error", action="error",
                    action_input={}, observation=str(e)
                ))
        
        return {"success": False, "result": None, "reasoning_trace": previous_steps,
                "iterations": self.max_iterations, "error": "Max iterations reached"}
    
    def log_message(self, to_agent: str, content: str, msg_type: str = "info"):
        message = {"from": self.agent_name, "to": to_agent, "content": content,
                   "type": msg_type, "timestamp": datetime.now().isoformat()}
        self.message_log.append(message)
        return message
    
    def get_reasoning_summary(self) -> str:
        if not self.state.reasoning_trace:
            return "No reasoning steps recorded."
        return "\n".join(
            f"**Step {s.step_number}**\n- Thought: {s.thought}\n- Action: {s.action}({json.dumps(s.action_input)})\n- Observation: {s.observation}"
            for s in self.state.reasoning_trace
        )
    
    def reset_state(self):
        self.state = AgentState()
        self.message_log = []
        self._last_action = None
        self._action_repeat_count = 0
    
    def _should_stop_early(self, observation: str) -> bool:
        return False
    
    def _extract_best_result_from_state(self) -> dict:
        return {"result": "Task completed based on gathered observations"}
