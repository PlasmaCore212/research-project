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
import sys
import threading
import time


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
        last_actions = []
        if previous_steps:
            history_parts = []
            for s in previous_steps:
                history_parts.append(
                    f"Step {s.step_number}:\nThought: {s.thought}\nAction: {s.action}\n"
                    f"Action Input: {json.dumps(s.action_input)}\nObservation: {s.observation}"
                )
                last_actions.append(s.action)
            history = "\n".join(history_parts)
        
        # Build "don't repeat" instruction if there are previous actions
        repeat_warning = ""
        if last_actions:
            repeat_warning = f"""
⚠️ DO NOT REPEAT ACTIONS: You already used: {', '.join(last_actions)}
   → Use DIFFERENT actions to make progress!
   → After searching, use compare_hotels/compare_flights or finish
   → After analyzing, use compare or finish with your recommendations"""

        return f"""{self._get_system_prompt()}

CURRENT GOAL: {goal}

AVAILABLE TOOLS (you MUST use ONLY these tools, no others):
{self._format_tools_for_prompt()}
- finish(result): Complete the task and return the final result

CURRENT KNOWLEDGE (use these values when tools need parameters):
{self.state.get_context_summary()}

PREVIOUS STEPS:
{history if history else "None - first step."}
{repeat_warning}

⭐ PROGRESSION PATH (follow this sequence):
1. FIRST: Use search_flights/search_hotels to find options
2. THEN: Use analyze_options or compare to evaluate candidates
3. FINALLY: Use finish to return your TOP recommendations

IMPORTANT RULES:
1. ONLY use tools from the AVAILABLE TOOLS list above
2. ALL required parameters must be provided - check CURRENT KNOWLEDGE for stored values
3. If a tool needs from_city/to_city/city, use values from CURRENT KNOWLEDGE
4. DO NOT call the same action twice - move to the NEXT step!

Respond with ONLY this JSON:
{{"thought": "What I learned and what I should do NEXT (not repeat)", "action": "tool_name", "action_input": {{"param": "value"}}}}

If using 'finish', action_input should be {{"result": "your final answer with TOP recommendations"}}"""

    def _execute_tool(self, action_name: str, action_input: Dict) -> str:
        """Execute a tool and return observation. Always returns a non-empty string."""
        if not action_name:
            return "ERROR: No action specified"

        if action_name == "finish":
            result = action_input.get('result', 'No result provided')
            return f"TASK COMPLETE: {result}"

        if action_name not in self.tools:
            return (f"ERROR: Unknown tool '{action_name}'. You MUST use ONLY these available tools: "
                   f"{list(self.tools.keys())}. Check the AVAILABLE TOOLS list and try again.")

        tool = self.tools[action_name]
        try:
            if not tool.function:
                return f"ERROR: No implementation for '{action_name}'"
            result = tool.function(**action_input)
            if result is None:
                return f"Tool '{action_name}' completed with no output"
            return str(result)
        except TypeError as e:
            # Handle missing or wrong parameters - provide helpful guidance
            error_msg = str(e)
            if "missing" in error_msg and "required positional argument" in error_msg:
                return (f"ERROR: Invalid parameters for {action_name}: {e}\n"
                       f"Expected parameters: {tool.parameters}\n"
                       f"Check CURRENT KNOWLEDGE for required values (e.g., last_search_from, last_search_to, search_city)")
            return f"ERROR: Invalid parameters for {action_name}: {e}. Expected: {tool.parameters}"
        except Exception as e:
            return f"ERROR executing {action_name}: {e}"
    
    def _show_progress(self, stop_event: threading.Event, agent_name: str, iteration: int):
        """Show real-time progress indicator during LLM call."""
        symbols = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        start_time = time.time()
        i = 0
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r    {symbols[i % len(symbols)]} [{agent_name}] Thinking... ({elapsed:.0f}s)    ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear line
        sys.stdout.flush()
    
    def run(self, goal: str) -> Dict[str, Any]:
        """Execute the ReAct loop to achieve a goal."""
        if not self.tools:
            self.tools = self._register_tools()
        
        self.state.add_desire(goal)
        previous_steps: List[ReActStep] = []
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if self.verbose:
                print(f"\n[{self.agent_name}] Iteration {iteration + 1}/{self.max_iterations} @ {timestamp}")
            
            try:
                # Start progress indicator in background thread
                stop_event = threading.Event()
                if self.verbose:
                    progress_thread = threading.Thread(
                        target=self._show_progress, 
                        args=(stop_event, self.agent_name, iteration + 1)
                    )
                    progress_thread.start()
                
                try:
                    response = self.llm.invoke(self._create_react_prompt(goal, previous_steps))
                finally:
                    stop_event.set()
                    if self.verbose:
                        progress_thread.join(timeout=1)
                
                llm_time = time.time() - iter_start
                if self.verbose:
                    print(f"    ⏱️  LLM response: {llm_time:.1f}s")
                
                parsed = json.loads(response)
                
                action = parsed.get("action", "")
                action_input = parsed.get("action_input", {})
                
                # Ensure action_input is a dict (LLM might return string or None)
                if action_input is None:
                    action_input = {}
                elif isinstance(action_input, str):
                    try:
                        action_input = json.loads(action_input)
                    except:
                        action_input = {"input": action_input}
                
                # Robust thought extraction - handle None, empty string, or missing
                raw_thought = parsed.get("thought")
                if raw_thought and isinstance(raw_thought, str) and raw_thought.strip():
                    thought = raw_thought.strip()
                else:
                    # Generate a contextual thought based on the action being taken
                    if action:
                        thought = f"Proceeding with {action} to gather more information and refine the analysis"
                    else:
                        thought = "Analyzing the current state to determine next steps"
                
                if self.verbose:
                    # Safe truncation for thought display
                    display_thought = thought[:100] if thought else "(no thought)"
                    print(f"  Thought: {display_thought}...")
                    print(f"  Action: {action if action else '(no action)'}")
                
                observation = self._execute_tool(action, action_input)
                
                # Ensure observation is never None and is a string
                if observation is None:
                    observation = "No observation returned from tool"
                observation = str(observation)
                
                if self.verbose:
                    # Safe truncation for observation display
                    display_obs = observation[:100] if observation else "(no observation)"
                    print(f"  Observation: {display_obs}...")
                
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
