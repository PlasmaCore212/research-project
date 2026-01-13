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
        model_name: str = "qwen2.5:14b",
        max_iterations: int = 10,
        min_iterations: int = 2,
        verbose: bool = True
    ):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.model_name = model_name
        self.max_iterations = max_iterations  # Safety cap only - LLM decides when to stop
        self.min_iterations = min_iterations
        self.verbose = verbose
        self._last_action = None
        self._action_repeat_count = 0
        
        # Temperature 0.3 for diverse but coherent reasoning
        self.llm = OllamaLLM(model=model_name, temperature=0.3, format="json")
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
        """Format tools with detailed parameter information for the LLM."""
        lines = []
        for name, action in self.tools.items():
            # Tool name and description
            lines.append(f"• {name}: {action.description}")
            # Parameters with clear formatting
            if action.parameters:
                params_str = []
                for param_name, param_desc in action.parameters.items():
                    # Mark required vs optional
                    if "REQUIRED" in param_desc.upper() or "optional" not in param_desc.lower():
                        params_str.append(f"    - {param_name}: {param_desc}")
                    else:
                        params_str.append(f"    - {param_name}: {param_desc}")
                lines.append("  Parameters:")
                lines.extend(params_str)
            else:
                lines.append("  Parameters: none")
            lines.append("")  # Blank line between tools
        return "\n".join(lines)
    
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
            repeat_warning = f"\n⚠️ Already used: {', '.join(last_actions)} - try different actions or finish."

        # Get actual tool names for this agent
        tool_names = list(self.tools.keys()) if self.tools else []
        tool_list_str = ", ".join(tool_names) if tool_names else "search, compare, finish"
        
        # Build available tools section prominently
        tools_section = f"""YOUR AVAILABLE TOOLS:
{self._format_tools_for_prompt()}
• finish: Complete your task and return final results
  Parameters:
    - result: dict with your findings

You can ONLY use these tools. Any other tool name will fail."""

        return f"""{self._get_system_prompt()}

{tools_section}

CURRENT GOAL: {goal}

CURRENT KNOWLEDGE:
{self.state.get_context_summary()}

PREVIOUS STEPS:
{history if history else "None - first step."}
{repeat_warning}

INSTRUCTIONS:
- Use your tools to search, analyze, and compare options
- You decide when you have enough information - use 'finish' when ready
- Aim for 3-5 diverse options across different price/quality tiers
- Think about what you've learned and what action would help most

RESPONSE FORMAT (JSON only):
{{
  "thought": "<Your reasoning>",
  "action": "<tool name from list above>",
  "action_input": {{<parameters as key-value pairs>}}
}}

EXAMPLES:
- Search: {{"thought": "...", "action": "search_flights", "action_input": {{"from_city": "NYC", "to_city": "SF"}}}}
- Compare: {{"thought": "...", "action": "compare_flights", "action_input": {{"flight_ids": ["FL001", "FL002"]}}}}
- Finish: {{"thought": "...", "action": "finish", "action_input": {{"result": {{"top_flights": [...], "reasoning": "..."}}}}}}"""

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
                # Extract the missing parameter name from error
                import re
                match = re.search(r"'(\w+)'", error_msg)
                missing_param = match.group(1) if match else "unknown"

                return (f"ERROR: Tool '{action_name}' is missing required parameter: '{missing_param}'\n"
                       f"Expected parameters: {tool.parameters}\n"
                       f"HINT: Check CURRENT KNOWLEDGE for available values. "
                       f"If comparing items, you need a LIST of IDs from previous search results.")
            return f"ERROR: Invalid parameters for {action_name}. Expected: {tool.parameters}\nYou provided: {action_input}"
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
                    # Show full thought and observation (increased from 100 to 300 chars)
                    display_thought = thought[:300] if thought else "(no thought)"
                    print(f"  Thought: {display_thought}{'...' if len(thought) > 300 else ''}")
                    print(f"  Action: {action if action else '(no action)'}")
                
                observation = self._execute_tool(action, action_input)
                
                # Ensure observation is never None and is a string
                if observation is None:
                    observation = "No observation returned from tool"
                observation = str(observation)
                
                if self.verbose:
                    # Show more of the observation (increased from 100 to 300 chars)
                    display_obs = observation[:300] if observation else "(no observation)"
                    print(f"  Observation: {display_obs}{'...' if len(observation) > 300 else ''}")
                
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
                
                # LLM-based stopping decision (agentic) - only way to stop early
                if iteration + 1 >= self.min_iterations:
                    should_stop, stop_reasoning = self._should_stop_early_llm(iteration + 1, previous_steps)
                    if should_stop:
                        if self.verbose:
                            print(f"  ✓ Early stop (LLM decision): {stop_reasoning}")
                        return {"success": True, "result": self._extract_best_result_from_state(),
                                "reasoning_trace": previous_steps, "iterations": iteration + 1,
                                "early_stop_reason": f"llm_decision: {stop_reasoning}"}
                
                # Subclass-specific stopping (for TimeAgent etc.)
                if self._should_stop_early(observation):
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
        """Override this to add custom early stopping logic (deprecated - use _should_stop_early_llm instead)."""
        return False

    def _should_stop_early_llm(self, iteration: int, previous_steps: List[ReActStep]) -> tuple[bool, str]:
        """
        Use LLM reasoning to decide if the agent has gathered enough information to complete the task.

        Returns:
            tuple[bool, str]: (should_stop, reasoning)
        """
        # Only use LLM decision after minimum iterations
        if iteration < self.min_iterations:
            return False, "Not enough iterations yet"

        # If no steps, don't stop
        if not previous_steps:
            return False, "No previous steps"

        # Build context of what has been done
        actions_taken = [s.action for s in previous_steps]
        observations = [s.observation for s in previous_steps]
        
        # Count unique actions
        unique_actions = set(actions_taken)
        has_searched = any('search' in a.lower() for a in actions_taken)
        has_analyzed = any('analyze' in a.lower() for a in actions_taken)
        has_compared = any('compare' in a.lower() for a in actions_taken)

        # Get current beliefs summary
        beliefs_summary = self.state.get_context_summary()
        
        # Check for obvious stopping conditions (no LLM needed)
        if has_searched and (has_analyzed or has_compared) and iteration >= 3:
            return True, f"Completed search and analysis after {iteration} iterations"
        
        # If we've done 4+ iterations and searched, just stop
        if has_searched and iteration >= 4:
            return True, f"Gathered enough options after {iteration} iterations"

        # Create LLM prompt for stopping decision
        prompt = f"""You are the {self.agent_name}. You have completed {iteration} iterations.

ACTIONS TAKEN: {', '.join(actions_taken)}

QUESTION: Have you found enough diverse options (3-5) to send to PolicyAgent?

SIMPLE RULE:
- If you've searched AND analyzed or compared: answer YES
- If you've done 3+ distinct actions: answer YES
- Otherwise: answer NO

Respond with JSON only:
{{"should_stop": true, "reasoning": "..."}} or {{"should_stop": false, "reasoning": "..."}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            should_stop = result.get("should_stop", False)
            reasoning = result.get("reasoning", "LLM decision")
            return should_stop, reasoning
        except Exception as e:
            # Fallback: stop if we've done enough
            if iteration >= 4:
                return True, f"Stopping after {iteration} iterations (fallback)"
            return False, f"LLM decision error: {e}"

    def _extract_best_result_from_state(self) -> dict:
        return {"result": "Task completed based on gathered observations"}
