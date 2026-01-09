"""
Enhanced State Management for Agentic Trip Planning System

This module implements a comprehensive state management system following
the BDI (Belief-Desire-Intention) architecture pattern. The state tracks:
- Agent beliefs about the world (current options, constraints)
- System metrics for performance analysis
- Complete message history for auditability
- Reasoning traces from ReAct agents

Author: Research Project
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    ORCHESTRATOR = "orchestrator"
    FLIGHT_AGENT = "flight_agent"
    HOTEL_AGENT = "hotel_agent"
    POLICY_AGENT = "policy_agent"
    TIME_AGENT = "time_agent"


class MessageType(Enum):
    """FIPA-ACL inspired message types for agent communication."""
    REQUEST = "request"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    CONFIRM = "confirm"
    FAILURE = "failure"


@dataclass
class AgentMessage:
    """
    Structured message for inter-agent communication.
    Follows FIPA-ACL message structure for standardized agent communication.
    """
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None


@dataclass
class ReasoningStep:
    """
    A single step in an agent's reasoning process.
    Captures the ReAct pattern: Thought -> Action -> Observation
    """
    agent: AgentRole
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class AgentMetrics:
    """
    Performance metrics for a single agent.
    Used for research analysis and system optimization.
    """
    agent_role: AgentRole
    total_reasoning_steps: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    average_response_time_ms: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0


@dataclass
class SystemMetrics:
    """
    System-wide performance metrics for research analysis.
    Tracks efficiency and reasoning patterns.
    """
    total_workflow_time_ms: float = 0.0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    negotiation_rounds: int = 0
    budget_issues_found: int = 0
    budget_issues_resolved: int = 0
    backtracking_count: int = 0


def merge_messages(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge message lists, appending new messages."""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


def merge_reasoning_traces(existing: Dict, new: Dict) -> Dict:
    """Merge reasoning traces from multiple agents."""
    if existing is None:
        existing = {}
    if new is None:
        new = {}
    
    merged = existing.copy()
    for agent, traces in new.items():
        if agent in merged:
            merged[agent] = merged[agent] + traces
        else:
            merged[agent] = traces
    return merged


class TripPlanningState(TypedDict):
    """
    Enhanced state for the multi-agent trip planning workflow.
    
    This state follows the BDI architecture pattern:
    - Beliefs: Current knowledge about flights, hotels, policies
    - Desires: User requirements and constraints
    - Intentions: Selected options and planned actions
    
    The state also tracks:
    - Complete message history for agent communication
    - Reasoning traces from each agent's ReAct process
    - System metrics for research analysis
    """
    
    # ===== User Input (Desires) =====
    user_request: str  # Natural language trip request
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str]
    budget: Optional[float]
    preferences: Dict[str, Any]  # Additional user preferences
    
    # ===== Agent Beliefs =====
    # Flight Agent beliefs
    available_flights: List[Dict[str, Any]]
    flight_analysis: Optional[Dict[str, Any]]  # Agent's analysis of flight options
    
    # Hotel Agent beliefs  
    available_hotels: List[Dict[str, Any]]
    hotel_analysis: Optional[Dict[str, Any]]  # Agent's analysis of hotel options
    
    # Budget constraints
    budget_constraints: Dict[str, Any]  # Dynamic budget allocation
    compliance_status: Dict[str, Any]  # Budget compliance assessment
    
    # Time Agent beliefs
    time_constraints: Dict[str, Any]
    feasibility_analysis: Optional[Dict[str, Any]]  # Timeline feasibility assessment
    
    # ===== Current Intentions =====
    selected_flight: Optional[Dict[str, Any]]
    selected_hotel: Optional[Dict[str, Any]]
    
    # ===== Workflow Control =====
    current_phase: str  # Current phase of the workflow
    requires_human_input: bool  # Whether human intervention is needed
    workflow_complete: bool
    
    # ===== Communication & Reasoning =====
    # Message history with reducer for accumulation
    messages: Annotated[List[Dict[str, Any]], merge_messages]
    
    # Reasoning traces from each agent's ReAct process
    reasoning_traces: Annotated[Dict[str, List[Dict]], merge_reasoning_traces]
    
    # ===== System Metrics =====
    metrics: Dict[str, Any]
    
    # ===== Error Handling =====
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    
    # ===== Final Output =====
    final_recommendation: Optional[Dict[str, Any]]
    explanation: str  # Natural language explanation of the recommendation
    cheaper_alternatives: Optional[List[Dict[str, Any]]]  # Budget-conscious alternatives


def create_initial_state(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    budget: Optional[float] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> TripPlanningState:
    """
    Create an initial state for a new trip planning workflow.
    
    Accepts structured input from frontend (no free-text user_request).
    The system will use agent reasoning to find the optimal combination
    that MAXIMIZES budget usage.
    
    Args:
        origin: Departure city/airport code
        destination: Arrival city/airport code
        departure_date: Date of departure (YYYY-MM-DD)
        return_date: Optional return date
        budget: Total budget to maximize usage of
        preferences: Structured preferences from frontend
        
    Returns:
        Initialized TripPlanningState ready for workflow execution
    """
    return TripPlanningState(
        # User input (structured, not free text)
        user_request="",  # Deprecated - kept for backwards compatibility
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        preferences=preferences or {},
        
        # Agent beliefs - initially empty
        available_flights=[],
        flight_analysis=None,
        available_hotels=[],
        hotel_analysis=None,
        budget_constraints={},
        compliance_status={},
        time_constraints={},
        feasibility_analysis=None,
        
        # Intentions - initially none
        selected_flight=None,
        selected_hotel=None,
        
        # Workflow control
        current_phase="initialization",
        requires_human_input=False,
        workflow_complete=False,
        
        # Communication & reasoning
        messages=[],
        reasoning_traces={},
        
        # Metrics
        metrics={
            "workflow_start_time": datetime.now().isoformat(),
            "total_llm_calls": 0,
            "total_tokens_used": 0,
            "agent_metrics": {},
            "negotiation_rounds": 0,
            "backtracking_count": 0
        },
        
        # Error handling
        errors=[],
        warnings=[],
        
        # Output
        final_recommendation=None,
        explanation=""
    )


def add_message_to_state(
    state: TripPlanningState,
    sender: AgentRole,
    receiver: AgentRole,
    message_type: MessageType,
    content: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Create a message update for the state.
    
    Returns a dict that can be used in a state update to add the message.
    """
    message = {
        "sender": sender.value,
        "receiver": receiver.value,
        "type": message_type.value,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    return {"messages": [message]}


def add_reasoning_trace(
    state: TripPlanningState,
    agent: AgentRole,
    thought: str,
    action: Optional[str] = None,
    action_input: Optional[Dict[str, Any]] = None,
    observation: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Create a reasoning trace update for the state.
    
    Returns a dict that can be used in a state update to add the trace.
    """
    trace = {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation,
        "timestamp": datetime.now().isoformat()
    }
    return {"reasoning_traces": {agent.value: [trace]}}


def update_metrics(
    state: TripPlanningState,
    agent: AgentRole,
    llm_calls: int = 0,
    tokens_used: int = 0,
    tool_calls: int = 0,
    successful_tools: int = 0
) -> Dict[str, Any]:
    """
    Create a metrics update for the state.
    
    Returns updated metrics dict.
    """
    metrics = state.get("metrics", {}).copy()
    
    # Update global metrics
    metrics["total_llm_calls"] = metrics.get("total_llm_calls", 0) + llm_calls
    metrics["total_tokens_used"] = metrics.get("total_tokens_used", 0) + tokens_used
    
    # Update agent-specific metrics
    agent_metrics = metrics.get("agent_metrics", {})
    agent_key = agent.value
    
    if agent_key not in agent_metrics:
        agent_metrics[agent_key] = {
            "llm_calls": 0,
            "tokens_used": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0
        }
    
    agent_metrics[agent_key]["llm_calls"] += llm_calls
    agent_metrics[agent_key]["tokens_used"] += tokens_used
    agent_metrics[agent_key]["tool_calls"] += tool_calls
    agent_metrics[agent_key]["successful_tool_calls"] += successful_tools
    
    metrics["agent_metrics"] = agent_metrics
    
    return {"metrics": metrics}
