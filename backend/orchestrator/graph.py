"""
LangGraph Workflow for Agentic Trip Planning System

This module implements a truly agentic workflow orchestration using LangGraph's StateGraph.
The workflow coordinates multiple ReAct-based agents following:
- Contract Net Protocol (Smith, 1980) for task delegation
- FIPA-ACL message patterns for agent communication
- BDI Architecture (Rao & Georgeff, 1995) for agent reasoning
- ReAct (Yao et al., 2023) for autonomous reasoning

KEY AGENTIC FEATURES:
1. PARALLEL EXECUTION: Flight and Hotel agents search simultaneously
2. REAL BACKTRACKING: Budget constraints trigger actual re-planning
3. INTER-AGENT MESSAGING: Agents communicate through structured messages
4. AUTONOMOUS REASONING: Each agent uses ReAct to decide which tools to use
5. ITERATIVE REFINEMENT: Maximum 10 backtracking iterations

MODULAR ARCHITECTURE:
- nodes/: Individual workflow node implementations
- helpers.py: Shared utility functions and constants
- agents_config.py: Agent instance configuration
- routing.py: Conditional routing logic

Author: Research Project - Laureys Indy
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from orchestrator.state import TripPlanningState, create_initial_state

# Import nodes from modular structure
from orchestrator.nodes import (
    initialize_node,
    parallel_search_node,
    negotiation_node,
    should_continue_negotiation,
    check_policy_node,
    check_time_node,
    time_policy_feedback_node,
    select_options_node,
    finalize_node,
)

# Import routing functions
from orchestrator.routing import (
    should_route_after_policy,
    should_backtrack_after_time,
    should_backtrack_after_time_feedback,
    increment_backtrack_counter,
)


def build_workflow() -> StateGraph:
    """
    Build LangGraph workflow with modular architecture.
    
    Flow:
    Initialize → Search → PolicyDecision → (Negotiation if needed) → Time → Select → Finalize
    
    KEY DESIGN:
    1. Agents return DIVERSE options across all price/quality tiers
    2. PolicyAgent FIRST tries to find valid combination
    3. Negotiation ONLY starts if no valid combination within budget
    4. PolicyAgent directs SPECIFIC agent(s) to adjust prices during negotiation
    """
    workflow = StateGraph(TripPlanningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("parallel_search", parallel_search_node)
    workflow.add_node("check_policy", check_policy_node)
    workflow.add_node("negotiation", negotiation_node)
    workflow.add_node("increment_backtrack", increment_backtrack_counter)
    workflow.add_node("check_time", check_time_node)
    workflow.add_node("time_policy_feedback", time_policy_feedback_node)
    workflow.add_node("select_options", select_options_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Define edges
    workflow.add_edge("initialize", "parallel_search")
    workflow.add_edge("parallel_search", "check_policy")
    
    # Policy routing: valid combo → time check, else → negotiation
    workflow.add_conditional_edges(
        "check_policy",
        should_route_after_policy,
        {"check_time": "check_time", "negotiation": "negotiation", "finalize": "finalize"}
    )
    
    # Negotiation loop
    workflow.add_conditional_edges(
        "negotiation",
        should_continue_negotiation,
        {"negotiation": "negotiation", "check_policy": "check_policy"}
    )
    
    workflow.add_edge("increment_backtrack", "parallel_search")
    
    # Time check routing
    workflow.add_conditional_edges(
        "check_time",
        should_backtrack_after_time,
        {"select_options": "select_options", "time_policy_feedback": "time_policy_feedback"}
    )
    
    # Time feedback routing
    workflow.add_conditional_edges(
        "time_policy_feedback",
        should_backtrack_after_time_feedback,
        {"increment_backtrack": "increment_backtrack", "select_options": "select_options"}
    )
    
    workflow.add_edge("select_options", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


def create_trip_planning_app():
    """Create and compile the trip planning application."""
    return build_workflow().compile()


def plan_trip(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = None,
    budget: float = None,
    hotel_checkin: str = None,
    hotel_checkout: str = None,
    meeting_time: str = None,
    meeting_date: str = None,
    meeting_coordinates: dict = None
) -> Dict[str, Any]:
    """
    Main entry point for multi-agent trip planning.
    
    Args:
        origin: Departure city code (e.g., 'NYC', 'SF')
        destination: Arrival city code
        departure_date: Date of departure (YYYY-MM-DD)
        return_date: Optional return date
        budget: Total budget (agents reason about how to use it wisely)
        hotel_checkin: Check-in date
        hotel_checkout: Check-out date
        meeting_time: Time of meeting (HH:MM)
        meeting_date: Date of meeting
        meeting_coordinates: Dict with 'lat' and 'lon' keys
    
    Returns:
        Final workflow state with recommendation
    """
    preferences = {
        "hotel_checkin": hotel_checkin or departure_date,
        "hotel_checkout": hotel_checkout or return_date,
    }
    
    if meeting_date and meeting_time:
        preferences["meeting_times"] = [f"{meeting_date} {meeting_time}"]
    
    if meeting_coordinates:
        preferences["meeting_location"] = meeting_coordinates
    
    if budget:
        preferences["adjusted_flight_budget"] = budget * 0.5
        preferences["adjusted_hotel_budget"] = budget * 0.5
    
    initial_state = create_initial_state(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        preferences=preferences
    )
    
    app = create_trip_planning_app()
    
    final_state = None
    for state in app.stream(initial_state):
        final_state = state
    
    return final_state


# Module can be imported but not run directly - use main.py or tests
