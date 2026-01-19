"""
Routing Functions for the Trip Planning Workflow.

This module contains all conditional routing logic for the LangGraph workflow.
All routing decisions are delegated to the Orchestrator.
"""

from typing import Dict, Any, Literal
from orchestrator.state import TripPlanningState
from orchestrator.helpers import MAX_BACKTRACKING_ITERATIONS, MAX_NEGOTIATION_ROUNDS
from orchestrator.agents_config import orchestrator


def should_route_after_policy(state: TripPlanningState) -> Literal["check_time", "negotiation", "finalize"]:
    """Route after policy check - orchestrator decides."""
    decision = orchestrator.decide_next_node("after_policy", state)
    policy_dec = state.get("policy_decision", {})
    print(f"  [ROUTING] should_route_after_policy → {decision} (LLM said: {policy_dec.get('action', 'N/A')})")
    return decision


def should_backtrack_after_time(state: TripPlanningState) -> Literal["select_options", "time_policy_feedback"]:
    """Route after time check - orchestrator decides."""
    return orchestrator.decide_next_node("after_time", state)


def should_backtrack_after_time_feedback(state: TripPlanningState) -> Literal["increment_backtrack", "select_options"]:
    """Route after time feedback - orchestrator decides."""
    return orchestrator.decide_next_node("after_time_feedback", state)


def increment_backtrack_counter(state: TripPlanningState) -> Dict[str, Any]:
    """Increment backtrack counter and prepare for re-search."""
    metrics = state.get("metrics", {}).copy()
    metrics["backtracking_count"] = metrics.get("backtracking_count", 0) + 1

    print(f"\n  📊 Backtracking iteration: {metrics['backtracking_count']}/{MAX_BACKTRACKING_ITERATIONS}")

    flight_alternatives = state.get("flight_alternatives", [])
    if flight_alternatives:
        print(f"  ✈️  Using {len(flight_alternatives)} alternative flights")
        return {
            "metrics": metrics,
            "available_flights": flight_alternatives,
            "flight_alternatives": []
        }

    return {"metrics": metrics}
