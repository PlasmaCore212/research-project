"""
Routing Functions for the Trip Planning Workflow.

This module contains all conditional routing logic for the LangGraph workflow:
- should_route_after_policy: Route after policy check
- should_backtrack_after_time: Route after time check
- should_backtrack_after_time_feedback: Route after time feedback
- increment_backtrack_counter: Increment backtrack counter node
"""

from typing import Dict, Any, Literal
from orchestrator.state import TripPlanningState
from orchestrator.helpers import MAX_BACKTRACKING_ITERATIONS, MAX_NEGOTIATION_ROUNDS


def should_route_after_policy(state: TripPlanningState) -> Literal["check_time", "negotiation", "finalize"]:
    """
    Route after policy check based on budget and negotiation state.
    
    - Within budget with good utilization → check_time
    - Low budget utilization → negotiation (quality upgrade)
    - Over budget → negotiation (cost reduction)
    - Max rounds reached → check_time (best effort)
    """
    compliance = state.get("compliance_status", {})
    metrics = state.get("metrics", {})
    negotiation_rounds = metrics.get("negotiation_rounds", 0)
    negotiation_converged = metrics.get("negotiation_converged", False)
    
    budget_remaining = compliance.get("budget_remaining", 0)
    total_cost = compliance.get("total_cost", 0)
    budget = state.get("budget", 2000)
    
    budget_utilization = (total_cost / budget * 100) if budget > 0 else 100
    
    # PRIORITY 1: If negotiation has converged (stagnation detected), STOP negotiating
    if negotiation_converged:
        print(f"\n  ✅ Negotiation converged at ${total_cost:.0f} ({budget_utilization:.0f}% utilization)")
        return "check_time"
    
    # PRIORITY 2: Max rounds reached
    if negotiation_rounds >= MAX_NEGOTIATION_ROUNDS:
        print(f"\n  ⚠️  Max negotiation rounds ({MAX_NEGOTIATION_ROUNDS}) reached")
        if state.get("selected_flight") and state.get("selected_hotel"):
            return "check_time"
        return "finalize"
    
    # Case 1: Within budget
    if budget_remaining >= 0:
        # Check if we should trigger quality upgrade negotiation
        # Only continue if utilization < 75% (matching PolicyAgent threshold)
        if budget_utilization < 75 and negotiation_rounds < MAX_NEGOTIATION_ROUNDS:
            print(f"\n  💎 QUALITY UPGRADE: {budget_utilization:.0f}% utilization - seeking premium options (round {negotiation_rounds + 1})")
            return "negotiation"
        
        print(f"\n  ✅ Within budget: ${total_cost:.0f} / ${budget:.0f} ({budget_utilization:.0f}% utilization)")
        return "check_time"
    
    # Case 2: Over budget - start negotiation
    print(f"\n  💰 BUDGET EXCEEDED by ${abs(budget_remaining):.0f} - starting negotiation")
    return "negotiation"


def should_backtrack_after_time(state: TripPlanningState) -> Literal["select_options", "time_policy_feedback"]:
    """Route to policy feedback on time conflicts, otherwise proceed to selection."""
    time_constraints = state.get("time_constraints", {})
    metrics = state.get("metrics", {})
    
    is_feasible = time_constraints.get("feasible", True)
    conflicts = time_constraints.get("conflicts", [])
    time_feedback_count = metrics.get("time_feedback_count", 0)
    
    severe_conflicts = [c for c in conflicts if isinstance(c, dict) and c.get("severity") == "error"]
    
    if not is_feasible and severe_conflicts and time_feedback_count < MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  ⏰ TIME CONFLICT: Reporting {len(severe_conflicts)} issue(s) to PolicyAgent")
        return "time_policy_feedback"
    
    return "select_options"


def should_backtrack_after_time_feedback(state: TripPlanningState) -> Literal["increment_backtrack", "select_options"]:
    """Route after time-policy feedback."""
    metrics = state.get("metrics", {})
    backtrack_count = metrics.get("backtracking_count", 0)
    flight_alternatives = state.get("flight_alternatives", [])
    
    if flight_alternatives and backtrack_count < MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  🔄 Backtracking with alternatives (iteration {backtrack_count + 1})")
        return "increment_backtrack"
    
    if backtrack_count >= MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  ⚠️  Max backtracking iterations reached")
    
    return "select_options"


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
