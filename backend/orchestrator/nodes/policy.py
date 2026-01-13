"""
Policy Check Node - PolicyAgent evaluates combinations and selects optimal.

This node handles policy compliance by:
1. Evaluating all flight+hotel combinations
2. Finding the optimal combination within budget
3. Generating alternatives for user choice
"""

from typing import Dict, Any
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, calculate_nights
from orchestrator.agents_config import policy_agent


def check_policy_node(state: TripPlanningState) -> Dict[str, Any]:
    """PolicyAgent evaluates all combinations and selects optimal within budget."""
    print("\n" + "-"*60)
    print("📋 POLICY AGENT - Finding Optimal Combination")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    budget = state.get("budget", 2000)
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    nights = calculate_nights(state)
    
    print(f"  Budget: ${budget}")
    print(f"  Nights: {nights}")
    print(f"  Options: {len(flights)} flights × {len(hotels)} hotels")
    
    # Use PolicyAgent to find the best combination
    combination_result = policy_agent.find_best_combination(
        flights=flights, hotels=hotels, budget=budget, nights=nights, preferences=preferences
    )
    
    metrics["combinations_evaluated"] = combination_result.combinations_evaluated
    
    if combination_result.success:
        selected_flight = combination_result.selected_flight
        selected_hotel = combination_result.selected_hotel
        cheaper_alternatives = combination_result.cheaper_alternatives
        
        print(f"\n  ✅ OPTIMAL COMBINATION FOUND:")
        print(f"     Flight: {selected_flight.get('airline', 'Unknown')} - ${selected_flight.get('price_usd', 0)}")
        print(f"     Hotel: {selected_hotel.get('name', 'Unknown')} ({selected_hotel.get('stars', '?')}★)")
        print(f"     Total: ${combination_result.total_cost} (${combination_result.budget_remaining} remaining)")
        
        if cheaper_alternatives:
            print(f"\n  🏨 ALTERNATIVES: {len(cheaper_alternatives)} options available")
        
        print(f"\n  💭 Reasoning: {combination_result.reasoning[:150]}...")
        
        messages = [create_cnp_message(
            performative="inform", sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={
                "policy_check_complete": True, "is_compliant": True,
                "selected_flight": selected_flight, "selected_hotel": selected_hotel,
                "total_cost": combination_result.total_cost,
                "budget_remaining": combination_result.budget_remaining
            }
        )]
        
        return {
            "selected_flight": selected_flight,
            "selected_hotel": selected_hotel,
            "cheaper_alternatives": cheaper_alternatives,
            "compliance_status": {
                "overall_status": "compliant",
                "is_compliant": True,
                "violations": [],
                "total_cost": combination_result.total_cost,
                "budget": budget,
                "budget_remaining": combination_result.budget_remaining,
                "reasoning": combination_result.reasoning,
                "combinations_evaluated": combination_result.combinations_evaluated,
                "cheaper_alternatives": cheaper_alternatives
            },
            "current_phase": "time_check",
            "messages": messages,
            "metrics": metrics
        }
    else:
        print(f"\n  ❌ NO VALID COMBINATIONS WITHIN BUDGET")
        print(f"     {combination_result.reasoning}")
        
        metrics["policy_feedback_loops"] = metrics.get("policy_feedback_loops", 0) + 1
        
        messages = [create_cnp_message(
            performative="inform", sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={"policy_check_complete": True, "is_compliant": False, "reason": combination_result.reasoning}
        )]
        
        return {
            "compliance_status": {
                "overall_status": "non_compliant",
                "is_compliant": False,
                "violations": [{"type": "no_valid_combination", "message": combination_result.reasoning, "severity": "error"}],
                "total_cost": 0, "budget": budget,
                "reasoning": combination_result.reasoning,
                "combinations_evaluated": combination_result.combinations_evaluated
            },
            "current_phase": "time_check",
            "messages": messages,
            "metrics": metrics
        }
