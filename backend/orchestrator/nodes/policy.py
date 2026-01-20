"""
Policy Check Node - PolicyAgent validates combinations, Orchestrator coordinates.

This node handles policy compliance by:
1. PolicyAgent selects optimal combination
2. PolicyAgent validates that combination
3. Orchestrator decides what to do with validation result
"""

from typing import Dict, Any
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, calculate_nights
from orchestrator.agents_config import policy_agent, orchestrator


def check_policy_node(state: TripPlanningState) -> Dict[str, Any]:
    """PolicyAgent validates combinations, Orchestrator coordinates routing."""
    print("\n" + "-"*60)
    print("📋 POLICY CHECK - Validation & Coordination")
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

    # Step 1: PolicyAgent selects optimal combo
    combination_result = policy_agent.find_best_combination(
        flights=flights, hotels=hotels, budget=budget, nights=nights, preferences=preferences
    )

    metrics["combinations_evaluated"] = combination_result.combinations_evaluated

    if not combination_result.success:
        print(f"\n  ❌ NO COMBINATIONS AVAILABLE")
        return {
            "compliance_status": {
                "overall_status": "non_compliant",
                "is_compliant": False,
                "violations": [{"type": "no_options", "message": combination_result.reasoning}],
                "total_cost": 0,
                "budget": budget
            },
            "current_phase": "finalize",
            "metrics": metrics
        }

    selected_flight = combination_result.selected_flight
    selected_hotel = combination_result.selected_hotel

    print(f"\n  ✅ SELECTED COMBINATION:")
    print(f"     Flight: {selected_flight.get('airline', 'Unknown')} - ${selected_flight.get('price_usd', 0)}")
    print(f"     Hotel: {selected_hotel.get('name', 'Unknown')} ({selected_hotel.get('stars', '?')}★)")

    # Step 2: PolicyAgent validates that combo
    validation = policy_agent.validate_combination(
        flight=selected_flight,
        hotel=selected_hotel,
        budget=budget,
        nights=nights
    )

    print(f"     Total: ${validation['total_cost']}")
    print(f"     Remaining: ${validation['budget_remaining']}")
    print(f"     Valid: {validation['is_valid']}")

    if validation['violations']:
        print(f"     Violations: {len(validation['violations'])}")

    # Step 3: Orchestrator decides what to do
    # Pass flight/hotel directly since state hasn't been updated yet
    decision = orchestrator.handle_policy_result(validation, state, selected_flight, selected_hotel)

    print(f"\n  🎯 ORCHESTRATOR DECISION: {decision['action']}")
    print(f"     {decision['reasoning']}")

    # Build compliance status
    compliance_status = {
        "overall_status": "compliant" if validation['is_valid'] else "non_compliant",
        "is_valid": validation['is_valid'],
        "violations": validation['violations'],
        "total_cost": validation['total_cost'],
        "budget": budget,
        "budget_remaining": validation['budget_remaining'],
        "reasoning": combination_result.reasoning,
        "combinations_evaluated": combination_result.combinations_evaluated,
        "cheaper_alternatives": combination_result.cheaper_alternatives
    }

    # Create message for tracking
    messages = [create_cnp_message(
        performative="inform",
        sender=AgentRole.POLICY_AGENT.value,
        receiver="orchestrator",
        content={
            "policy_check_complete": True,
            "is_valid": validation['is_valid'],
            "total_cost": validation['total_cost'],
            "budget_remaining": validation['budget_remaining']
        }
    )]

    return {
        "selected_flight": selected_flight,
        "selected_hotel": selected_hotel,
        "cheaper_alternatives": combination_result.cheaper_alternatives,
        "compliance_status": compliance_status,
        "policy_decision": decision,  # Store LLM's decision for routing
        "current_phase": decision['next_node'],  # Orchestrator decides routing
        "messages": messages,
        "metrics": metrics,
        "previous_total_cost": validation['total_cost']  # For stagnation detection
    }
