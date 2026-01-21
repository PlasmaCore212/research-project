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
    # Check if negotiation already ran and accepted - if so, skip re-negotiating
    negotiation_feedback = state.get("negotiation_feedback", {})
    negotiation_accepted = negotiation_feedback.get("should_accept", False) or negotiation_feedback.get("at_market_max", False)
    negotiation_rounds = metrics.get("negotiation_rounds", 0)

    if negotiation_accepted and negotiation_rounds > 0:
        # Negotiation already finished and accepted - don't re-negotiate
        print(f"\n  ✅ NEGOTIATION ALREADY ACCEPTED (round {negotiation_rounds}) - Proceeding to time check")
        decision = {"action": "accept", "reasoning": "Negotiation already accepted this booking.", "next_node": "check_time"}
    else:
        # Pass flight/hotel directly since state hasn't been updated yet
        decision = orchestrator.handle_policy_result(validation, state, selected_flight, selected_hotel)

    print(f"\n  🎯 ORCHESTRATOR DECISION: {decision['action']}")
    print(f"     {decision['reasoning']}")

    # If orchestrator says to use best option (max rounds reached), restore it
    if decision.get("use_best_option"):
        best_option = state.get("best_option_seen", {})
        if best_option:
            selected_flight = best_option.get("flight")
            selected_hotel = best_option.get("hotel")
            print(f"\n  🏆 USING BEST OPTION FROM ROUND {best_option.get('round', 0)}")
            print(f"     Flight: {selected_flight.get('flight_id', 'N/A')}, Hotel: {selected_hotel.get('hotel_id', 'N/A')}")
            print(f"     Total: ${best_option.get('total_cost', 0)} ({best_option.get('utilization', 0):.1f}% utilization)")

            # Re-validate the best option for compliance status
            validation = policy_agent.validate_combination(
                flight=selected_flight,
                hotel=selected_hotel,
                budget=budget,
                nights=nights
            )
            # Build compliance status for best option
            compliance_status = {
                "overall_status": "compliant" if validation['is_valid'] else "non_compliant",
                "is_valid": validation['is_valid'],
                "violations": validation['violations'],
                "total_cost": validation['total_cost'],
                "budget": budget,
                "budget_remaining": validation['budget_remaining'],
                "reasoning": f"Best option from round {best_option.get('round', 0)}",
                "combinations_evaluated": 1,
                "cheaper_alternatives": []
            }
    else:
        # Build compliance status for current selection
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

    # BEST OPTION TRACKING: Track best combination across all negotiation rounds
    # Calculate budget utilization (higher is better if within budget)
    utilization = (validation['total_cost'] / budget * 100) if budget > 0 else 0

    # Get previous best (if any)
    best_option = state.get("best_option_seen", {})
    best_utilization = best_option.get("utilization", 0)
    best_is_valid = best_option.get("is_valid", False)

    # Determine if current option is better than previous best
    # Priority: 1) Valid combinations first, 2) Higher utilization (closer to budget)
    is_better = (
        (validation['is_valid'] and not best_is_valid) or  # First valid option
        (validation['is_valid'] == best_is_valid and utilization > best_utilization)  # Same validity, better utilization
    )

    # Update best option if current is better
    if is_better or not best_option:
        best_option = {
            "flight": selected_flight.copy() if isinstance(selected_flight, dict) else selected_flight,
            "hotel": selected_hotel.copy() if isinstance(selected_hotel, dict) else selected_hotel,
            "total_cost": validation['total_cost'],
            "utilization": utilization,
            "is_valid": validation['is_valid'],
            "round": metrics.get("negotiation_rounds", 0),
            "violations": validation['violations']
        }
        if validation['is_valid']:
            print(f"  🏆 NEW BEST VALID: ${validation['total_cost']} ({utilization:.1f}% utilization)")
        else:
            print(f"  📊 NEW BEST (invalid): ${validation['total_cost']} ({utilization:.1f}% utilization)")

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
        "previous_total_cost": validation['total_cost'],  # For stagnation detection
        "best_option_seen": best_option  # Track best across all rounds
    }
