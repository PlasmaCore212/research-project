"""
Selection and Finalization Nodes.

This module contains:
1. select_options_node: Orchestrator confirms PolicyAgent selection
2. finalize_node: Generate final trip recommendation
"""

from typing import Dict, Any
from datetime import datetime
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, calculate_nights, dict_to_flight, dict_to_hotel
from orchestrator.agents_config import orchestrator


def select_options_node(state: TripPlanningState) -> Dict[str, Any]:
    """Orchestrator makes final selection using Chain-of-Thought reasoning."""
    print("\n" + "-"*60)
    print("🎯 ORCHESTRATOR - Chain-of-Thought Selection")
    print("-"*60)

    # ALWAYS use orchestrator for selection
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    time_constraints = state.get("time_constraints", {})
    budget = state.get("budget", 2000)
    
    if not flights or not hotels:
        print("  ⚠️ No options to select from")
        return {"selected_flight": None, "selected_hotel": None, "current_phase": "finalizing", "messages": []}
    
    # Convert to models
    flight_models = []
    for f in flights[:5]:
        try:
            flight_models.append(dict_to_flight(f, origin, destination))
        except:
            pass
    
    hotel_models = []
    for h in hotels[:5]:
        try:
            hotel_models.append(dict_to_hotel(h, destination))
        except:
            pass
    
    context = {
        "flight_options": flight_models, "hotel_options": hotel_models,
        "total_budget": budget, "nights": 2,
        "time_feasibility": time_constraints.get("feasible", True)
    }
    
    result = orchestrator.select_bookings(context)
    
    selected_flight = result.get("selected_flight")
    selected_hotel = result.get("selected_hotel")
    
    if selected_flight and hasattr(selected_flight, 'model_dump'):
        selected_flight = selected_flight.model_dump()
    elif not selected_flight and flights:
        selected_flight = flights[0]
    
    if selected_hotel and hasattr(selected_hotel, 'model_dump'):
        selected_hotel = selected_hotel.model_dump()
    elif not selected_hotel and hotels:
        selected_hotel = hotels[0]
    
    messages = []
    if selected_flight:
        messages.append(create_cnp_message(
            performative="accept", sender=AgentRole.ORCHESTRATOR.value,
            receiver=AgentRole.FLIGHT_AGENT.value,
            content={"accepted_proposal": selected_flight.get('flight_id', '')}
        ))
        print(f"  ✓ Selected flight: {selected_flight.get('airline', 'Unknown')}")
    
    if selected_hotel:
        messages.append(create_cnp_message(
            performative="accept", sender=AgentRole.ORCHESTRATOR.value,
            receiver=AgentRole.HOTEL_AGENT.value,
            content={"accepted_proposal": selected_hotel.get('hotel_id', '')}
        ))
        print(f"  ✓ Selected hotel: {selected_hotel.get('name', 'Unknown')}")
    
    return {
        "selected_flight": selected_flight, "selected_hotel": selected_hotel,
        "current_phase": "finalizing", "messages": messages
    }


def finalize_node(state: TripPlanningState) -> Dict[str, Any]:
    """Finalize trip recommendation with explanation and metrics."""
    print("\n" + "-"*60)
    print("📝 FINALIZING RECOMMENDATION")
    print("-"*60)
    
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    cheaper_alternatives = state.get("cheaper_alternatives", [])
    compliance = state.get("compliance_status", {})
    time_constraints = state.get("time_constraints", {})
    metrics = state.get("metrics", {})
    budget = state.get("budget", 0)
    
    # FIXED: Pass full state to calculate_nights, not preferences
    nights = calculate_nights(state)
    if nights <= 0:
        nights = 2  # Default fallback
    
    # Calculate total cost
    flight_cost = 0
    hotel_cost_per_night = 0
    if selected_flight:
        flight_cost = selected_flight.get("price_usd", selected_flight.get("price", 0))
    if selected_hotel:
        hotel_cost_per_night = selected_hotel.get("price_per_night_usd", selected_hotel.get("price", 0))
    
    hotel_total = hotel_cost_per_night * nights
    total_cost = flight_cost + hotel_total
    
    # Build final recommendation
    final_recommendation = {
        "flight": selected_flight,
        "hotel": selected_hotel,
        "total_estimated_cost": total_cost,
        "flight_cost": flight_cost,
        "hotel_cost": hotel_total,
        "nights": nights,
        "compliance_status": compliance.get("overall_status", "unknown"),
        "timeline_feasible": time_constraints.get("feasible", True),
        "cheaper_alternatives": cheaper_alternatives,
        "generated_at": datetime.now().isoformat()
    }
    
    # Generate improved explanation
    explanation_parts = ["## Trip Planning Complete\n"]
    
    if selected_flight:
        f = selected_flight
        flight_class = f.get('class', 'Economy')
        explanation_parts.append(
            f"**Flight**: {f.get('airline', 'Unknown')} ({flight_class}) from {f.get('from_city', '')} to "
            f"{f.get('to_city', '')} at **${flight_cost}**. "
            f"Departure: {f.get('departure_time', 'TBD')}, Arrival: {f.get('arrival_time', 'TBD')}."
        )
    
    if selected_hotel:
        h = selected_hotel
        explanation_parts.append(
            f"**Hotel**: {h.get('name', 'Unknown')} ({h.get('stars', 'N/A')}★) in {h.get('city', '')} at "
            f"**${hotel_cost_per_night}/night × {nights} nights = ${hotel_total}**."
        )
    
    # Clear budget breakdown
    explanation_parts.append(f"\n### 💰 Budget Summary")
    explanation_parts.append(f"- Flight: ${flight_cost}")
    explanation_parts.append(f"- Hotel: ${hotel_cost_per_night}/night × {nights} nights = ${hotel_total}")
    explanation_parts.append(f"- **Total: ${total_cost}** / ${budget} budget")
    explanation_parts.append(f"- Remaining: ${budget - total_cost}")
    explanation_parts.append(f"- Utilization: {(total_cost / budget * 100):.1f}%" if budget > 0 else "- Utilization: N/A")
    
    explanation_parts.append(f"\n**Budget Status**: {compliance.get('overall_status', 'Not checked')}")
    
    if time_constraints.get("feasible"):
        explanation_parts.append("**Timeline**: Schedule is feasible with adequate buffer times.")
    else:
        explanation_parts.append("**Timeline**: ⚠️ Some scheduling concerns - review recommended.")
    
    if cheaper_alternatives:
        explanation_parts.append("\n### 🔄 Alternatives Considered")
        for i, alt in enumerate(cheaper_alternatives[:3], 1):
            explanation_parts.append(
                f"{i}. {alt.get('hotel', {}).get('name', 'Unknown')} - Total: ${alt.get('total_cost', 0)}"
            )
    
    explanation_parts.append("\n### 📊 Workflow Metrics")
    explanation_parts.append(f"- Backtracking iterations: {metrics.get('backtracking_count', 0)}")
    explanation_parts.append(f"- Negotiation rounds: {metrics.get('negotiation_rounds', 0)}")
    explanation_parts.append(f"- Message exchanges: {metrics.get('message_exchanges', 0)}")
    
    explanation = "\n\n".join(explanation_parts)
    
    metrics["workflow_end_time"] = datetime.now().isoformat()
    
    final_message = create_cnp_message(
        performative="inform", sender=AgentRole.ORCHESTRATOR.value, receiver="user",
        content={"action": "recommendation_complete", "recommendation": final_recommendation}
    )
    
    print("\n" + "="*60)
    print("✅ TRIP PLANNING COMPLETE")
    print("="*60)
    print(explanation)
    
    return {
        "final_recommendation": final_recommendation,
        "explanation": explanation,
        "workflow_complete": True,
        "current_phase": "complete",
        "messages": [final_message],
        "metrics": metrics
    }
