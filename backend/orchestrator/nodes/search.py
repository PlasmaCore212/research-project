"""
Parallel Search Node - Flight & Hotel agents search simultaneously.

This node executes parallel search by:
1. FlightAgent searches without budget constraints
2. HotelAgent searches without budget constraints
3. Both return diverse options across price/quality tiers
4. PolicyAgent will later select the best combination
"""

from typing import Dict, Any
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message
from orchestrator.agents_config import flight_agent, hotel_agent
from agents.models import FlightQuery, HotelQuery


def parallel_search_node(state: TripPlanningState) -> Dict[str, Any]:
    """Execute Flight and Hotel search in parallel without budget constraints."""
    print("\n" + "-"*60)
    print("🔄 PARALLEL SEARCH - Flight & Hotel Agents (No Budget Filter)")
    print("-"*60)
    
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    budget = state.get("budget")
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    
    print(f"  Total budget: ${budget} (PolicyAgent will validate later)")
    print(f"  Strategy: Each agent returns their SINGLE best option")
    
    # ===== FLIGHT AGENT =====
    print("\n  ✈️  Flight Agent searching (no budget filter)...")
    flight_agent.reset_state()
    
    flight_query = FlightQuery(
        from_city=origin, to_city=destination, max_price=None,
        departure_after=preferences.get("departure_after", "06:00"),
        departure_before=preferences.get("departure_before", "21:00"),
        class_preference=preferences.get("class_preference", "Economy")
    )
    
    flight_result = flight_agent.search_flights(flight_query)
    
    if hasattr(flight_result, 'flights'):
        flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in flight_result.flights]
        flight_reasoning = flight_result.reasoning if hasattr(flight_result, 'reasoning') else ""
    else:
        flights = flight_result.get('flights', [])
        flight_reasoning = flight_result.get('reasoning', '')
    
    flight_traces = []
    if hasattr(flight_agent, 'state') and flight_agent.state:
        for step in flight_agent.state.reasoning_trace:
            flight_traces.append({
                "thought": step.thought, "action": step.action,
                "action_input": step.action_input, "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    print(f"  ✓ Flight Agent found {len(flights)} options after {len(flight_traces)} reasoning steps")
    
    # ===== HOTEL AGENT =====
    print("\n  🏨 Hotel Agent searching (no budget filter)...")
    hotel_agent.reset_state()
    
    meeting_location = None
    meetings = state.get("meetings", [])
    if meetings and len(meetings) > 0:
        first_meeting = meetings[0]
        if isinstance(first_meeting, dict) and "location" in first_meeting:
            meeting_location = first_meeting["location"]
    
    hotel_query = HotelQuery(
        city=destination, max_price_per_night=None, min_stars=None,
        required_amenities=preferences.get("required_amenities"),
        meeting_location=meeting_location
    )
    
    hotel_result = hotel_agent.search_hotels(hotel_query)
    
    if hasattr(hotel_result, 'hotels'):
        hotels = [h.model_dump() if hasattr(h, 'model_dump') else h for h in hotel_result.hotels]
        hotel_reasoning = hotel_result.reasoning if hasattr(hotel_result, 'reasoning') else ""
    else:
        hotels = hotel_result.get('hotels', [])
        hotel_reasoning = hotel_result.get('reasoning', '')
    
    hotel_traces = []
    if hasattr(hotel_agent, 'state') and hotel_agent.state:
        for step in hotel_agent.state.reasoning_trace:
            hotel_traces.append({
                "thought": step.thought, "action": step.action,
                "action_input": step.action_input, "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    print(f"  ✓ Hotel Agent found {len(hotels)} options after {len(hotel_traces)} reasoning steps")
    
    # ===== CREATE CNP MESSAGES =====
    messages = [
        create_cnp_message(
            performative="propose", sender=AgentRole.FLIGHT_AGENT.value,
            receiver=AgentRole.ORCHESTRATOR.value,
            content={"proposal": "flight_options", "options_count": len(flights),
                    "best_option": flights[0] if flights else None,
                    "reasoning_steps": len(flight_traces),
                    "analysis": flight_reasoning[:500] if flight_reasoning else ""}
        ),
        create_cnp_message(
            performative="propose", sender=AgentRole.HOTEL_AGENT.value,
            receiver=AgentRole.ORCHESTRATOR.value,
            content={"proposal": "hotel_options", "options_count": len(hotels),
                    "best_option": hotels[0] if hotels else None,
                    "reasoning_steps": len(hotel_traces),
                    "analysis": hotel_reasoning[:500] if hotel_reasoning else ""}
        )
    ]
    
    metrics["parallel_searches_executed"] = metrics.get("parallel_searches_executed", 0) + 1
    
    print(f"\n  ✓ Parallel search complete: {len(flights)} flights, {len(hotels)} hotels")
    
    return {
        "available_flights": flights,
        "available_hotels": hotels,
        "flight_analysis": {"total_options": len(flights), "recommended": flights[0] if flights else None, "reasoning": flight_reasoning},
        "hotel_analysis": {"total_options": len(hotels), "recommended": hotels[0] if hotels else None, "reasoning": hotel_reasoning},
        "current_phase": "budget_check",
        "messages": messages,
        "reasoning_traces": {AgentRole.FLIGHT_AGENT.value: flight_traces, AgentRole.HOTEL_AGENT.value: hotel_traces},
        "metrics": metrics
    }
