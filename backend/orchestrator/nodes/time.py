"""
Time Check Nodes - Timeline feasibility analysis and feedback.

This module contains:
1. check_time_node: TimeAgent validates timeline feasibility
2. time_policy_feedback_node: TimeAgent reports conflicts to PolicyAgent
"""

from typing import Dict, Any
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import (
    create_cnp_message, calculate_nights, dict_to_flight, dict_to_hotel,
    MAX_BACKTRACKING_ITERATIONS
)
from orchestrator.agents_config import time_agent, policy_agent
from agents.models import FlightQuery, HotelQuery, FlightSearchResult, HotelSearchResult, Meeting


def check_time_node(state: TripPlanningState) -> Dict[str, Any]:
    """TimeAgent validates timeline feasibility with selected options."""
    print("\n" + "-"*60)
    print("⏰ TIME AGENT - Timeline Feasibility Analysis")
    print("-"*60)
    
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    available_flights = state.get("available_flights", [])
    available_hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    departure_date = state.get("departure_date", "")
    preferences = state.get("preferences", {})
    
    time_agent.reset_state()
    
    # Build flight models
    flight_models = []
    if selected_flight:
        try:
            flight_models.append(dict_to_flight(selected_flight, origin, destination))
        except Exception as e:
            print(f"  Warning: Could not create Flight model: {e}")
    if not flight_models:
        for f in available_flights[:3]:
            try:
                flight_models.append(dict_to_flight(f, origin, destination))
            except:
                pass
    
    # Build hotel models
    hotel_models = []
    if selected_hotel:
        try:
            hotel_models.append(dict_to_hotel(selected_hotel, destination))
        except Exception as e:
            print(f"  Warning: Could not create Hotel model: {e}")
    if not hotel_models:
        for h in available_hotels[:3]:
            try:
                hotel_models.append(dict_to_hotel(h, destination))
            except:
                pass
    
    # Create result objects
    flight_query = FlightQuery(from_city=origin, to_city=destination)
    hotel_query = HotelQuery(city=destination)
    
    flight_result = FlightSearchResult(query=flight_query, flights=flight_models, reasoning="Flights for time analysis")
    hotel_result = HotelSearchResult(query=hotel_query, hotels=hotel_models, reasoning="Hotels for time analysis")
    
    # Parse meeting times
    meeting_times = preferences.get("meeting_times", [])
    meeting_location = preferences.get("meeting_location", {"lat": 37.7749, "lon": -122.4194})
    meetings = []
    for mt in meeting_times:
        try:
            if " " in str(mt):
                date_part, time_part = str(mt).split(" ", 1)
            else:
                date_part = departure_date
                time_part = str(mt)
            meetings.append(Meeting(date=date_part, time=time_part, location=meeting_location, duration_minutes=60))
        except Exception as e:
            print(f"  Warning: Could not parse meeting time '{mt}': {e}")
    
    # Get coordinates - only need airport coords and meeting location
    from utils.routing import get_airport_coords
    
    airport_coords = get_airport_coords(destination)
    
    # Use meeting location directly (already provided in preferences)
    if isinstance(meeting_location, dict) and meeting_location.get("lat"):
        meeting_coords = meeting_location
    else:
        # Fallback - should not happen if test data provides meeting_coordinates
        meeting_coords = {"lat": 40.7580, "lon": -73.9855}  # Default fallback
    
    # Run time agent's feasibility check
    result = time_agent.check_feasibility(
        flight_result=flight_result, hotel_result=hotel_result, meetings=meetings,
        arrival_city_coords=meeting_coords, airport_coords=airport_coords,
        departure_date=departure_date, destination_city=destination
    )
    
    # Handle result
    if hasattr(result, 'is_feasible'):
        is_feasible = result.is_feasible
        conflicts = [c.model_dump() if hasattr(c, 'model_dump') else c for c in result.conflicts]
        time_reasoning = result.reasoning
        timeline = result.timeline if hasattr(result, 'timeline') else {}
    else:
        is_feasible = result.get('is_feasible', True)
        conflicts = result.get('conflicts', [])
        time_reasoning = result.get('reasoning', '')
        timeline = result.get('timeline', {})
    
    # Collect reasoning traces
    time_traces = []
    if hasattr(time_agent, 'state') and time_agent.state:
        for step in time_agent.state.reasoning_trace:
            time_traces.append({
                "thought": step.thought, "action": step.action,
                "action_input": step.action_input, "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    messages = [create_cnp_message(
        performative="inform", sender=AgentRole.TIME_AGENT.value,
        receiver=AgentRole.ORCHESTRATOR.value,
        content={
            "feasibility_check_complete": True, "is_feasible": is_feasible,
            "conflicts_count": len(conflicts), "conflicts": conflicts,
            "timeline": timeline, "reasoning": time_reasoning[:500] if time_reasoning else ""
        }
    )]
    
    print(f"  ✓ Timeline feasible: {is_feasible}")
    if conflicts:
        print(f"  ⚠️ Found {len(conflicts)} scheduling conflicts")
    
    return {
        "time_constraints": {"feasible": is_feasible, "timeline": timeline, "conflicts": conflicts, "reasoning": time_reasoning},
        "feasibility_analysis": {"is_feasible": is_feasible, "timeline": timeline, "conflicts": conflicts, "reasoning": time_reasoning},
        "messages": messages,
        "reasoning_traces": {AgentRole.TIME_AGENT.value: time_traces}
    }


def time_policy_feedback_node(state: TripPlanningState) -> Dict[str, Any]:
    """TimeAgent reports conflicts to PolicyAgent for flight re-selection."""
    print("\n" + "-"*60)
    print("⏰ TIME→POLICY FEEDBACK - Requesting Better Flight Options")
    print("-"*60)
    
    time_constraints = state.get("time_constraints", {})
    conflicts = time_constraints.get("conflicts", [])
    available_flights = state.get("available_flights", [])
    available_hotels = state.get("available_hotels", [])
    selected_flight = state.get("selected_flight", {})
    budget = state.get("budget", 2000)
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    nights = calculate_nights(state)
    messages = list(state.get("messages", []))
    
    time_feedback_count = metrics.get("time_feedback_count", 0) + 1
    metrics["time_feedback_count"] = time_feedback_count
    
    print(f"  Time feedback iteration: {time_feedback_count}/{MAX_BACKTRACKING_ITERATIONS}")
    
    conflict_details = [c.get("message", str(c)) if isinstance(c, dict) else str(c) for c in conflicts]
    print(f"  Conflicts: {conflict_details[:2]}...")
    
    # TimeAgent sends feedback to PolicyAgent
    messages.append(create_cnp_message(
        performative="inform", sender=AgentRole.TIME_AGENT.value,
        receiver=AgentRole.POLICY_AGENT.value,
        content={
            "issue": "timeline_conflict", "conflicts": conflict_details,
            "current_flight": selected_flight.get("flight_id", "unknown"),
            "request": "Find flight with earlier arrival"
        }
    ))
    
    print(f"\n  ┌─ CNP MESSAGE: TimeAgent → PolicyAgent (timeline_conflict)")
    print(f"  └─ Request: Earlier arrival time needed")
    
    # Find flights with earlier arrivals
    current_arrival = selected_flight.get("arrival_time", "12:00")
    try:
        arr_h, arr_m = map(int, current_arrival.split(":"))
        current_arr_mins = arr_h * 60 + arr_m
    except:
        current_arr_mins = 12 * 60
    
    better_flights = []
    for f in available_flights:
        try:
            arr_time = f.get("arrival_time", "12:00")
            arr_h, arr_m = map(int, arr_time.split(":"))
            if arr_h * 60 + arr_m < current_arr_mins:
                better_flights.append(f)
        except:
            continue
    
    better_flights.sort(key=lambda x: x.get("arrival_time", "23:59"))
    print(f"  Found {len(better_flights)} flights with earlier arrival times")
    
    if better_flights:
        print(f"  [PolicyAgent] Re-evaluating with earlier flights...")
        combination_result = policy_agent.find_best_combination(
            flights=better_flights, hotels=available_hotels,
            budget=budget, nights=nights, preferences=preferences
        )
        
        if combination_result.success:
            new_flight = combination_result.selected_flight
            new_hotel = combination_result.selected_hotel
            new_arrival = new_flight.get('arrival_time', '12:00')
            
            print(f"  ✅ Found better flight arriving at {new_arrival}")
            
            # Calculate new buffer
            meeting_times = preferences.get('meeting_times', [])
            is_now_feasible = True
            new_buffer = None
            if meeting_times:
                mt = meeting_times[0]
                meeting_time = str(mt).split(' ', 1)[1] if ' ' in str(mt) else str(mt)
                try:
                    arr_h, arr_m = map(int, new_arrival.split(':'))
                    meet_h, meet_m = map(int, meeting_time.split(':'))
                    new_buffer = (meet_h * 60 + meet_m) - (arr_h * 60 + arr_m + 45)
                    is_now_feasible = new_buffer >= 120
                except:
                    pass
            
            messages.append(create_cnp_message(
                performative="inform", sender=AgentRole.POLICY_AGENT.value,
                receiver=AgentRole.TIME_AGENT.value,
                content={"response": "alternative_found", "new_flight": new_flight.get("flight_id")}
            ))
            
            return {
                "selected_flight": new_flight,
                "selected_hotel": new_hotel,
                "time_constraints": {"feasible": is_now_feasible, "timeline": {"flight_arrival": new_arrival}, "conflicts": [], "reasoning": f"Buffer: {new_buffer} min"},
                "feasibility_analysis": {"is_feasible": is_now_feasible, "reasoning": f"Earlier flight arriving at {new_arrival}"},
                "compliance_status": {
                    "overall_status": "compliant", "is_compliant": True, "violations": [],
                    "total_cost": combination_result.total_cost, "budget": budget,
                    "budget_remaining": combination_result.budget_remaining
                },
                "current_phase": "select_options",
                "messages": messages,
                "metrics": metrics
            }
    
    print(f"  ⚠️ No earlier flights available - proceeding with current selection")
    
    messages.append(create_cnp_message(
        performative="inform", sender=AgentRole.POLICY_AGENT.value,
        receiver=AgentRole.TIME_AGENT.value,
        content={"response": "no_alternative", "reasoning": "No earlier flights within budget"}
    ))
    
    return {"current_phase": "select_options", "messages": messages, "metrics": metrics}
