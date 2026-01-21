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
    """TimeAgent reports conflicts to PolicyAgent for flight re-selection.
    
    KEY IMPROVEMENT: Calculates the required arrival time based on meeting schedule
    and uses FlightAgent to search for flights arriving BEFORE that time.
    """
    from orchestrator.agents_config import flight_agent
    from agents.models import FlightQuery
    
    print("\n" + "-"*60)
    print("⏰ TIME→POLICY FEEDBACK - Requesting Better Flight Options")
    print("-"*60)
    
    time_constraints = state.get("time_constraints", {})
    conflicts = time_constraints.get("conflicts", [])
    available_flights = state.get("available_flights", [])
    available_hotels = state.get("available_hotels", [])
    selected_flight = state.get("selected_flight", {})
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    budget = state.get("budget", 2000)
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {}).copy()
    nights = calculate_nights(state)
    messages = list(state.get("messages", []))
    
    time_feedback_count = metrics.get("time_feedback_count", 0) + 1
    metrics["time_feedback_count"] = time_feedback_count
    metrics["negotiation_rounds"] = metrics.get("negotiation_rounds", 0) + 1
    
    print(f"  Time feedback iteration: {time_feedback_count}/{MAX_BACKTRACKING_ITERATIONS}")
    
    conflict_details = [c.get("message", str(c)) if isinstance(c, dict) else str(c) for c in conflicts]
    print(f"  Conflicts: {conflict_details[:2]}...")
    
    # Calculate REQUIRED ARRIVAL TIME from meeting time
    # Formula: meeting_time - (transit ~45min + buffer ~120min) = ~165 min before meeting
    meeting_times = preferences.get('meeting_times', [])
    required_arrival_time = None
    
    if meeting_times:
        mt = meeting_times[0]
        meeting_time_str = str(mt).split(' ', 1)[1] if ' ' in str(mt) else str(mt)
        try:
            meet_h, meet_m = map(int, meeting_time_str.split(':'))
            # Meeting minus 165 min (2h45m buffer for transit + preparation)
            required_mins = meet_h * 60 + meet_m - 165
            if required_mins < 0:
                required_mins = 0  # Very early morning
            req_h, req_m = required_mins // 60, required_mins % 60
            required_arrival_time = f"{req_h:02d}:{req_m:02d}"
            print(f"  📊 Meeting at {meeting_time_str} → Need flight arriving BEFORE {required_arrival_time}")
        except:
            required_arrival_time = "10:00"  # Default fallback
    
    # NO BIAS: Don't constrain by current price - use budget remaining
    # The orchestrator should decide what price range is acceptable based on budget
    selected_hotel = state.get("selected_hotel", {})
    current_total = selected_flight.get("price_usd", 0) + (selected_hotel.get("price_per_night_usd", 0) * nights if selected_hotel else 0)
    budget_remaining = budget - current_total

    # Search range: Allow flexibility based on overall budget, not current flight price
    # Min: Don't go below minimum market price
    # Max: Can use remaining budget if needed for earlier flight
    price_min = 100  # Minimum reasonable flight price
    price_max = min(budget * 0.6, budget_remaining + selected_flight.get("price_usd", 0))  # Can use up to 60% of total budget on flight

    # TimeAgent sends feedback to PolicyAgent with SPECIFIC requirements
    messages.append(create_cnp_message(
        performative="inform", sender=AgentRole.TIME_AGENT.value,
        receiver=AgentRole.POLICY_AGENT.value,
        content={
            "issue": "timeline_conflict",
            "conflicts": conflict_details,
            "current_flight": selected_flight.get("flight_id", "unknown"),
            "required_arrival_before": required_arrival_time,
            "target_price_range": f"${price_min}-${price_max}",
            "request": f"Find flight arriving BEFORE {required_arrival_time} - any price within budget"
        }
    ))

    print(f"\n  ┌─ CNP MESSAGE: TimeAgent → PolicyAgent (timeline_conflict)")
    print(f"  │ Required: Arrive BEFORE {required_arrival_time}")
    print(f"  │ Price Range: ${price_min}-${price_max} (based on budget, not current flight)")
    print(f"  └─ Orchestrator will select best option that meets time requirement")
    
    # STEP 1: Use FlightAgent to search for earlier flights
    print(f"\n  ✈️  [FlightAgent] Searching for flights arriving before {required_arrival_time}...")
    
    flight_query = FlightQuery(
        from_city=origin,
        to_city=destination,
        max_price=price_max,
        departure_before=required_arrival_time  # This filters by departure, but we check arrival manually
    )
    
    # Search with FlightAgent's loader directly for precise control
    all_flights = flight_agent.loader.search(from_city=origin, to_city=destination)

    # Filter for earlier arrivals within budget (NO BIAS on price)
    better_flights = []
    for f in all_flights:
        try:
            arr_time = f.get("arrival_time", "12:00")
            arr_h, arr_m = map(int, arr_time.split(":"))
            flight_price = f.get("price_usd", 0)

            arr_mins = arr_h * 60 + arr_m
            req_mins = int(required_arrival_time.split(":")[0]) * 60 + int(required_arrival_time.split(":")[1]) if required_arrival_time else 10 * 60

            # Arrives BEFORE required time AND within budget
            is_early_enough = arr_mins <= req_mins
            is_affordable = flight_price <= price_max

            if is_early_enough and is_affordable:
                better_flights.append(f)
        except:
            continue

    # Sort by arrival time (earliest first) - NO BIAS on price in sorting
    better_flights.sort(key=lambda x: x.get("arrival_time", "23:59"))
    print(f"  ✈️  Found {len(better_flights)} flights arriving before {required_arrival_time} within budget")

    # Show diversity: cheapest, median, and premium options
    if len(better_flights) > 3:
        sorted_by_price = sorted(better_flights, key=lambda x: x.get("price_usd", 999))
        print(f"     Price range: ${sorted_by_price[0].get('price_usd')}-${sorted_by_price[-1].get('price_usd')}")
        print(f"     Classes available: {set(f.get('class', 'Economy') for f in better_flights)}")
    
    if better_flights:
        print(f"\n  [PolicyAgent] Re-evaluating with {len(better_flights)} earlier flights...")
        
        # Update available flights with the new options
        messages.append(create_cnp_message(
            performative="inform", sender=AgentRole.FLIGHT_AGENT.value,
            receiver=AgentRole.POLICY_AGENT.value,
            content={"response": "earlier_flights_found", "count": len(better_flights)}
        ))
        
        combination_result = policy_agent.find_best_combination(
            flights=better_flights, hotels=available_hotels,
            budget=budget, nights=nights, preferences=preferences
        )
        
        if combination_result.success:
            new_flight = combination_result.selected_flight
            new_hotel = combination_result.selected_hotel
            new_arrival = new_flight.get('arrival_time', '12:00')
            
            print(f"  ✅ Found better flight: {new_flight.get('flight_id')} arriving at {new_arrival} (${new_flight.get('price_usd', 0)})")
            
            # Calculate new buffer
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
                    print(f"  ✅ New buffer: {new_buffer} min (feasible: {is_now_feasible})")
                except:
                    pass
            
            messages.append(create_cnp_message(
                performative="inform", sender=AgentRole.POLICY_AGENT.value,
                receiver=AgentRole.TIME_AGENT.value,
                content={"response": "alternative_found", "new_flight": new_flight.get("flight_id"), "new_arrival": new_arrival}
            ))
            
            return {
                "selected_flight": new_flight,
                "selected_hotel": new_hotel,
                "flight_alternatives": better_flights,
                "available_flights": better_flights,
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
        content={"response": "no_alternative", "reasoning": f"No flights arriving before {required_arrival_time} within budget"}
    ))
    
    return {"current_phase": "select_options", "messages": messages, "metrics": metrics, "flight_alternatives": []}
