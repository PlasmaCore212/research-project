"""
LangGraph Workflow for Agentic Trip Planning System

This module implements the workflow orchestration using LangGraph's StateGraph.
The workflow coordinates multiple ReAct-based agents:
- Flight Agent: Searches and analyzes flight options
- Hotel Agent: Searches and analyzes hotel options
- Policy Agent: Checks compliance with company policies
- Time Agent: Validates timeline feasibility

The workflow follows an agentic pattern where:
1. Each agent reasons autonomously using ReAct
2. Agents communicate through structured messages
3. The orchestrator coordinates and resolves conflicts
4. Backtracking and re-planning is supported

Author: Research Project
"""

import os
import sys
from typing import Dict, Any, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.state import (
    TripPlanningState, 
    create_initial_state,
    add_message_to_state,
    add_reasoning_trace,
    update_metrics,
    AgentRole,
    MessageType
)
from orchestrator.orchestrator import TripOrchestrator
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.policy_agent import PolicyAgent
from agents.time_agent import TimeManagementAgent
from agents.models import (
    FlightQuery, HotelQuery, FlightSearchResult, HotelSearchResult, 
    Flight, Hotel, Meeting, TimeCheckResult
)
from data.loaders import FlightDataLoader, HotelDataLoader, PolicyDataLoader

# Initialize agents (they maintain their own state/beliefs)
flight_agent = FlightAgent()
hotel_agent = HotelAgent()
policy_agent = PolicyAgent()
time_agent = TimeManagementAgent()
orchestrator = TripOrchestrator()

# Initialize data loaders
flight_loader = FlightDataLoader()
hotel_loader = HotelDataLoader()
policy_loader = PolicyDataLoader()


def initialize_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Initialize the workflow by loading data and setting up agent beliefs.
    
    This node:
    1. Loads flight, hotel, and policy data
    2. Initializes the orchestrator's beliefs
    3. Creates initial message indicating workflow start
    """
    print("\n" + "="*60)
    print("🚀 INITIALIZING AGENTIC TRIP PLANNING WORKFLOW")
    print("="*60)
    
    # Load data using loaders
    flights_data = flight_loader.flights
    hotels_data = hotel_loader.hotels
    policies_data = policy_loader.get_policy("standard")
    
    # Initialize orchestrator beliefs
    orchestrator.memory.add_belief("workflow_started", datetime.now().isoformat())
    orchestrator.memory.add_belief("user_request", state.get("user_request", ""))
    orchestrator.memory.add_belief("origin", state.get("origin", ""))
    orchestrator.memory.add_belief("destination", state.get("destination", ""))
    
    # Create initialization message
    init_message = {
        "sender": AgentRole.ORCHESTRATOR.value,
        "receiver": "all",
        "type": MessageType.INFORM.value,
        "content": {
            "action": "workflow_initialized",
            "flights_loaded": len(flights_data),
            "hotels_loaded": len(hotels_data),
            "policies_loaded": bool(policies_data)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✓ Loaded {len(flights_data)} flights")
    print(f"✓ Loaded {len(hotels_data)} hotels")
    print(f"✓ Loaded policy rules")
    
    return {
        "current_phase": "searching",
        "policy_rules": policies_data,
        "messages": [init_message]
    }


def search_flights_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Flight Agent searches for flights using ReAct reasoning.
    
    The agent:
    1. Analyzes the user request
    2. Searches for matching flights
    3. Reasons about which flights best match requirements
    4. Returns its analysis and recommendations
    """
    print("\n" + "-"*60)
    print("✈️  FLIGHT AGENT - ReAct Reasoning")
    print("-"*60)
    
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    departure_date = state.get("departure_date", "")
    budget = state.get("budget")
    preferences = state.get("preferences", {})
    
    # Build FlightQuery from state
    query = FlightQuery(
        from_city=origin,
        to_city=destination,
        max_price=int(budget * 0.6) if budget else None,
        departure_after=preferences.get("departure_after", "06:00"),
        departure_before=preferences.get("departure_before", "21:00"),
        class_preference=preferences.get("class_preference", "Economy")
    )
    
    # Run the ReAct-based flight agent
    result = flight_agent.search_flights(query)
    
    # Extract flights from result (handle both Pydantic model and dict)
    if hasattr(result, 'flights'):
        flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in result.flights]
        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
    else:
        flights = result.get('flights', [])
        reasoning = result.get('reasoning', '')
    
    # Extract reasoning traces
    reasoning_traces = {
        AgentRole.FLIGHT_AGENT.value: [{
            "thought": f"Searching for flights from {origin} to {destination}",
            "action": "search_flights",
            "action_input": query.model_dump(),
            "observation": f"Found {len(flights)} matching flights",
            "timestamp": datetime.now().isoformat()
        }]
    }
    
    # Add any traces from the agent's internal reasoning
    if hasattr(flight_agent, 'state') and flight_agent.state:
        for step in flight_agent.state.reasoning_trace:
            reasoning_traces[AgentRole.FLIGHT_AGENT.value].append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # Get recommended flight
    recommended = flights[0] if flights else None
    
    # Create message about results
    flight_message = {
        "sender": AgentRole.FLIGHT_AGENT.value,
        "receiver": AgentRole.ORCHESTRATOR.value,
        "type": MessageType.INFORM.value,
        "content": {
            "action": "flights_found",
            "count": len(flights),
            "analysis": reasoning,
            "recommendation": recommended
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✓ Found {len(flights)} flights")
    if recommended:
        print(f"✓ Recommended: {recommended.get('airline', 'Unknown')} - ${recommended.get('price_usd', 'N/A')}")
    
    return {
        "available_flights": flights,
        "flight_analysis": {
            "total_options": len(flights),
            "recommended": recommended,
            "reasoning": reasoning
        },
        "messages": [flight_message],
        "reasoning_traces": reasoning_traces
    }


def search_hotels_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Hotel Agent searches for hotels using ReAct reasoning.
    
    The agent:
    1. Analyzes destination and requirements
    2. Searches for matching hotels
    3. Reasons about amenities, location, and value
    4. Returns its analysis and recommendations
    """
    print("\n" + "-"*60)
    print("🏨 HOTEL AGENT - ReAct Reasoning")
    print("-"*60)
    
    destination = state.get("destination", "")
    budget = state.get("budget")
    preferences = state.get("preferences", {})
    
    # Build HotelQuery from state
    query = HotelQuery(
        city=destination,
        max_price_per_night=int(budget * 0.4) if budget else None,
        min_stars=int(preferences.get("min_rating", 3)),
        required_amenities=preferences.get("required_amenities")
    )
    
    # Run the ReAct-based hotel agent
    result = hotel_agent.search_hotels(query)
    
    # Extract hotels from result (handle both Pydantic model and dict)
    if hasattr(result, 'hotels'):
        hotels = [h.model_dump() if hasattr(h, 'model_dump') else h for h in result.hotels]
        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
    else:
        hotels = result.get('hotels', [])
        reasoning = result.get('reasoning', '')
    
    # Extract reasoning traces
    reasoning_traces = {
        AgentRole.HOTEL_AGENT.value: [{
            "thought": f"Searching for hotels in {destination}",
            "action": "search_hotels",
            "action_input": query.model_dump(),
            "observation": f"Found {len(hotels)} matching hotels",
            "timestamp": datetime.now().isoformat()
        }]
    }
    
    # Add agent's internal reasoning
    if hasattr(hotel_agent, 'state') and hotel_agent.state:
        for step in hotel_agent.state.reasoning_trace:
            reasoning_traces[AgentRole.HOTEL_AGENT.value].append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # Get recommended hotel
    recommended = hotels[0] if hotels else None
    
    # Create message
    hotel_message = {
        "sender": AgentRole.HOTEL_AGENT.value,
        "receiver": AgentRole.ORCHESTRATOR.value,
        "type": MessageType.INFORM.value,
        "content": {
            "action": "hotels_found",
            "count": len(hotels),
            "analysis": reasoning,
            "recommendation": recommended
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✓ Found {len(hotels)} hotels")
    if recommended:
        print(f"✓ Recommended: {recommended.get('name', 'Unknown')} - ${recommended.get('price_per_night_usd', 'N/A')}/night")
    
    return {
        "available_hotels": hotels,
        "hotel_analysis": {
            "total_options": len(hotels),
            "recommended": recommended,
            "reasoning": reasoning
        },
        "messages": [hotel_message],
        "reasoning_traces": reasoning_traces
    }


def check_policy_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Policy Agent checks compliance using ReAct reasoning.
    
    The agent:
    1. Loads and interprets company policies
    2. Checks flight and hotel options against policies
    3. Identifies violations and suggests alternatives
    4. Provides compliance assessment
    """
    print("\n" + "-"*60)
    print("📋 POLICY AGENT - ReAct Reasoning")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    
    # Create proper result objects for policy agent
    # Convert flight dicts to Flight models
    flight_models = []
    for f in flights[:3]:  # Top 3 flights
        try:
            flight_models.append(Flight(
                flight_id=f.get('flight_id', ''),
                airline=f.get('airline', ''),
                from_city=f.get('from_city', origin),
                to_city=f.get('to_city', destination),
                departure_time=f.get('departure_time', '09:00'),
                arrival_time=f.get('arrival_time', '12:00'),
                duration_hours=f.get('duration_hours', 3.0),
                price_usd=f.get('price_usd', 0),
                seats_available=f.get('seats_available', 10),
                **{'class': f.get('class', f.get('flight_class', 'Economy'))}
            ))
        except Exception as e:
            print(f"Warning: Could not create Flight model: {e}")
    
    # Convert hotel dicts to Hotel models
    hotel_models = []
    for h in hotels[:3]:  # Top 3 hotels
        try:
            hotel_models.append(Hotel(
                hotel_id=h.get('hotel_id', ''),
                name=h.get('name', ''),
                city=h.get('city', destination),
                city_name=h.get('city_name', destination),
                business_area=h.get('business_area', ''),
                tier=h.get('tier', 'standard'),
                stars=h.get('stars', 3),
                price_per_night_usd=h.get('price_per_night_usd', 0),
                distance_to_business_center_km=h.get('distance_to_business_center_km', 1.0),
                distance_to_airport_km=h.get('distance_to_airport_km', 10.0),
                amenities=h.get('amenities', []),
                rooms_available=h.get('rooms_available', 5),
                coordinates=h.get('coordinates', {'lat': 0, 'lng': 0})
            ))
        except Exception as e:
            print(f"Warning: Could not create Hotel model: {e}")
    
    # Create result objects
    flight_query = FlightQuery(from_city=origin, to_city=destination)
    hotel_query = HotelQuery(city=destination)
    
    flight_result = FlightSearchResult(
        query=flight_query,
        flights=flight_models,
        reasoning="Flights from previous search"
    )
    
    hotel_result = HotelSearchResult(
        query=hotel_query,
        hotels=hotel_models,
        reasoning="Hotels from previous search"
    )
    
    # Run ReAct-based policy agent
    result = policy_agent.check_compliance(flight_result, hotel_result)
    
    # Handle result (could be Pydantic model or dict)
    if hasattr(result, 'is_compliant'):
        is_compliant = result.is_compliant
        violations = [v.model_dump() if hasattr(v, 'model_dump') else v for v in result.violations]
        reasoning = result.reasoning
    else:
        is_compliant = result.get('is_compliant', True)
        violations = result.get('violations', [])
        reasoning = result.get('reasoning', '')
    
    status = "compliant" if is_compliant else "non_compliant"
    
    # Extract reasoning traces
    reasoning_traces = {
        AgentRole.POLICY_AGENT.value: [{
            "thought": "Checking all options against company travel policies",
            "action": "check_compliance",
            "action_input": {"flight_count": len(flights), "hotel_count": len(hotels)},
            "observation": f"Compliance check complete: {status}",
            "timestamp": datetime.now().isoformat()
        }]
    }
    
    # Add agent's internal reasoning
    if hasattr(policy_agent, 'state') and policy_agent.state:
        for step in policy_agent.state.reasoning_trace:
            reasoning_traces[AgentRole.POLICY_AGENT.value].append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # Create message
    policy_message = {
        "sender": AgentRole.POLICY_AGENT.value,
        "receiver": AgentRole.ORCHESTRATOR.value,
        "type": MessageType.INFORM.value,
        "content": {
            "action": "compliance_checked",
            "overall_status": status,
            "is_compliant": is_compliant,
            "violations": violations
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✓ Compliance status: {status}")
    if violations:
        print(f"⚠ Found {len(violations)} policy violations")
    
    return {
        "compliance_status": {
            "overall_status": status,
            "is_compliant": is_compliant,
            "violations": violations,
            "compliant_flights": flights,  # Keep all flights, mark status
            "compliant_hotels": hotels,    # Keep all hotels, mark status
            "reasoning": reasoning
        },
        "current_phase": "validating",
        "messages": [policy_message],
        "reasoning_traces": reasoning_traces
    }


def check_time_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Time Agent checks timeline feasibility using ReAct reasoning.
    
    The agent:
    1. Analyzes flight times and meeting schedules
    2. Calculates transit times and buffer requirements
    3. Identifies scheduling conflicts
    4. Suggests timeline adjustments
    """
    print("\n" + "-"*60)
    print("⏰ TIME AGENT - ReAct Reasoning")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    departure_date = state.get("departure_date", "")
    preferences = state.get("preferences", {})
    
    # Create flight models from dicts
    flight_models = []
    for f in flights[:3]:
        try:
            flight_models.append(Flight(
                flight_id=f.get('flight_id', ''),
                airline=f.get('airline', ''),
                from_city=f.get('from_city', origin),
                to_city=f.get('to_city', destination),
                departure_time=f.get('departure_time', '09:00'),
                arrival_time=f.get('arrival_time', '12:00'),
                duration_hours=f.get('duration_hours', 3.0),
                price_usd=f.get('price_usd', 0),
                seats_available=f.get('seats_available', 10),
                **{'class': f.get('class', f.get('flight_class', 'Economy'))}
            ))
        except Exception as e:
            print(f"Warning: Could not create Flight model: {e}")
    
    # Create hotel models from dicts
    hotel_models = []
    for h in hotels[:3]:
        try:
            hotel_models.append(Hotel(
                hotel_id=h.get('hotel_id', ''),
                name=h.get('name', ''),
                city=h.get('city', destination),
                city_name=h.get('city_name', destination),
                business_area=h.get('business_area', ''),
                tier=h.get('tier', 'standard'),
                stars=h.get('stars', 3),
                price_per_night_usd=h.get('price_per_night_usd', 0),
                distance_to_business_center_km=h.get('distance_to_business_center_km', 1.0),
                distance_to_airport_km=h.get('distance_to_airport_km', 10.0),
                amenities=h.get('amenities', []),
                rooms_available=h.get('rooms_available', 5),
                coordinates=h.get('coordinates', {'lat': 37.7749, 'lng': -122.4194})
            ))
        except Exception as e:
            print(f"Warning: Could not create Hotel model: {e}")
    
    # Create result objects
    flight_query = FlightQuery(from_city=origin, to_city=destination)
    hotel_query = HotelQuery(city=destination)
    
    flight_result = FlightSearchResult(
        query=flight_query,
        flights=flight_models,
        reasoning="Flights from previous search"
    )
    
    hotel_result = HotelSearchResult(
        query=hotel_query,
        hotels=hotel_models,
        reasoning="Hotels from previous search"
    )
    
    # Parse meeting times into Meeting objects
    meeting_times = preferences.get("meeting_times", [])
    meetings = []
    for mt in meeting_times:
        try:
            # Parse "2026-01-20 14:00" format
            if " " in mt:
                date_part, time_part = mt.split(" ", 1)
            else:
                date_part = departure_date
                time_part = mt
            meetings.append(Meeting(
                date=date_part,
                time=time_part,
                location={"lat": 37.7749, "lng": -122.4194},  # Default SF coords
                duration_minutes=60
            ))
        except Exception as e:
            print(f"Warning: Could not parse meeting time '{mt}': {e}")
    
    # Default coordinates for destination
    city_coords = {"lat": 37.7749, "lng": -122.4194}  # SF default
    airport_coords = {"lat": 37.6213, "lng": -122.379}  # SFO default
    
    # Run ReAct-based time agent
    result = time_agent.check_feasibility(
        flight_result=flight_result,
        hotel_result=hotel_result,
        meetings=meetings,
        arrival_city_coords=city_coords,
        airport_coords=airport_coords,
        departure_date=departure_date
    )
    
    # Handle result (could be Pydantic model or dict)
    if hasattr(result, 'is_feasible'):
        is_feasible = result.is_feasible
        conflicts = [c.model_dump() if hasattr(c, 'model_dump') else c for c in result.conflicts]
        reasoning = result.reasoning
        timeline = result.timeline
    else:
        is_feasible = result.get('is_feasible', True)
        conflicts = result.get('conflicts', [])
        reasoning = result.get('reasoning', '')
        timeline = result.get('timeline', {})
    
    # Extract reasoning traces
    reasoning_traces = {
        AgentRole.TIME_AGENT.value: [{
            "thought": "Analyzing timeline feasibility for trip schedule",
            "action": "check_feasibility",
            "action_input": {"flights": len(flights), "meetings": len(meetings)},
            "observation": f"Feasibility: {is_feasible}",
            "timestamp": datetime.now().isoformat()
        }]
    }
    
    # Add agent's internal reasoning
    if hasattr(time_agent, 'state') and time_agent.state:
        for step in time_agent.state.reasoning_trace:
            reasoning_traces[AgentRole.TIME_AGENT.value].append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # Create message
    time_message = {
        "sender": AgentRole.TIME_AGENT.value,
        "receiver": AgentRole.ORCHESTRATOR.value,
        "type": MessageType.INFORM.value,
        "content": {
            "action": "feasibility_checked",
            "feasible": is_feasible,
            "timeline": timeline,
            "conflicts": conflicts
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"✓ Timeline feasible: {is_feasible}")
    if conflicts:
        print(f"⚠ Found {len(conflicts)} scheduling conflicts")
    
    return {
        "time_constraints": {
            "feasible": is_feasible,
            "timeline": timeline,
            "conflicts": conflicts,
            "reasoning": reasoning
        },
        "feasibility_analysis": {
            "is_feasible": is_feasible,
            "timeline": timeline,
            "conflicts": conflicts,
            "reasoning": reasoning
        },
        "messages": [time_message],
        "reasoning_traces": reasoning_traces
    }


def select_options_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Orchestrator selects the best flight and hotel using Chain-of-Thought reasoning.
    
    The orchestrator:
    1. Reviews all agent analyses
    2. Considers compliance requirements
    3. Uses CoT to reason about trade-offs
    4. Selects optimal combination
    """
    print("\n" + "-"*60)
    print("🎯 ORCHESTRATOR - Chain-of-Thought Selection")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    compliance = state.get("compliance_status", {})
    time_constraints = state.get("time_constraints", {})
    budget = state.get("budget", 2000)
    
    if not flights or not hotels:
        print("[Orchestrator] No options to select from")
        return {
            "selected_flight": None,
            "selected_hotel": None,
            "current_phase": "finalizing",
            "messages": []
        }
    
    # Convert dicts to proper Flight/Hotel models for orchestrator
    flight_models = []
    for f in flights[:5]:
        try:
            flight_models.append(Flight(
                flight_id=f.get('flight_id', ''),
                airline=f.get('airline', ''),
                from_city=f.get('from_city', origin),
                to_city=f.get('to_city', destination),
                departure_time=f.get('departure_time', '09:00'),
                arrival_time=f.get('arrival_time', '12:00'),
                duration_hours=f.get('duration_hours', 3.0),
                price_usd=f.get('price_usd', 0),
                seats_available=f.get('seats_available', 10),
                **{'class': f.get('class', f.get('flight_class', 'Economy'))}
            ))
        except Exception as e:
            print(f"Warning: Could not create Flight model: {e}")
    
    hotel_models = []
    for h in hotels[:5]:
        try:
            hotel_models.append(Hotel(
                hotel_id=h.get('hotel_id', ''),
                name=h.get('name', ''),
                city=h.get('city', destination),
                city_name=h.get('city_name', destination),
                business_area=h.get('business_area', ''),
                tier=h.get('tier', 'standard'),
                stars=h.get('stars', 3),
                price_per_night_usd=h.get('price_per_night_usd', 0),
                distance_to_business_center_km=h.get('distance_to_business_center_km', 1.0),
                distance_to_airport_km=h.get('distance_to_airport_km', 10.0),
                amenities=h.get('amenities', []),
                rooms_available=h.get('rooms_available', 5),
                coordinates=h.get('coordinates', {'lat': 0, 'lng': 0})
            ))
        except Exception as e:
            print(f"Warning: Could not create Hotel model: {e}")
    
    # Use orchestrator's CoT selection with proper keys
    context = {
        "flight_options": flight_models,
        "hotel_options": hotel_models,
        "total_budget": budget,
        "nights": 2,  # Assume 2 nights for business trip
        "time_feasibility": time_constraints.get("feasible", True),
        "flight_analysis": state.get("flight_analysis", {}),
        "hotel_analysis": state.get("hotel_analysis", {})
    }
    
    result = orchestrator.select_bookings(context)
    
    # Get selected items (might be models or IDs)
    selected_flight = result.get("selected_flight")
    selected_hotel = result.get("selected_hotel")
    
    # Convert to dict if model
    if selected_flight and hasattr(selected_flight, 'model_dump'):
        selected_flight = selected_flight.model_dump()
    elif not selected_flight and flights:
        # Fallback to first flight
        selected_flight = flights[0]
        
    if selected_hotel and hasattr(selected_hotel, 'model_dump'):
        selected_hotel = selected_hotel.model_dump()
    elif not selected_hotel and hotels:
        # Fallback to first hotel
        selected_hotel = hotels[0]
    
    # Create selection message
    selection_message = {
        "sender": AgentRole.ORCHESTRATOR.value,
        "receiver": "all",
        "type": MessageType.PROPOSE.value,
        "content": {
            "action": "options_selected",
            "selected_flight": selected_flight,
            "selected_hotel": selected_hotel,
            "reasoning": result.get("reasoning", "")
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if selected_flight:
        price = selected_flight.get('price_usd', selected_flight.get('price', 'N/A'))
        print(f"✓ Selected flight: {selected_flight.get('airline', 'Unknown')} - ${price}")
    if selected_hotel:
        price = selected_hotel.get('price_per_night_usd', selected_hotel.get('price', 'N/A'))
        print(f"✓ Selected hotel: {selected_hotel.get('name', 'Unknown')} - ${price}/night")
    
    return {
        "selected_flight": selected_flight,
        "selected_hotel": selected_hotel,
        "current_phase": "finalizing",
        "messages": [selection_message]
    }


def finalize_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Finalize the trip recommendation with a comprehensive explanation.
    
    This node:
    1. Compiles all agent analyses
    2. Creates final recommendation
    3. Generates natural language explanation
    4. Records final metrics
    """
    print("\n" + "-"*60)
    print("📝 FINALIZING RECOMMENDATION")
    print("-"*60)
    
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    flight_analysis = state.get("flight_analysis", {})
    hotel_analysis = state.get("hotel_analysis", {})
    compliance = state.get("compliance_status", {})
    time_constraints = state.get("time_constraints", {})
    
    # Calculate total cost
    total_cost = 0
    if selected_flight:
        total_cost += selected_flight.get("price_usd", selected_flight.get("price", 0))
    if selected_hotel:
        # Assume 2 nights for calculation
        total_cost += selected_hotel.get("price_per_night_usd", selected_hotel.get("price", 0)) * 2
    
    # Build final recommendation
    final_recommendation = {
        "flight": selected_flight,
        "hotel": selected_hotel,
        "total_estimated_cost": total_cost,
        "compliance_status": compliance.get("overall_status", "unknown"),
        "timeline_feasible": time_constraints.get("feasible", True),
        "generated_at": datetime.now().isoformat()
    }
    
    # Generate explanation
    explanation_parts = []
    
    if selected_flight:
        f = selected_flight
        price = f.get('price_usd', f.get('price', 'N/A'))
        explanation_parts.append(
            f"**Flight**: {f.get('airline', 'Unknown')} flight from {f.get('from_city', f.get('origin', ''))} to "
            f"{f.get('to_city', f.get('destination', ''))} at ${price}. "
            f"Departure: {f.get('departure_time', 'TBD')}, Arrival: {f.get('arrival_time', 'TBD')}."
        )
    
    if selected_hotel:
        h = selected_hotel
        price = h.get('price_per_night_usd', h.get('price', 'N/A'))
        stars = h.get('stars', h.get('rating', 'N/A'))
        explanation_parts.append(
            f"**Hotel**: {h.get('name', 'Unknown')} in {h.get('city', '')} at "
            f"${price}/night. Rating: {stars}/5."
        )
    
    explanation_parts.append(f"**Total Estimated Cost**: ${total_cost}")
    explanation_parts.append(f"**Policy Compliance**: {compliance.get('overall_status', 'Not checked')}")
    
    if time_constraints.get("feasible"):
        explanation_parts.append("**Timeline**: Schedule is feasible with adequate buffer times.")
    else:
        explanation_parts.append("**Timeline**: ⚠️ Some scheduling concerns - review recommended.")
    
    explanation = "\n\n".join(explanation_parts)
    
    # Update metrics
    metrics = state.get("metrics", {})
    metrics["workflow_end_time"] = datetime.now().isoformat()
    
    # Final message
    final_message = {
        "sender": AgentRole.ORCHESTRATOR.value,
        "receiver": "user",
        "type": MessageType.CONFIRM.value,
        "content": {
            "action": "recommendation_complete",
            "recommendation": final_recommendation
        },
        "timestamp": datetime.now().isoformat()
    }
    
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


def should_continue_after_policy(state: TripPlanningState) -> Literal["check_time", "search_flights"]:
    """
    Determine next step after policy check.
    
    If there are severe violations with no compliant options,
    we may need to search again with adjusted criteria.
    """
    compliance = state.get("compliance_status", {})
    status = compliance.get("overall_status", "compliant")
    
    # Check if we have any compliant options
    compliant_flights = compliance.get("compliant_flights", [])
    compliant_hotels = compliance.get("compliant_hotels", [])
    
    # If no compliant options at all, we might need to re-search
    # For now, continue to time check - can implement backtracking later
    if status == "non_compliant" and not compliant_flights and not compliant_hotels:
        # Record backtracking
        metrics = state.get("metrics", {})
        metrics["backtracking_count"] = metrics.get("backtracking_count", 0) + 1
        
        # For now, continue anyway - real implementation would re-search
        print("⚠️ No compliant options found - continuing with available options")
    
    return "check_time"


def should_continue_after_time(state: TripPlanningState) -> Literal["select_options", "search_flights"]:
    """
    Determine next step after time feasibility check.
    
    If timeline is not feasible, we might need different flights.
    """
    time_constraints = state.get("time_constraints", {})
    feasible = time_constraints.get("feasible", True)
    
    if not feasible:
        conflicts = time_constraints.get("conflicts", [])
        print(f"⚠️ Timeline has {len(conflicts)} conflicts - continuing to selection")
        # Real implementation could trigger re-search for different flights
    
    return "select_options"


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for trip planning.
    
    Workflow structure:
    1. Initialize -> Search (parallel flights & hotels)
    2. Search -> Policy Check
    3. Policy Check -> Time Check (with possible backtrack)
    4. Time Check -> Select Options (with possible backtrack)
    5. Select Options -> Finalize
    6. Finalize -> END
    """
    
    # Create the state graph
    workflow = StateGraph(TripPlanningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("search_flights", search_flights_node)
    workflow.add_node("search_hotels", search_hotels_node)
    workflow.add_node("check_policy", check_policy_node)
    workflow.add_node("check_time", check_time_node)
    workflow.add_node("select_options", select_options_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    # After initialization, search for flights and hotels
    workflow.add_edge("initialize", "search_flights")
    workflow.add_edge("search_flights", "search_hotels")
    
    # After searches complete, check policy
    workflow.add_edge("search_hotels", "check_policy")
    
    # After policy check, conditionally proceed
    workflow.add_conditional_edges(
        "check_policy",
        should_continue_after_policy,
        {
            "check_time": "check_time",
            "search_flights": "search_flights"  # Backtrack if needed
        }
    )
    
    # After time check, conditionally proceed
    workflow.add_conditional_edges(
        "check_time",
        should_continue_after_time,
        {
            "select_options": "select_options",
            "search_flights": "search_flights"  # Backtrack if needed
        }
    )
    
    # Finalize and end
    workflow.add_edge("select_options", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


def create_trip_planning_app():
    """
    Create and compile the trip planning application.
    
    Returns:
        Compiled LangGraph application ready to run
    """
    workflow = build_workflow()
    app = workflow.compile()
    return app


# Convenience function for running the workflow
def plan_trip(
    user_request: str,
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = None,
    budget: float = None,
    preferences: dict = None
) -> Dict[str, Any]:
    """
    Plan a business trip using the multi-agent system.
    
    Args:
        user_request: Natural language description of the trip
        origin: Departure city/airport code
        destination: Arrival city/airport code
        departure_date: Date of departure (YYYY-MM-DD)
        return_date: Optional return date
        budget: Optional maximum budget
        preferences: Optional dict of preferences
        
    Returns:
        Final state with recommendation and explanation
    """
    # Create initial state
    initial_state = create_initial_state(
        user_request=user_request,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        preferences=preferences or {}
    )
    
    # Create and run the app
    app = create_trip_planning_app()
    
    # Run the workflow
    final_state = None
    for state in app.stream(initial_state):
        final_state = state
    
    return final_state


if __name__ == "__main__":
    # Example usage
    result = plan_trip(
        user_request="I need to travel from New York to Los Angeles for a business meeting",
        origin="NYC",
        destination="LAX",
        departure_date="2024-03-15",
        return_date="2024-03-17",
        budget=1500,
        preferences={
            "preferred_airline": None,
            "min_rating": 4.0,
            "meeting_times": ["2024-03-15 14:00"]
        }
    )
    
    if result:
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        # Get the last node's output
        for node_name, node_state in result.items():
            if "final_recommendation" in node_state:
                print(f"Recommendation: {node_state.get('final_recommendation')}")
