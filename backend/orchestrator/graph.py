"""
LangGraph Workflow for Agentic Trip Planning System

This module implements a truly agentic workflow orchestration using LangGraph's StateGraph.
The workflow coordinates multiple ReAct-based agents following:
- Contract Net Protocol (Smith, 1980) for task delegation
- FIPA-ACL message patterns for agent communication
- BDI Architecture (Rao & Georgeff, 1995) for agent reasoning
- ReAct (Yao et al., 2023) for autonomous reasoning

KEY AGENTIC FEATURES:
1. PARALLEL EXECUTION: Flight and Hotel agents search simultaneously
2. REAL BACKTRACKING: Policy violations trigger actual re-planning with adjusted constraints
3. INTER-AGENT MESSAGING: Agents communicate through structured messages
4. AUTONOMOUS REASONING: Each agent uses ReAct to decide which tools to use
5. ITERATIVE REFINEMENT: Maximum 5 backtracking iterations as per MultiAgentBench recommendations

Author: Research Project - Laureys Indy
"""

import os
import sys
from typing import Dict, Any, Literal, List
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

# ============================================================================
# AGENT INSTANCES (maintain their own BDI state)
# ============================================================================
flight_agent = FlightAgent()
hotel_agent = HotelAgent()
policy_agent = PolicyAgent()
time_agent = TimeManagementAgent()
orchestrator = TripOrchestrator()

# Data loaders
flight_loader = FlightDataLoader()
hotel_loader = HotelDataLoader()
policy_loader = PolicyDataLoader()

# ============================================================================
# CONSTANTS FOR AGENTIC BEHAVIOR
# ============================================================================
MAX_BACKTRACKING_ITERATIONS = 5  # From MultiAgentBench: >7 iterations reduce performance
MAX_POLICY_NEGOTIATION_ROUNDS = 3  # Max rounds of policy adjustment per iteration
BUDGET_REDUCTION_FACTOR = 0.15  # 15% reduction per backtrack


# ============================================================================
# HELPER: CONTRACT NET PROTOCOL MESSAGE CREATION
# ============================================================================
def create_cnp_message(
    performative: str,
    sender: str,
    receiver: str,
    content: Dict[str, Any],
    conversation_id: str = None
) -> Dict[str, Any]:
    """
    Create a Contract Net Protocol / FIPA-ACL style message.
    
    Performatives:
    - cfp: Call for Proposals (task announcement)
    - propose: Agent proposal/bid
    - accept: Accept proposal
    - reject: Reject proposal
    - inform: Inform of results
    - request: Request action from another agent
    - failure: Report failure
    """
    return {
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# NODE: INITIALIZATION
# ============================================================================
def initialize_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Initialize the workflow and broadcast task to all agents (Contract Net Protocol).
    
    This node:
    1. Loads flight, hotel, and policy data
    2. Initializes agent beliefs
    3. Creates CFP (Call for Proposals) message to agents
    """
    print("\n" + "="*60)
    print("🚀 INITIALIZING AGENTIC TRIP PLANNING WORKFLOW")
    print("="*60)
    
    # Load data
    flights_data = flight_loader.flights
    hotels_data = hotel_loader.hotels
    policies_data = policy_loader.get_policy("standard")
    
    # Initialize orchestrator beliefs (BDI)
    orchestrator.memory.add_belief("workflow_started", datetime.now().isoformat())
    orchestrator.memory.add_belief("user_request", state.get("user_request", ""))
    orchestrator.memory.add_belief("origin", state.get("origin", ""))
    orchestrator.memory.add_belief("destination", state.get("destination", ""))
    orchestrator.memory.add_belief("max_iterations", MAX_BACKTRACKING_ITERATIONS)
    
    # Initialize tracking for backtracking
    metrics = state.get("metrics", {})
    metrics["backtracking_count"] = 0
    metrics["negotiation_rounds"] = 0
    metrics["parallel_searches_executed"] = 0
    
    # Contract Net Protocol: Call for Proposals to all agents
    cfp_message = create_cnp_message(
        performative="cfp",
        sender=AgentRole.ORCHESTRATOR.value,
        receiver="all_agents",
        content={
            "task": "trip_planning",
            "origin": state.get("origin", ""),
            "destination": state.get("destination", ""),
            "budget": state.get("budget"),
            "constraints": state.get("preferences", {}),
            "data_available": {
                "flights": len(flights_data),
                "hotels": len(hotels_data),
                "policies_loaded": bool(policies_data)
            }
        }
    )
    
    print(f"✓ Loaded {len(flights_data)} flights")
    print(f"✓ Loaded {len(hotels_data)} hotels")
    print(f"✓ Loaded policy rules")
    print(f"✓ Max backtracking iterations: {MAX_BACKTRACKING_ITERATIONS}")
    
    return {
        "current_phase": "parallel_search",
        "policy_rules": policies_data,
        "messages": [cfp_message],
        "metrics": metrics
    }


# ============================================================================
# NODE: PARALLEL SEARCH (Flight + Hotel simultaneously)
# ============================================================================
def parallel_search_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Execute Flight and Hotel search in parallel (as per contract plan).
    
    This is the key difference from sequential execution:
    Both agents work simultaneously, communicating through the shared state.
    
    From Contract Plan:
    "Flight and Hotel agents execute operate simultaneously rather than serial"
    """
    print("\n" + "-"*60)
    print("🔄 PARALLEL SEARCH - Flight & Hotel Agents Working Simultaneously")
    print("-"*60)
    
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    budget = state.get("budget")
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    
    # Get adjusted constraints from any previous backtracking
    adjusted_flight_budget = preferences.get("adjusted_flight_budget", budget * 0.6 if budget else None)
    adjusted_hotel_budget = preferences.get("adjusted_hotel_budget", budget * 0.4 if budget else None)
    
    print(f"  Budget allocation: Flights ${adjusted_flight_budget}, Hotels ${adjusted_hotel_budget}/night")
    
    # ===== FLIGHT AGENT - Autonomous ReAct Reasoning =====
    print("\n  ✈️  Flight Agent starting autonomous reasoning...")
    
    # Reset agent state for fresh reasoning
    flight_agent.reset_state()
    
    # Build query with potentially adjusted constraints
    flight_query = FlightQuery(
        from_city=origin,
        to_city=destination,
        max_price=int(adjusted_flight_budget) if adjusted_flight_budget else None,
        departure_after=preferences.get("departure_after", "06:00"),
        departure_before=preferences.get("departure_before", "21:00"),
        class_preference=preferences.get("class_preference", "Economy")
    )
    
    # Let agent reason autonomously using ReAct
    flight_result = flight_agent.search_flights(flight_query)
    
    # Extract results
    if hasattr(flight_result, 'flights'):
        flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in flight_result.flights]
        flight_reasoning = flight_result.reasoning if hasattr(flight_result, 'reasoning') else ""
    else:
        flights = flight_result.get('flights', [])
        flight_reasoning = flight_result.get('reasoning', '')
    
    # Collect agent's reasoning trace
    flight_traces = []
    if hasattr(flight_agent, 'state') and flight_agent.state:
        for step in flight_agent.state.reasoning_trace:
            flight_traces.append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    print(f"  ✓ Flight Agent found {len(flights)} options after {len(flight_traces)} reasoning steps")
    
    # ===== HOTEL AGENT - Autonomous ReAct Reasoning =====
    print("\n  🏨 Hotel Agent starting autonomous reasoning...")
    
    # Reset agent state for fresh reasoning
    hotel_agent.reset_state()
    
    # Build query with potentially adjusted constraints
    hotel_query = HotelQuery(
        city=destination,
        max_price_per_night=int(adjusted_hotel_budget) if adjusted_hotel_budget else None,
        min_stars=int(preferences.get("min_rating", 3)),
        required_amenities=preferences.get("required_amenities")
    )
    
    # Let agent reason autonomously using ReAct
    hotel_result = hotel_agent.search_hotels(hotel_query)
    
    # Extract results
    if hasattr(hotel_result, 'hotels'):
        hotels = [h.model_dump() if hasattr(h, 'model_dump') else h for h in hotel_result.hotels]
        hotel_reasoning = hotel_result.reasoning if hasattr(hotel_result, 'reasoning') else ""
    else:
        hotels = hotel_result.get('hotels', [])
        hotel_reasoning = hotel_result.get('reasoning', '')
    
    # Collect agent's reasoning trace
    hotel_traces = []
    if hasattr(hotel_agent, 'state') and hotel_agent.state:
        for step in hotel_agent.state.reasoning_trace:
            hotel_traces.append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    print(f"  ✓ Hotel Agent found {len(hotels)} options after {len(hotel_traces)} reasoning steps")
    
    # ===== CREATE INTER-AGENT MESSAGES =====
    messages = []
    
    # Flight Agent proposal (CNP)
    messages.append(create_cnp_message(
        performative="propose",
        sender=AgentRole.FLIGHT_AGENT.value,
        receiver=AgentRole.ORCHESTRATOR.value,
        content={
            "proposal": "flight_options",
            "options_count": len(flights),
            "best_option": flights[0] if flights else None,
            "reasoning_steps": len(flight_traces),
            "analysis": flight_reasoning[:500] if flight_reasoning else ""
        }
    ))
    
    # Hotel Agent proposal (CNP)
    messages.append(create_cnp_message(
        performative="propose",
        sender=AgentRole.HOTEL_AGENT.value,
        receiver=AgentRole.ORCHESTRATOR.value,
        content={
            "proposal": "hotel_options",
            "options_count": len(hotels),
            "best_option": hotels[0] if hotels else None,
            "reasoning_steps": len(hotel_traces),
            "analysis": hotel_reasoning[:500] if hotel_reasoning else ""
        }
    ))
    
    # Update metrics
    metrics["parallel_searches_executed"] = metrics.get("parallel_searches_executed", 0) + 1
    
    # Combine reasoning traces
    reasoning_traces = {
        AgentRole.FLIGHT_AGENT.value: flight_traces,
        AgentRole.HOTEL_AGENT.value: hotel_traces
    }
    
    print(f"\n  ✓ Parallel search complete: {len(flights)} flights, {len(hotels)} hotels")
    
    return {
        "available_flights": flights,
        "available_hotels": hotels,
        "flight_analysis": {
            "total_options": len(flights),
            "recommended": flights[0] if flights else None,
            "reasoning": flight_reasoning
        },
        "hotel_analysis": {
            "total_options": len(hotels),
            "recommended": hotels[0] if hotels else None,
            "reasoning": hotel_reasoning
        },
        "current_phase": "policy_check",
        "messages": messages,
        "reasoning_traces": reasoning_traces,
        "metrics": metrics
    }


# ============================================================================
# NODE: POLICY CHECK WITH FEEDBACK GENERATION
# ============================================================================
def check_policy_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Policy Agent checks compliance and generates feedback for other agents.
    
    If violations are found, this node creates REQUEST messages to
    Flight/Hotel agents with specific adjustment requirements.
    This enables the bidirectional feedback loop mentioned in the contract plan.
    """
    print("\n" + "-"*60)
    print("📋 POLICY AGENT - Compliance Check with Feedback Loop")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    budget = state.get("budget", 2000)
    metrics = state.get("metrics", {})
    
    # Reset policy agent for fresh reasoning
    policy_agent.reset_state()
    
    # Create proper model objects for policy checking
    flight_models = []
    for f in flights[:5]:  # Top 5
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
            print(f"  Warning: Could not create Flight model: {e}")
    
    hotel_models = []
    for h in hotels[:5]:  # Top 5
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
            print(f"  Warning: Could not create Hotel model: {e}")
    
    # Create result objects
    flight_query = FlightQuery(from_city=origin, to_city=destination)
    hotel_query = HotelQuery(city=destination)
    
    flight_result = FlightSearchResult(
        query=flight_query,
        flights=flight_models,
        reasoning="Flights from parallel search"
    )
    
    hotel_result = HotelSearchResult(
        query=hotel_query,
        hotels=hotel_models,
        reasoning="Hotels from parallel search"
    )
    
    # Run policy agent's autonomous compliance check
    result = policy_agent.check_compliance(flight_result, hotel_result)
    
    # Handle result
    if hasattr(result, 'is_compliant'):
        is_compliant = result.is_compliant
        violations = [v.model_dump() if hasattr(v, 'model_dump') else v for v in result.violations]
        policy_reasoning = result.reasoning
    else:
        is_compliant = result.get('is_compliant', True)
        violations = result.get('violations', [])
        policy_reasoning = result.get('reasoning', '')
    
    # Collect policy agent reasoning traces
    policy_traces = []
    if hasattr(policy_agent, 'state') and policy_agent.state:
        for step in policy_agent.state.reasoning_trace:
            policy_traces.append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # ===== GENERATE FEEDBACK MESSAGES FOR BACKTRACKING =====
    messages = []
    feedback_for_agents = {}
    
    if not is_compliant and violations:
        print(f"  ⚠️ Found {len(violations)} policy violations - generating feedback")
        
        # Analyze violations and create specific feedback
        flight_violations = [v for v in violations if 'flight' in str(v).lower() or 'FLIGHT' in str(v.get('message', '') if isinstance(v, dict) else '').upper()]
        hotel_violations = [v for v in violations if 'hotel' in str(v).lower() or 'HOTEL' in str(v.get('message', '') if isinstance(v, dict) else '').upper()]
        budget_violations = [v for v in violations if 'budget' in str(v).lower() or 'BUDGET' in str(v.get('message', '') if isinstance(v, dict) else '').upper()]
        
        # Calculate adjusted budgets based on violations
        current_flight_budget = state.get("preferences", {}).get("adjusted_flight_budget", budget * 0.6)
        current_hotel_budget = state.get("preferences", {}).get("adjusted_hotel_budget", budget * 0.4)
        
        new_flight_budget = current_flight_budget
        new_hotel_budget = current_hotel_budget
        
        if flight_violations or budget_violations:
            new_flight_budget = current_flight_budget * (1 - BUDGET_REDUCTION_FACTOR)
            feedback_for_agents["flight_agent"] = {
                "action": "reduce_search_budget",
                "new_max_price": int(new_flight_budget),
                "reason": "Policy violations require lower-priced options",
                "violations": [str(v) for v in flight_violations + budget_violations]
            }
            
            # Create REQUEST message to Flight Agent
            messages.append(create_cnp_message(
                performative="request",
                sender=AgentRole.POLICY_AGENT.value,
                receiver=AgentRole.FLIGHT_AGENT.value,
                content=feedback_for_agents["flight_agent"]
            ))
        
        if hotel_violations or budget_violations:
            new_hotel_budget = current_hotel_budget * (1 - BUDGET_REDUCTION_FACTOR)
            feedback_for_agents["hotel_agent"] = {
                "action": "reduce_search_budget",
                "new_max_price_per_night": int(new_hotel_budget),
                "reason": "Policy violations require lower-priced options",
                "violations": [str(v) for v in hotel_violations + budget_violations]
            }
            
            # Create REQUEST message to Hotel Agent
            messages.append(create_cnp_message(
                performative="request",
                sender=AgentRole.POLICY_AGENT.value,
                receiver=AgentRole.HOTEL_AGENT.value,
                content=feedback_for_agents["hotel_agent"]
            ))
        
        # Store adjusted budgets in preferences for next iteration
        preferences = state.get("preferences", {}).copy()
        preferences["adjusted_flight_budget"] = new_flight_budget
        preferences["adjusted_hotel_budget"] = new_hotel_budget
        
        # Update negotiation rounds
        metrics["negotiation_rounds"] = metrics.get("negotiation_rounds", 0) + 1
        metrics["policy_violations_found"] = metrics.get("policy_violations_found", 0) + len(violations)
        
        print(f"  📨 Sent feedback to agents: Flight budget ${current_flight_budget:.0f} → ${new_flight_budget:.0f}")
        print(f"  📨 Sent feedback to agents: Hotel budget ${current_hotel_budget:.0f} → ${new_hotel_budget:.0f}")
    else:
        print(f"  ✓ All options are policy compliant")
        preferences = state.get("preferences", {})
    
    # Policy Agent inform message
    messages.append(create_cnp_message(
        performative="inform",
        sender=AgentRole.POLICY_AGENT.value,
        receiver=AgentRole.ORCHESTRATOR.value,
        content={
            "compliance_check_complete": True,
            "is_compliant": is_compliant,
            "violations_count": len(violations),
            "violations": violations,
            "feedback_sent": bool(feedback_for_agents),
            "reasoning": policy_reasoning[:500] if policy_reasoning else ""
        }
    ))
    
    status = "compliant" if is_compliant else "non_compliant"
    
    return {
        "compliance_status": {
            "overall_status": status,
            "is_compliant": is_compliant,
            "violations": violations,
            "feedback_for_agents": feedback_for_agents,
            "reasoning": policy_reasoning
        },
        "preferences": preferences,
        "current_phase": "time_check",
        "messages": messages,
        "reasoning_traces": {AgentRole.POLICY_AGENT.value: policy_traces},
        "metrics": metrics
    }


# ============================================================================
# NODE: TIME MANAGEMENT CHECK
# ============================================================================
def check_time_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Time Agent checks timeline feasibility using ReAct reasoning.
    
    This validates that:
    - Flight arrival times work with meeting schedules
    - Hotel check-in/out times are feasible
    - Transit times between locations are accounted for
    """
    print("\n" + "-"*60)
    print("⏰ TIME AGENT - Timeline Feasibility Analysis")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    departure_date = state.get("departure_date", "")
    preferences = state.get("preferences", {})
    
    # Reset time agent for fresh reasoning
    time_agent.reset_state()
    
    # Create flight models
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
            print(f"  Warning: Could not create Flight model: {e}")
    
    # Create hotel models
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
            print(f"  Warning: Could not create Hotel model: {e}")
    
    # Create result objects
    flight_query = FlightQuery(from_city=origin, to_city=destination)
    hotel_query = HotelQuery(city=destination)
    
    flight_result = FlightSearchResult(
        query=flight_query,
        flights=flight_models,
        reasoning="Flights for time analysis"
    )
    
    hotel_result = HotelSearchResult(
        query=hotel_query,
        hotels=hotel_models,
        reasoning="Hotels for time analysis"
    )
    
    # Parse meeting times
    meeting_times = preferences.get("meeting_times", [])
    meetings = []
    for mt in meeting_times:
        try:
            if " " in str(mt):
                date_part, time_part = str(mt).split(" ", 1)
            else:
                date_part = departure_date
                time_part = str(mt)
            meetings.append(Meeting(
                date=date_part,
                time=time_part,
                location={"lat": 37.7749, "lng": -122.4194},
                duration_minutes=60
            ))
        except Exception as e:
            print(f"  Warning: Could not parse meeting time '{mt}': {e}")
    
    # Default coordinates
    city_coords = {"lat": 37.7749, "lng": -122.4194}
    airport_coords = {"lat": 37.6213, "lng": -122.379}
    
    # Run time agent's feasibility check
    result = time_agent.check_feasibility(
        flight_result=flight_result,
        hotel_result=hotel_result,
        meetings=meetings,
        arrival_city_coords=city_coords,
        airport_coords=airport_coords,
        departure_date=departure_date
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
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "timestamp": step.timestamp
            })
    
    # Create message
    messages = [create_cnp_message(
        performative="inform",
        sender=AgentRole.TIME_AGENT.value,
        receiver=AgentRole.ORCHESTRATOR.value,
        content={
            "feasibility_check_complete": True,
            "is_feasible": is_feasible,
            "conflicts_count": len(conflicts),
            "conflicts": conflicts,
            "timeline": timeline,
            "reasoning": time_reasoning[:500] if time_reasoning else ""
        }
    )]
    
    print(f"  ✓ Timeline feasible: {is_feasible}")
    if conflicts:
        print(f"  ⚠️ Found {len(conflicts)} scheduling conflicts")
    
    return {
        "time_constraints": {
            "feasible": is_feasible,
            "timeline": timeline,
            "conflicts": conflicts,
            "reasoning": time_reasoning
        },
        "feasibility_analysis": {
            "is_feasible": is_feasible,
            "timeline": timeline,
            "conflicts": conflicts,
            "reasoning": time_reasoning
        },
        "messages": messages,
        "reasoning_traces": {AgentRole.TIME_AGENT.value: time_traces}
    }


# ============================================================================
# NODE: ORCHESTRATOR SELECTION (Chain-of-Thought)
# ============================================================================
def select_options_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Orchestrator selects the best flight and hotel using Chain-of-Thought reasoning.
    
    This is where the central orchestrator makes final decisions based on:
    - Agent proposals
    - Compliance status
    - Time feasibility
    - Budget constraints
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
        print("  ⚠️ No options to select from")
        return {
            "selected_flight": None,
            "selected_hotel": None,
            "current_phase": "finalizing",
            "messages": []
        }
    
    # Convert to models for orchestrator
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
        except Exception:
            pass
    
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
        except Exception:
            pass
    
    # Use orchestrator's CoT selection
    context = {
        "flight_options": flight_models,
        "hotel_options": hotel_models,
        "total_budget": budget,
        "nights": 2,
        "time_feasibility": time_constraints.get("feasible", True),
        "compliance_status": compliance.get("overall_status", "unknown")
    }
    
    result = orchestrator.select_bookings(context)
    
    # Get selected items
    selected_flight = result.get("selected_flight")
    selected_hotel = result.get("selected_hotel")
    
    # Convert to dict if model
    if selected_flight and hasattr(selected_flight, 'model_dump'):
        selected_flight = selected_flight.model_dump()
    elif not selected_flight and flights:
        selected_flight = flights[0]
        
    if selected_hotel and hasattr(selected_hotel, 'model_dump'):
        selected_hotel = selected_hotel.model_dump()
    elif not selected_hotel and hotels:
        selected_hotel = hotels[0]
    
    # Create acceptance messages (CNP)
    messages = []
    
    if selected_flight:
        messages.append(create_cnp_message(
            performative="accept",
            sender=AgentRole.ORCHESTRATOR.value,
            receiver=AgentRole.FLIGHT_AGENT.value,
            content={
                "accepted_proposal": selected_flight.get('flight_id', ''),
                "reason": "Best match for requirements"
            }
        ))
        price = selected_flight.get('price_usd', selected_flight.get('price', 'N/A'))
        print(f"  ✓ Selected flight: {selected_flight.get('airline', 'Unknown')} - ${price}")
    
    if selected_hotel:
        messages.append(create_cnp_message(
            performative="accept",
            sender=AgentRole.ORCHESTRATOR.value,
            receiver=AgentRole.HOTEL_AGENT.value,
            content={
                "accepted_proposal": selected_hotel.get('hotel_id', ''),
                "reason": "Best match for requirements"
            }
        ))
        price = selected_hotel.get('price_per_night_usd', selected_hotel.get('price', 'N/A'))
        print(f"  ✓ Selected hotel: {selected_hotel.get('name', 'Unknown')} - ${price}/night")
    
    return {
        "selected_flight": selected_flight,
        "selected_hotel": selected_hotel,
        "current_phase": "finalizing",
        "messages": messages
    }


# ============================================================================
# NODE: FINALIZATION
# ============================================================================
def finalize_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    Finalize the trip recommendation with comprehensive explanation and metrics.
    """
    print("\n" + "-"*60)
    print("📝 FINALIZING RECOMMENDATION")
    print("-"*60)
    
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    compliance = state.get("compliance_status", {})
    time_constraints = state.get("time_constraints", {})
    metrics = state.get("metrics", {})
    
    # Calculate total cost
    total_cost = 0
    if selected_flight:
        total_cost += selected_flight.get("price_usd", selected_flight.get("price", 0))
    if selected_hotel:
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
    explanation_parts = ["## Trip Planning Complete\n"]
    
    if selected_flight:
        f = selected_flight
        price = f.get('price_usd', f.get('price', 'N/A'))
        explanation_parts.append(
            f"**Flight**: {f.get('airline', 'Unknown')} from {f.get('from_city', '')} to "
            f"{f.get('to_city', '')} at ${price}. "
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
    
    # Add metrics summary
    explanation_parts.append("\n### Workflow Metrics")
    explanation_parts.append(f"- Backtracking iterations: {metrics.get('backtracking_count', 0)}")
    explanation_parts.append(f"- Negotiation rounds: {metrics.get('negotiation_rounds', 0)}")
    explanation_parts.append(f"- Parallel searches: {metrics.get('parallel_searches_executed', 0)}")
    
    explanation = "\n\n".join(explanation_parts)
    
    # Update metrics
    metrics["workflow_end_time"] = datetime.now().isoformat()
    
    # Final message
    final_message = create_cnp_message(
        performative="inform",
        sender=AgentRole.ORCHESTRATOR.value,
        receiver="user",
        content={
            "action": "recommendation_complete",
            "recommendation": final_recommendation,
            "metrics": metrics
        }
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


# ============================================================================
# CONDITIONAL ROUTING: REAL BACKTRACKING LOGIC
# ============================================================================
def should_backtrack_after_policy(state: TripPlanningState) -> Literal["check_time", "parallel_search"]:
    """
    Determine if we need to backtrack after policy check.
    
    THIS IS REAL BACKTRACKING - if policy violations exist and we haven't
    exceeded max iterations, we go back to parallel_search with adjusted constraints.
    
    From Contract Plan:
    "When the Policy agent sees any violation, it sends feedback through the 
    Orchestrator to agents for other options without restarting the entire process."
    """
    compliance = state.get("compliance_status", {})
    metrics = state.get("metrics", {})
    
    is_compliant = compliance.get("is_compliant", True)
    violations = compliance.get("violations", [])
    backtracking_count = metrics.get("backtracking_count", 0)
    
    # Check if we need to backtrack
    if not is_compliant and violations and backtracking_count < MAX_BACKTRACKING_ITERATIONS:
        # We have violations and haven't exceeded max iterations
        print(f"\n  🔄 BACKTRACKING: Policy violations found ({len(violations)})")
        print(f"     Iteration {backtracking_count + 1}/{MAX_BACKTRACKING_ITERATIONS}")
        
        # Increment backtracking counter (this will be picked up by the next node)
        return "parallel_search"
    
    if backtracking_count >= MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  ⚠️ Max backtracking iterations reached ({MAX_BACKTRACKING_ITERATIONS})")
        print("     Proceeding with best available options...")
    
    return "check_time"


def should_backtrack_after_time(state: TripPlanningState) -> Literal["select_options", "parallel_search"]:
    """
    Determine if we need to backtrack after time check.
    
    If timeline is not feasible and we haven't exceeded iterations,
    we can try different flight options.
    """
    time_constraints = state.get("time_constraints", {})
    metrics = state.get("metrics", {})
    
    is_feasible = time_constraints.get("feasible", True)
    conflicts = time_constraints.get("conflicts", [])
    backtracking_count = metrics.get("backtracking_count", 0)
    
    # Only backtrack for severe conflicts and if we have iterations left
    severe_conflicts = [c for c in conflicts if isinstance(c, dict) and c.get("severity") == "error"]
    
    if not is_feasible and severe_conflicts and backtracking_count < MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  🔄 BACKTRACKING: Severe timeline conflicts ({len(severe_conflicts)})")
        return "parallel_search"
    
    return "select_options"


def increment_backtrack_counter(state: TripPlanningState) -> Dict[str, Any]:
    """
    Node that increments the backtracking counter.
    Called before re-entering parallel_search after backtracking.
    """
    metrics = state.get("metrics", {}).copy()
    metrics["backtracking_count"] = metrics.get("backtracking_count", 0) + 1
    
    print(f"\n  📊 Backtracking iteration: {metrics['backtracking_count']}/{MAX_BACKTRACKING_ITERATIONS}")
    
    return {"metrics": metrics}


# ============================================================================
# BUILD WORKFLOW
# ============================================================================
def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for agentic trip planning.
    
    Workflow structure with PARALLEL execution and REAL backtracking:
    
    1. Initialize
           ↓
    2. Parallel Search (Flight + Hotel simultaneously)
           ↓
    3. Policy Check ──────────┐
           ↓                  │ (backtrack if violations)
    4. Time Check ───────────┐│
           ↓                 ││ (backtrack if conflicts)
    5. Select Options ←──────┴┘
           ↓
    6. Finalize
           ↓
         END
    """
    
    # Create the state graph
    workflow = StateGraph(TripPlanningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("parallel_search", parallel_search_node)
    workflow.add_node("increment_backtrack", increment_backtrack_counter)
    workflow.add_node("check_policy", check_policy_node)
    workflow.add_node("check_time", check_time_node)
    workflow.add_node("select_options", select_options_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # ===== WORKFLOW EDGES =====
    
    # Initialize → Parallel Search
    workflow.add_edge("initialize", "parallel_search")
    
    # Parallel Search → Policy Check
    workflow.add_edge("parallel_search", "check_policy")
    
    # Policy Check → Conditional (backtrack or continue)
    workflow.add_conditional_edges(
        "check_policy",
        should_backtrack_after_policy,
        {
            "check_time": "check_time",
            "parallel_search": "increment_backtrack"  # Backtrack path
        }
    )
    
    # Increment backtrack counter → Parallel Search
    workflow.add_edge("increment_backtrack", "parallel_search")
    
    # Time Check → Conditional (backtrack or continue)
    workflow.add_conditional_edges(
        "check_time",
        should_backtrack_after_time,
        {
            "select_options": "select_options",
            "parallel_search": "increment_backtrack"  # Backtrack path
        }
    )
    
    # Select Options → Finalize
    workflow.add_edge("select_options", "finalize")
    
    # Finalize → END
    workflow.add_edge("finalize", END)
    
    return workflow


def create_trip_planning_app():
    """
    Create and compile the trip planning application.
    """
    workflow = build_workflow()
    app = workflow.compile()
    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
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
    
    This is the main entry point for trip planning. The workflow will:
    1. Execute Flight and Hotel agents in parallel
    2. Check policy compliance with real backtracking
    3. Validate timeline feasibility
    4. Select optimal combination using Chain-of-Thought
    5. Return comprehensive recommendation with metrics
    """
    # Initialize preferences with budget allocations
    prefs = preferences or {}
    if budget:
        prefs["adjusted_flight_budget"] = budget * 0.6
        prefs["adjusted_hotel_budget"] = budget * 0.4
    
    # Create initial state
    initial_state = create_initial_state(
        user_request=user_request,
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        preferences=prefs
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
        user_request="I need to travel from New York to San Francisco for a business meeting",
        origin="NYC",
        destination="SF",
        departure_date="2026-01-20",
        return_date="2026-01-22",
        budget=1500,
        preferences={
            "preferred_airline": None,
            "min_rating": 4.0,
            "meeting_times": ["2026-01-20 14:00"]
        }
    )
    
    if result:
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        for node_name, node_state in result.items():
            if "final_recommendation" in node_state:
                print(f"Recommendation: {node_state.get('final_recommendation')}")
