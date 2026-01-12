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
2. REAL BACKTRACKING: Budget constraints trigger actual re-planning with adjusted allocations
3. INTER-AGENT MESSAGING: Agents communicate through structured messages
4. AUTONOMOUS REASONING: Each agent uses ReAct to decide which tools to use
5. ITERATIVE REFINEMENT: Maximum 5 backtracking iterations as per MultiAgentBench recommendations

Author: Research Project - Laureys Indy
"""

import os
import sys
import time
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
from agents.time_agent import TimeManagementAgent
from agents.policy_agent import PolicyComplianceAgent
from agents.models import (
    FlightQuery, HotelQuery, FlightSearchResult, HotelSearchResult, 
    Flight, Hotel, Meeting, TimeCheckResult
)
from data.loaders import FlightDataLoader, HotelDataLoader

# === AGENT INSTANCES ===
# Using qwen2.5:14b for better tool-calling accuracy
MODEL_NAME = "qwen2.5:14b"
flight_agent = FlightAgent(model_name=MODEL_NAME)
hotel_agent = HotelAgent(model_name=MODEL_NAME)
time_agent = TimeManagementAgent(model_name=MODEL_NAME)
policy_agent = PolicyComplianceAgent(model_name=MODEL_NAME)
orchestrator = TripOrchestrator(model_name=MODEL_NAME)
flight_loader = FlightDataLoader()
hotel_loader = HotelDataLoader()

# === CONSTANTS ===
# Safety caps only - actual termination is decided by PolicyAgent LLM reasoning
MAX_BACKTRACKING_ITERATIONS = 10
MAX_NEGOTIATION_ROUNDS = 10


# === HELPER FUNCTIONS ===
def _calculate_nights(state_or_prefs: Dict) -> int:
    """Calculate nights from departure/return dates or check-in/check-out."""
    # Try state-level departure/return dates first
    dep = state_or_prefs.get('departure_date')
    ret = state_or_prefs.get('return_date')
    if dep and ret:
        try:
            dep_dt = datetime.strptime(dep, '%Y-%m-%d')
            ret_dt = datetime.strptime(ret, '%Y-%m-%d')
            return max(1, (ret_dt - dep_dt).days)
        except:
            pass
    
    # Fall back to preferences hotel_checkin/checkout
    prefs = state_or_prefs.get('preferences', state_or_prefs)
    if prefs.get('hotel_checkin') and prefs.get('hotel_checkout'):
        try:
            checkin = datetime.strptime(prefs['hotel_checkin'], '%Y-%m-%d')
            checkout = datetime.strptime(prefs['hotel_checkout'], '%Y-%m-%d')
            return max(1, (checkout - checkin).days)
        except:
            pass
    return 1


def _dict_to_flight(f: Dict, origin: str = '', dest: str = '') -> Flight:
    """Convert flight dict to Flight model."""
    return Flight(
        flight_id=f.get('flight_id', ''),
        airline=f.get('airline', ''),
        from_city=f.get('from_city', origin),
        to_city=f.get('to_city', dest),
        departure_time=f.get('departure_time', '09:00'),
        arrival_time=f.get('arrival_time', '12:00'),
        duration_hours=f.get('duration_hours', 3.0),
        price_usd=f.get('price_usd', 0),
        seats_available=f.get('seats_available', 10),
        **{'class': f.get('class', f.get('flight_class', 'Economy'))}
    )


def _dict_to_hotel(h: Dict, dest: str = '') -> Hotel:
    """Convert hotel dict to Hotel model."""
    # Extract coordinates, handling both 'lon' and 'lng' formats
    coords = h.get('coordinates', {'lat': 37.7749, 'lon': -122.4194})
    if 'lng' in coords and 'lon' not in coords:
        coords['lon'] = coords['lng']
    
    return Hotel(
        hotel_id=h.get('hotel_id', ''),
        name=h.get('name', ''),
        city=h.get('city', dest),
        city_name=h.get('city_name', dest),
        business_area=h.get('business_area', ''),
        tier=h.get('tier', 'standard'),
        stars=h.get('stars', 3),
        price_per_night_usd=h.get('price_per_night_usd', 0),
        distance_to_business_center_km=h.get('distance_to_business_center_km', 1.0),
        distance_to_airport_km=h.get('distance_to_airport_km', 10.0),
        amenities=h.get('amenities', []),
        rooms_available=h.get('rooms_available', 5),
        coordinates=coords
    )


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


# === NODE: INITIALIZATION ===
def initialize_node(state: TripPlanningState) -> Dict[str, Any]:
    """Initialize workflow and broadcast CFP to agents."""
    print("\n" + "="*60)
    print("🚀 INITIALIZING AGENTIC TRIP PLANNING WORKFLOW")
    print("="*60)
    
    # Load data
    flights_data = flight_loader.flights
    hotels_data = hotel_loader.hotels
    budget = state.get("budget", 2000)
    
    # Initialize orchestrator beliefs (BDI)
    orchestrator.memory.add_belief("workflow_started", datetime.now().isoformat())
    orchestrator.memory.add_belief("user_request", state.get("user_request", ""))
    orchestrator.memory.add_belief("origin", state.get("origin", ""))
    orchestrator.memory.add_belief("destination", state.get("destination", ""))
    orchestrator.memory.add_belief("max_iterations", MAX_BACKTRACKING_ITERATIONS)
    orchestrator.memory.add_belief("total_budget", budget)
    
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
            "budget": budget,
            "constraints": state.get("preferences", {}),
            "data_available": {
                "flights": len(flights_data),
                "hotels": len(hotels_data)
            }
        }
    )
    
    print(f"✓ Loaded {len(flights_data)} flights")
    print(f"✓ Loaded {len(hotels_data)} hotels")
    print(f"✓ Total budget: ${budget}")
    print(f"✓ Max backtracking iterations: {MAX_BACKTRACKING_ITERATIONS}")
    
    return {
        "current_phase": "parallel_search",
        "messages": [cfp_message],
        "metrics": metrics
    }


# === NODE: PARALLEL SEARCH ===
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
    
    print(f"  Total budget: ${budget} (PolicyAgent will allocate later)")
    print(f"  Strategy: Search all options, then find best combination within budget")
    
    # ===== FLIGHT AGENT - Search WITHOUT budget constraint =====
    print("\n  ✈️  Flight Agent searching (no budget filter)...")
    
    # Reset agent state for fresh reasoning
    flight_agent.reset_state()
    
    # Build query WITHOUT max_price - let PolicyAgent decide allocation
    flight_query = FlightQuery(
        from_city=origin,
        to_city=destination,
        max_price=None,  # No budget constraint - search all options
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
    
    # ===== HOTEL AGENT - Search WITHOUT budget constraint =====
    print("\n  🏨 Hotel Agent searching (no budget filter)...")
    
    # Reset agent state for fresh reasoning
    hotel_agent.reset_state()
    
    # Build query WITHOUT max_price - let PolicyAgent decide allocation
    # Extract meeting location if available for proximity-based hotel selection
    meeting_location = None
    meetings = state.get("meetings", [])
    if meetings and len(meetings) > 0:
        first_meeting = meetings[0]
        if isinstance(first_meeting, dict) and "location" in first_meeting:
            meeting_location = first_meeting["location"]  # {"lat": float, "lon": float}
    
    hotel_query = HotelQuery(
        city=destination,
        max_price_per_night=None,  # No budget constraint - search all options
        min_stars=None,  # Let agent reason about quality - no hardcoded minimum
        required_amenities=preferences.get("required_amenities"),
        meeting_location=meeting_location  # Pass meeting coords for proximity scoring
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
        "current_phase": "budget_check",
        "messages": messages,
        "reasoning_traces": reasoning_traces,
        "metrics": metrics
    }


# === NODE: CNP NEGOTIATION ===
def negotiation_node(state: TripPlanningState) -> Dict[str, Any]:
    """CNP negotiation loop - PolicyAgent provides feedback, booking agents refine."""
    print("\n" + "-"*60)
    print("🤝 CNP NEGOTIATION - Policy ↔ Booking Agents")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    budget = state.get("budget", 2000)
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    
    # IMPORTANT: Accumulate messages from state, not fresh list
    messages = list(state.get("messages", []))
    messages_before = len(messages)
    
    negotiation_round = metrics.get("negotiation_rounds", 0)
    nights = _calculate_nights(state)
    
    # Detect quality upgrade scenario (budget underutilized)
    compliance = state.get("compliance_status", {})
    total_cost = compliance.get("total_cost", 0)
    budget_remaining = compliance.get("budget_remaining", 0)
    is_quality_upgrade = budget_remaining > 0 and (total_cost / budget * 100) < 50 if budget > 0 else False
    
    if is_quality_upgrade:
        print(f"  💎 QUALITY UPGRADE NEGOTIATION")
        print(f"     Current: ${total_cost:.0f}, Budget: ${budget:.0f}")
        print(f"     Seeking higher quality options with ${budget_remaining:.0f} extra budget...")
        metrics["quality_upgrade_attempted"] = True
    
    # Track feedback history for context
    feedback_history = metrics.get("feedback_history", [])
    
    # Check if we've exceeded max negotiation rounds
    if negotiation_round >= MAX_NEGOTIATION_ROUNDS:
        print(f"  ⚠️  Max negotiation rounds ({MAX_NEGOTIATION_ROUNDS}) reached")
        print(f"  Proceeding with best available options...")
        return {
            "current_phase": "policy_final",
            "metrics": metrics,
            "messages": messages
        }
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  Negotiation Round: {negotiation_round + 1}/{MAX_NEGOTIATION_ROUNDS} @ {timestamp}")
    print(f"  Current proposals: {len(flights)} flights, {len(hotels)} hotels")
    
    # Track previous min cost for convergence detection
    previous_min_cost = metrics.get("previous_min_cost", None)
    
    # Step 1: PolicyAgent uses LLM to reason about feedback (with convergence context)
    step_start = time.time()
    print(f"  [PolicyAgent] Reasoning about proposals...")
    feedback = policy_agent.generate_feedback(
        flights=flights,
        hotels=hotels,
        budget=budget,
        nights=nights,
        negotiation_round=negotiation_round,
        feedback_history=feedback_history,
        previous_min_cost=previous_min_cost
    )
    print(f"  [PolicyAgent] Done ({time.time() - step_start:.1f}s)")
    
    # Track current min cost for next round's convergence check
    if feedback.get("current_min_cost"):
        metrics["previous_min_cost"] = feedback["current_min_cost"]
    
    # Display LLM reasoning
    if feedback.get("reasoning"):
        print(f"  💭 PolicyAgent reasoning: {feedback.get('reasoning', '')[:150]}...")
    
    # Check if PolicyAgent decided to accept proposals
    if not feedback.get("needs_refinement"):
        print(f"  ✅ PolicyAgent: Proposals accepted - proceeding to selection")
        
        # Count messages added in this round
        new_messages = len(messages) - messages_before
        metrics["message_exchanges"] = len(messages)
        
        # CRITICAL: Mark that negotiation has converged (best effort accepted)
        # This prevents infinite loop when budget is still exceeded but no improvement possible
        metrics["negotiation_converged"] = True
        
        return {
            "current_phase": "policy_final",
            "metrics": metrics,
            "messages": messages
        }
    
    print(f"  📣 PolicyAgent requests refinement from booking agents")
    
    # Increment negotiation round
    metrics["negotiation_rounds"] = negotiation_round + 1
    
    # Step 2: Send feedback to booking agents and get refined proposals
    refined_flights = flights  # Default: keep current
    refined_hotels = hotels
    
    # Track if any feedback was actually given
    feedback_given = False
    
    # Flight Agent refinement (if feedback provided)
    if feedback.get("flight_feedback"):
        feedback_given = True
        flight_feedback = feedback["flight_feedback"]
        issue = flight_feedback.get('issue', 'general') or 'general'
        reason = flight_feedback.get('reasoning', 'Refinement needed') or 'Refinement needed'
        
        # Display the actual CNP message
        print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
        print(f"  │ From: PolicyAgent → To: FlightAgent")
        print(f"  │ Performative: REJECT")
        print(f"  │ Issue: {issue}")
        print(f"  │ Reason: {reason[:80]}..." if len(reason) > 80 else f"  │ Reason: {reason}")
        print(f"  └────────────────────────────────────────────────────────")
        
        # Create rejection message (CNP)
        messages.append(create_cnp_message(
            performative="reject",
            sender=AgentRole.POLICY_AGENT.value,
            receiver=AgentRole.FLIGHT_AGENT.value,
            content={
                "rejection_reason": flight_feedback.get("issue"),
                "feedback": flight_feedback,
                "round": negotiation_round + 1
            }
        ))
        
        # FlightAgent reasons about feedback and refines proposal
        refine_start = time.time()
        print(f"  [FlightAgent] Refining proposal...")
        flight_result = flight_agent.refine_proposal(
            feedback=flight_feedback,
            previous_flights=flights
        )
        print(f"  [FlightAgent] Done ({time.time() - refine_start:.1f}s)")
        
        if flight_result.flights:
            new_flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in flight_result.flights]
            
            # MERGE with original flights to preserve fallback options
            # Use set to track IDs and avoid duplicates
            seen_ids = {f.get('flight_id') for f in new_flights}
            for orig_f in flights:
                if orig_f.get('flight_id') not in seen_ids:
                    new_flights.append(orig_f)
                    seen_ids.add(orig_f.get('flight_id'))
            
            refined_flights = new_flights
            
            # Display the CNP response message
            print(f"  ┌─ CNP MESSAGE ─────────────────────────────────────────")
            print(f"  │ From: FlightAgent → To: PolicyAgent")
            print(f"  │ Performative: PROPOSE (refined)")
            print(f"  │ Options: {len(refined_flights)} flights (merged with originals)")
            print(f"  │ Addressing: {flight_feedback.get('issue')}")
            print(f"  └────────────────────────────────────────────────────────")
            
            # Create proposal message (CNP)
            messages.append(create_cnp_message(
                performative="propose",
                sender=AgentRole.FLIGHT_AGENT.value,
                receiver=AgentRole.POLICY_AGENT.value,
                content={
                    "proposal_type": "refined",
                    "options_count": len(refined_flights),
                    "addressing_issue": flight_feedback.get("issue"),
                    "round": negotiation_round + 1
                }
            ))
    
    # Hotel Agent refinement (if feedback provided)
    if feedback.get("hotel_feedback"):
        feedback_given = True
        hotel_feedback = feedback["hotel_feedback"]
        h_issue = hotel_feedback.get('issue', 'general') or 'general'
        h_reason = hotel_feedback.get('reasoning', 'Refinement needed') or 'Refinement needed'
        
        # Display the actual CNP message
        print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
        print(f"  │ From: PolicyAgent → To: HotelAgent")
        print(f"  │ Performative: REJECT")
        print(f"  │ Issue: {h_issue}")
        print(f"  │ Reason: {h_reason[:80]}..." if len(h_reason) > 80 else f"  │ Reason: {h_reason}")
        print(f"  └────────────────────────────────────────────────────────")
        
        # Create rejection message (CNP)
        messages.append(create_cnp_message(
            performative="reject",
            sender=AgentRole.POLICY_AGENT.value,
            receiver=AgentRole.HOTEL_AGENT.value,
            content={
                "rejection_reason": hotel_feedback.get("issue"),
                "feedback": hotel_feedback,
                "round": negotiation_round + 1
            }
        ))
        
        # HotelAgent reasons about feedback and refines proposal
        refine_start = time.time()
        print(f"  [HotelAgent] Refining proposal...")
        hotel_result = hotel_agent.refine_proposal(
            feedback=hotel_feedback,
            previous_hotels=hotels
        )
        print(f"  [HotelAgent] Done ({time.time() - refine_start:.1f}s)")
        
        if hotel_result.hotels:
            new_hotels = [h.model_dump() if hasattr(h, 'model_dump') else h for h in hotel_result.hotels]
            
            # MERGE with original hotels to preserve fallback options
            # Use set to track IDs and avoid duplicates
            seen_ids = {h.get('hotel_id') for h in new_hotels}
            for orig_h in hotels:
                if orig_h.get('hotel_id') not in seen_ids:
                    new_hotels.append(orig_h)
                    seen_ids.add(orig_h.get('hotel_id'))
            
            refined_hotels = new_hotels
            
            # Display the CNP response message
            print(f"  ┌─ CNP MESSAGE ─────────────────────────────────────────")
            print(f"  │ From: HotelAgent → To: PolicyAgent")
            print(f"  │ Performative: PROPOSE (refined)")
            print(f"  │ Options: {len(refined_hotels)} hotels (merged with originals)")
            print(f"  │ Addressing: {hotel_feedback.get('issue')}")
            print(f"  └────────────────────────────────────────────────────────")
            
            # Create proposal message (CNP)
            messages.append(create_cnp_message(
                performative="propose",
                sender=AgentRole.HOTEL_AGENT.value,
                receiver=AgentRole.POLICY_AGENT.value,
                content={
                    "proposal_type": "refined",
                    "options_count": len(refined_hotels),
                    "addressing_issue": hotel_feedback.get("issue"),
                    "round": negotiation_round + 1
                }
            ))
    
    # If needs_refinement was True but no specific feedback was given, add a status message
    if not feedback_given:
        messages.append(create_cnp_message(
            performative="inform",
            sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={
                "status": "review_ongoing",
                "round": negotiation_round + 1,
                "reasoning": feedback.get("reasoning", "Continuing negotiation")[:200]
            }
        ))
        print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
        print(f"  │ From: PolicyAgent → To: all_agents")
        print(f"  │ Performative: INFORM (review_ongoing)")
        print(f"  │ Status: Continuing negotiation round {negotiation_round + 1}")
        print(f"  └────────────────────────────────────────────────────────")
    
    # Track message exchanges - count new messages this round
    new_messages_this_round = len(messages) - messages_before
    metrics["message_exchanges"] = len(messages)
    
    # Update feedback history for context in future rounds
    if feedback.get("flight_feedback"):
        issue = feedback["flight_feedback"].get("issue")
        if issue and issue not in feedback_history:
            feedback_history.append(issue)
    if feedback.get("hotel_feedback"):
        issue = feedback["hotel_feedback"].get("issue")
        if issue and issue not in feedback_history:
            feedback_history.append(issue)
    metrics["feedback_history"] = feedback_history
    
    print(f"\n  📊 Negotiation round {negotiation_round + 1} complete")
    print(f"     New messages this round: {new_messages_this_round}")
    print(f"     Total messages: {len(messages)}")
    print(f"     Refined flights: {len(refined_flights)}")
    print(f"     Refined hotels: {len(refined_hotels)}")
    
    return {
        "available_flights": refined_flights,
        "available_hotels": refined_hotels,
        "current_phase": "negotiation",  # Loop back for another round
        "messages": messages,
        "metrics": metrics
    }


def should_continue_negotiation(state: TripPlanningState) -> Literal["negotiation", "check_policy"]:
    """
    CNP NEGOTIATION ROUTING: Always verify refined proposals with PolicyAgent.
    
    Key insight: After booking agents refine their proposals, we MUST go back to
    check_policy to verify if the refinement actually solved the budget issue.
    
    Flow:
    - policy_final → proposals accepted by PolicyAgent, verify with check_policy
    - negotiation phase → more rounds needed, but FIRST verify current refinements
    - max rounds → verify best effort with check_policy
    """
    current_phase = state.get("current_phase", "")
    metrics = state.get("metrics", {})
    negotiation_rounds = metrics.get("negotiation_rounds", 0)
    
    print(f"\n  🔀 Negotiation routing: phase='{current_phase}', rounds={negotiation_rounds}/{MAX_NEGOTIATION_ROUNDS}")
    
    # CRITICAL FIX: After any refinement, ALWAYS go to check_policy to verify
    # This ensures we don't loop infinitely without checking if budget is now satisfied
    
    if current_phase == "policy_final":
        # PolicyAgent accepted - verify the selection
        print(f"  ✅ Proposals accepted - verifying with PolicyAgent")
        return "check_policy"
    
    if negotiation_rounds >= MAX_NEGOTIATION_ROUNDS:
        # Max rounds - verify best effort
        print(f"  ⚠️  Max rounds reached - final verification with PolicyAgent")
        return "check_policy"
    
    # Still negotiating - go back to check_policy to see if refinements worked
    # If budget is now satisfied, routing will proceed to time_check
    # If still over budget, routing will come back to negotiation
    print(f"  🔄 Verifying refined proposals with PolicyAgent")
    return "check_policy"


# === NODE: POLICY CHECK ===
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
    nights = _calculate_nights(state)  # Pass full state for date access
    
    print(f"  Budget: ${budget}")
    print(f"  Nights: {nights}")
    print(f"  Options: {len(flights)} flights × {len(hotels)} hotels")
    
    # Use PolicyAgent to find the best combination
    combination_result = policy_agent.find_best_combination(
        flights=flights,
        hotels=hotels,
        budget=budget,
        nights=nights,
        preferences=preferences
    )
    
    # Track metrics
    metrics["combinations_evaluated"] = combination_result.combinations_evaluated
    
    if combination_result.success:
        selected_flight = combination_result.selected_flight
        selected_hotel = combination_result.selected_hotel
        cheaper_alternatives = combination_result.cheaper_alternatives
        
        print(f"\n  ✅ OPTIMAL COMBINATION FOUND (Maximizing Budget):")
        print(f"     Flight: {selected_flight.get('airline', 'Unknown')} - ${selected_flight.get('price_usd', 0)}")
        print(f"     Hotel: {selected_hotel.get('name', 'Unknown')} ({selected_hotel.get('stars', '?')}★) - ${selected_hotel.get('price_per_night_usd', 0)}/night")
        print(f"     Total: ${combination_result.total_cost} (${combination_result.budget_remaining} remaining)")
        
        # Show hotel alternatives if available
        if cheaper_alternatives:
            print(f"\n  🏨 HOTEL ALTERNATIVES:")
            for alt in cheaper_alternatives:
                category = alt.get('category', '🏨')
                hotel = alt.get('hotel', {})
                vs = alt.get('vs_selected', 0)
                vs_str = f"+${vs:.0f}" if vs > 0 else f"Save ${abs(vs):.0f}"
                print(f"     {category}: {hotel.get('name', 'Unknown')} ({hotel.get('stars', '?')}★)")
                print(f"        ${alt['total_cost']:.0f} ({vs_str}) - {alt.get('reasoning', '')}")
        
        print(f"\n  💭 Reasoning: {combination_result.reasoning[:200]}...")
        
        # Create success message
        messages = [create_cnp_message(
            performative="inform",
            sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={
                "policy_check_complete": True,
                "is_compliant": True,
                "selected_flight": selected_flight,
                "selected_hotel": selected_hotel,
                "total_cost": combination_result.total_cost,
                "budget_remaining": combination_result.budget_remaining,
                "combinations_evaluated": combination_result.combinations_evaluated,
                "cheaper_alternatives": cheaper_alternatives
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
        # No valid combinations found
        print(f"\n  ❌ NO VALID COMBINATIONS WITHIN BUDGET")
        print(f"     {combination_result.reasoning}")
        
        metrics["policy_feedback_loops"] = metrics.get("policy_feedback_loops", 0) + 1
        
        # Create feedback message
        messages = [create_cnp_message(
            performative="inform",
            sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={
                "policy_check_complete": True,
                "is_compliant": False,
                "reason": combination_result.reasoning,
                "combinations_evaluated": combination_result.combinations_evaluated
            }
        )]
        
        return {
            "compliance_status": {
                "overall_status": "non_compliant",
                "is_compliant": False,
                "violations": [{
                    "type": "no_valid_combination",
                    "message": combination_result.reasoning,
                    "severity": "error"
                }],
                "total_cost": 0,
                "budget": budget,
                "reasoning": combination_result.reasoning,
                "combinations_evaluated": combination_result.combinations_evaluated
            },
            "current_phase": "time_check",
            "messages": messages,
            "metrics": metrics
        }


# === NODE: TIME CHECK ===
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
    
    # Build flight models using helper
    flight_models = []
    if selected_flight:
        try:
            flight_models.append(_dict_to_flight(selected_flight, origin, destination))
        except Exception as e:
            print(f"  Warning: Could not create Flight model: {e}")
    if not flight_models:
        for f in available_flights[:3]:
            try:
                flight_models.append(_dict_to_flight(f, origin, destination))
            except:
                pass
    
    # Build hotel models using helper
    hotel_models = []
    if selected_hotel:
        try:
            hotel_models.append(_dict_to_hotel(selected_hotel, destination))
        except Exception as e:
            print(f"  Warning: Could not create Hotel model: {e}")
    if not hotel_models:
        for h in available_hotels[:3]:
            try:
                hotel_models.append(_dict_to_hotel(h, destination))
            except:
                pass
    
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
    
    # Parse meeting times - use actual meeting location from preferences
    meeting_times = preferences.get("meeting_times", [])
    meeting_location = preferences.get("meeting_location", {"lat": 37.7749, "lon": -122.4194})  # Default to SF
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
                location=meeting_location,  # Use the actual meeting location
                duration_minutes=60
            ))
        except Exception as e:
            print(f"  Warning: Could not parse meeting time '{mt}': {e}")
    
    # City/airport coordinates for transit calculation
    # Import the coordinate lookup helpers
    from utils.routing import get_airport_coords, get_city_center_coords, geocode_address
    
    # Get proper airport coordinates for destination city
    airport_coords = get_airport_coords(destination)
    city_coords = get_city_center_coords(destination)
    
    # If meeting location is provided as an address/name, geocode it
    if isinstance(meeting_location, str):
        geocoded = geocode_address(meeting_location, destination)
        if geocoded:
            city_coords = geocoded
    elif isinstance(meeting_location, dict) and meeting_location.get("lat"):
        city_coords = meeting_location
    
    # Run time agent's feasibility check with destination city for proper routing
    result = time_agent.check_feasibility(
        flight_result=flight_result,
        hotel_result=hotel_result,
        meetings=meetings,
        arrival_city_coords=city_coords,
        airport_coords=airport_coords,
        departure_date=departure_date,
        destination_city=destination  # Pass destination for proper coordinate lookup
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


# === NODE: TIME POLICY FEEDBACK ===
def time_policy_feedback_node(state: TripPlanningState) -> Dict[str, Any]:
    """
    TimeAgent reports conflicts to PolicyAgent, which then requests better flight options.
    This enables inter-agent communication where timeline issues trigger flight re-selection.
    """
    print("\n" + "-"*60)
    print("⏰ TIME→POLICY FEEDBACK - Requesting Better Flight Options")
    print("-"*60)
    
    time_constraints = state.get("time_constraints", {})
    conflicts = time_constraints.get("conflicts", [])
    available_flights = state.get("available_flights", [])
    available_hotels = state.get("available_hotels", [])
    selected_flight = state.get("selected_flight", {})
    selected_hotel = state.get("selected_hotel", {})
    budget = state.get("budget", 2000)
    preferences = state.get("preferences", {})
    metrics = state.get("metrics", {})
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    nights = _calculate_nights(state)
    
    messages = list(state.get("messages", []))
    
    # Increment time feedback counter
    time_feedback_count = metrics.get("time_feedback_count", 0) + 1
    metrics["time_feedback_count"] = time_feedback_count
    
    print(f"  Time feedback iteration: {time_feedback_count}/{MAX_BACKTRACKING_ITERATIONS}")
    
    # Extract conflict details for PolicyAgent
    conflict_details = []
    for c in conflicts:
        if isinstance(c, dict):
            conflict_details.append(c.get("message", str(c)))
        else:
            conflict_details.append(str(c))
    
    print(f"  Conflicts: {conflict_details[:2]}...")
    
    # TimeAgent sends feedback to PolicyAgent
    time_to_policy_msg = create_cnp_message(
        performative="inform",
        sender=AgentRole.TIME_AGENT.value,
        receiver=AgentRole.POLICY_AGENT.value,
        content={
            "issue": "timeline_conflict",
            "conflicts": conflict_details,
            "current_flight": selected_flight.get("flight_id", "unknown"),
            "current_arrival": selected_flight.get("arrival_time", "unknown"),
            "request": "Find flight with earlier arrival to allow sufficient buffer time"
        }
    )
    messages.append(time_to_policy_msg)
    
    print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
    print(f"  │ From: TimeAgent → To: PolicyAgent")
    print(f"  │ Performative: INFORM (timeline_conflict)")
    print(f"  │ Request: Earlier arrival time needed")
    print(f"  └────────────────────────────────────────────────────────")
    
    # PolicyAgent asks FlightAgent to find better options
    # Filter flights with earlier arrival times
    current_arrival = selected_flight.get("arrival_time", "12:00")
    try:
        current_arr_h, current_arr_m = map(int, current_arrival.split(":"))
        current_arr_mins = current_arr_h * 60 + current_arr_m
    except:
        current_arr_mins = 12 * 60  # Default noon
    
    # Find flights with earlier arrivals
    better_flights = []
    for f in available_flights:
        try:
            arr_time = f.get("arrival_time", "12:00")
            arr_h, arr_m = map(int, arr_time.split(":"))
            arr_mins = arr_h * 60 + arr_m
            if arr_mins < current_arr_mins:
                better_flights.append(f)
        except:
            continue
    
    # Sort by arrival time (earliest first)
    better_flights.sort(key=lambda x: x.get("arrival_time", "23:59"))
    
    print(f"  Found {len(better_flights)} flights with earlier arrival times")
    
    if better_flights:
        # PolicyAgent re-evaluates with better flight options
        print(f"  [PolicyAgent] Re-evaluating with earlier flights...")
        
        # Use PolicyAgent to find best combination with the better flights
        combination_result = policy_agent.find_best_combination(
            flights=better_flights,
            hotels=available_hotels,
            budget=budget,
            nights=nights,
            preferences=preferences
        )
        
        if combination_result.success:
            new_flight = combination_result.selected_flight
            new_hotel = combination_result.selected_hotel
            
            print(f"  ✅ Found better flight: {new_flight.get('airline', 'Unknown')} arriving at {new_flight.get('arrival_time', '?')}")
            
            # RE-VALIDATE TIME with new flight
            # Calculate new timeline to show in results
            new_arrival = new_flight.get('arrival_time', '12:00')
            meeting_times = preferences.get('meeting_times', [])
            meeting_time = None
            if meeting_times:
                mt = meeting_times[0]
                if ' ' in str(mt):
                    _, meeting_time = str(mt).split(' ', 1)
                else:
                    meeting_time = str(mt)
            
            # Simple time check for display
            is_now_feasible = True
            new_buffer = None
            if meeting_time:
                try:
                    arr_h, arr_m = map(int, new_arrival.split(':'))
                    meet_h, meet_m = map(int, meeting_time.split(':'))
                    arr_mins = arr_h * 60 + arr_m
                    meet_mins = meet_h * 60 + meet_m
                    # Add ~45 min for airport-to-hotel transit
                    hotel_arrival_mins = arr_mins + 45
                    new_buffer = meet_mins - hotel_arrival_mins
                    is_now_feasible = new_buffer >= 120  # Need 2 hours buffer
                    print(f"  📊 New timing: arrive {new_arrival}, hotel ~{(hotel_arrival_mins//60):02d}:{(hotel_arrival_mins%60):02d}, meeting {meeting_time}, buffer {new_buffer} min")
                    if is_now_feasible:
                        print(f"  ✅ NEW FLIGHT IS FEASIBLE! Buffer: {new_buffer} minutes")
                    else:
                        print(f"  ⚠️  Still tight: {new_buffer} min buffer (need 120+)")
                except:
                    pass
            
            # Update time_constraints with NEW timeline
            new_timeline = {
                "flight_arrival": new_arrival,
                "hotel_arrival": f"{((arr_h*60+arr_m+45)//60):02d}:{((arr_h*60+arr_m+45)%60):02d}" if new_arrival else "unknown",
                "meeting_1": meeting_time or "unknown",
                "meeting_1_buffer": new_buffer
            }
            
            new_conflicts = [] if is_now_feasible else [{
                "conflict_type": "insufficient_buffer",
                "severity": "warning" if new_buffer and new_buffer >= 60 else "error",
                "message": f"Buffer time {new_buffer} min is below recommended 2 hours"
            }]
            
            # PolicyAgent informs of new selection
            policy_response_msg = create_cnp_message(
                performative="inform",
                sender=AgentRole.POLICY_AGENT.value,
                receiver=AgentRole.TIME_AGENT.value,
                content={
                    "response": "alternative_found",
                    "new_flight": new_flight.get("flight_id"),
                    "new_arrival": new_flight.get("arrival_time"),
                    "is_now_feasible": is_now_feasible,
                    "new_buffer": new_buffer,
                    "reasoning": "Selected flight with earlier arrival to resolve timeline conflict"
                }
            )
            messages.append(policy_response_msg)
            
            return {
                "selected_flight": new_flight,
                "selected_hotel": new_hotel,
                # UPDATE the time_constraints with the NEW timeline
                "time_constraints": {
                    "feasible": is_now_feasible,
                    "timeline": new_timeline,
                    "conflicts": new_conflicts,
                    "reasoning": f"Re-validated after selecting earlier flight. Buffer: {new_buffer} min"
                },
                "feasibility_analysis": {
                    "is_feasible": is_now_feasible,
                    "timeline": new_timeline,
                    "conflicts": new_conflicts,
                    "reasoning": f"Selected earlier flight arriving at {new_arrival}"
                },
                "compliance_status": {
                    "overall_status": "compliant",
                    "is_compliant": True,
                    "violations": [],
                    "total_cost": combination_result.total_cost,
                    "budget": budget,
                    "budget_remaining": combination_result.budget_remaining,
                    "reasoning": "Re-selected after timeline feedback",
                },
                "current_phase": "select_options",
                "messages": messages,
                "metrics": metrics
            }
    
    # No better options found - proceed with current selection
    print(f"  ⚠️ No earlier flights available within budget - proceeding with current selection")
    
    policy_response_msg = create_cnp_message(
        performative="inform",
        sender=AgentRole.POLICY_AGENT.value,
        receiver=AgentRole.TIME_AGENT.value,
        content={
            "response": "no_alternative",
            "reasoning": "No earlier flights available within budget constraints"
        }
    )
    messages.append(policy_response_msg)
    
    return {
        "current_phase": "select_options",
        "messages": messages,
        "metrics": metrics
    }


# === NODE: SELECT OPTIONS ===
def select_options_node(state: TripPlanningState) -> Dict[str, Any]:
    """Orchestrator confirms PolicyAgent selection or makes fallback choice."""
    print("\n" + "-"*60)
    print("🎯 ORCHESTRATOR - Chain-of-Thought Selection")
    print("-"*60)
    
    # Check if PolicyAgent already made selections (preferred path)
    policy_flight = state.get("selected_flight")
    policy_hotel = state.get("selected_hotel")
    compliance = state.get("compliance_status", {})
    
    # If PolicyAgent already selected compliant options, use them directly
    if policy_flight and policy_hotel and compliance.get("is_compliant", False):
        print("  ✓ Using PolicyAgent's optimal selections")
        
        # Convert to dict if needed
        if hasattr(policy_flight, 'model_dump'):
            policy_flight = policy_flight.model_dump()
        if hasattr(policy_hotel, 'model_dump'):
            policy_hotel = policy_hotel.model_dump()
        
        price_flight = policy_flight.get('price_usd', policy_flight.get('price', 'N/A'))
        price_hotel = policy_hotel.get('price_per_night_usd', policy_hotel.get('price', 'N/A'))
        print(f"  ✓ Selected flight: {policy_flight.get('airline', 'Unknown')} - ${price_flight}")
        print(f"  ✓ Selected hotel: {policy_hotel.get('name', 'Unknown')} - ${price_hotel}/night")
        
        messages = [
            create_cnp_message(
                performative="accept",
                sender=AgentRole.ORCHESTRATOR.value,
                receiver=AgentRole.POLICY_AGENT.value,
                content={"accepted": True, "reason": "PolicyAgent selection approved"}
            )
        ]
        
        return {
            "selected_flight": policy_flight,
            "selected_hotel": policy_hotel,
            "current_phase": "finalizing",
            "messages": messages
        }
    
    # Fallback: Run orchestrator selection if PolicyAgent didn't select
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    origin = state.get("origin", "")
    destination = state.get("destination", "")
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
    
    # Convert to models using helpers
    flight_models = []
    for f in flights[:5]:
        try:
            flight_models.append(_dict_to_flight(f, origin, destination))
        except:
            pass
    
    hotel_models = []
    for h in hotels[:5]:
        try:
            hotel_models.append(_dict_to_hotel(h, destination))
        except:
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


# === NODE: FINALIZE ===
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
    preferences = state.get("preferences", {})
    nights = _calculate_nights(preferences) or 2
    
    # Calculate total cost
    total_cost = 0
    if selected_flight:
        total_cost += selected_flight.get("price_usd", selected_flight.get("price", 0))
    if selected_hotel:
        total_cost += selected_hotel.get("price_per_night_usd", selected_hotel.get("price", 0)) * nights
    
    # Build final recommendation
    final_recommendation = {
        "flight": selected_flight,
        "hotel": selected_hotel,
        "total_estimated_cost": total_cost,
        "compliance_status": compliance.get("overall_status", "unknown"),
        "timeline_feasible": time_constraints.get("feasible", True),
        "cheaper_alternatives": cheaper_alternatives,
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
    explanation_parts.append(f"**Budget Status**: {compliance.get('overall_status', 'Not checked')}")
    
    if time_constraints.get("feasible"):
        explanation_parts.append("**Timeline**: Schedule is feasible with adequate buffer times.")
    else:
        explanation_parts.append("**Timeline**: ⚠️ Some scheduling concerns - review recommended.")
    
    # Show cheaper alternatives
    if cheaper_alternatives:
        explanation_parts.append("\n### 💰 Cheaper Alternatives")
        for i, alt in enumerate(cheaper_alternatives[:3], 1):
            flight_info = alt.get('flight', {})
            hotel_info = alt.get('hotel', {})
            explanation_parts.append(
                f"{i}. **{flight_info.get('airline', 'Unknown')}** + "
                f"**{hotel_info.get('name', 'Unknown')}** ({hotel_info.get('stars', '?')}★) - "
                f"Total: ${alt.get('total_cost', 0)} "
                f"(save ${alt.get('savings_vs_selected', 0)})"
            )
    
    # Add metrics summary
    explanation_parts.append("\n### Workflow Metrics")
    explanation_parts.append(f"- Backtracking iterations: {metrics.get('backtracking_count', 0)}")
    explanation_parts.append(f"- Negotiation rounds: {metrics.get('negotiation_rounds', 0)}")
    explanation_parts.append(f"- Message exchanges: {metrics.get('message_exchanges', 0)}")
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


# === ROUTING FUNCTIONS ===
def should_backtrack_after_policy(state: TripPlanningState) -> Literal["check_time", "finalize"]:
    """Route to time_check if compliant, else finalize with error."""
    compliance = state.get("compliance_status", {})
    
    is_compliant = compliance.get("is_compliant", True)
    
    if is_compliant:
        print(f"\n  ✅ Valid combination found - proceeding to time check")
        return "check_time"
    else:
        # No valid combination exists within budget
        print(f"\n  ❌ No valid combinations - finalizing with recommendation")
        return "finalize"


def should_backtrack_after_time(state: TripPlanningState) -> Literal["select_options", "time_policy_feedback"]:
    """Route to policy feedback on time conflicts, otherwise proceed to selection."""
    time_constraints = state.get("time_constraints", {})
    metrics = state.get("metrics", {})
    
    is_feasible = time_constraints.get("feasible", True)
    conflicts = time_constraints.get("conflicts", [])
    time_feedback_count = metrics.get("time_feedback_count", 0)
    
    # Only provide feedback for conflicts and if we have iterations left
    severe_conflicts = [c for c in conflicts if isinstance(c, dict) and c.get("severity") == "error"]
    
    if not is_feasible and severe_conflicts and time_feedback_count < MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  ⏰ TIME CONFLICT: Reporting {len(severe_conflicts)} issue(s) to PolicyAgent")
        return "time_policy_feedback"
    
    return "select_options"


def increment_backtrack_counter(state: TripPlanningState) -> Dict[str, Any]:
    """
    Increment backtrack counter and prepare for re-search.
    
    If flight alternatives were provided by time_policy_feedback, use them
    to replace the available flights for the next search iteration.
    """
    metrics = state.get("metrics", {}).copy()
    metrics["backtracking_count"] = metrics.get("backtracking_count", 0) + 1
    
    print(f"\n  📊 Backtracking iteration: {metrics['backtracking_count']}/{MAX_BACKTRACKING_ITERATIONS}")
    
    # Use flight alternatives from time feedback if available
    flight_alternatives = state.get("flight_alternatives", [])
    if flight_alternatives:
        print(f"  ✈️  Using {len(flight_alternatives)} alternative flights from TimeAgent feedback")
        return {
            "metrics": metrics,
            "available_flights": flight_alternatives,  # Replace flights with alternatives
            "flight_alternatives": []  # Clear alternatives after use
        }
    
    return {"metrics": metrics}


def should_backtrack_after_time_feedback(state: TripPlanningState) -> Literal["increment_backtrack", "select_options"]:
    """
    Route after time-policy feedback:
    - If backtracking iterations remaining AND flight alternatives provided → re-search
    - Otherwise → proceed to selection with current options
    """
    metrics = state.get("metrics", {})
    backtrack_count = metrics.get("backtracking_count", 0)
    flight_alternatives = state.get("flight_alternatives", [])
    
    # Check if we have alternatives and iterations remaining
    if flight_alternatives and backtrack_count < MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  🔄 Time conflict resolved - backtracking with alternatives (iteration {backtrack_count + 1}/{MAX_BACKTRACKING_ITERATIONS})")
        return "increment_backtrack"
    
    if backtrack_count >= MAX_BACKTRACKING_ITERATIONS:
        print(f"\n  ⚠️  Max backtracking iterations reached - proceeding with current selection")
    elif not flight_alternatives:
        print(f"\n  ℹ️  No flight alternatives available - proceeding with current selection")
    
    return "select_options"


# === BUILD WORKFLOW ===
def build_workflow() -> StateGraph:
    """
    Build LangGraph workflow with new architecture:
    
    Initialize → Search → PolicyDecision → (if needed) Negotiation → PolicyDecision → Time → Select → Finalize
    
    KEY CHANGES:
    1. Agents return DIVERSE options across all price/quality tiers
    2. PolicyAgent FIRST tries to find valid combination
    3. Negotiation ONLY starts if no valid combination exists within budget
    4. PolicyAgent directs SPECIFIC agent(s) to adjust prices during negotiation
    """
    
    # Create the state graph
    workflow = StateGraph(TripPlanningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("parallel_search", parallel_search_node)  # Agents return diverse options
    workflow.add_node("check_policy", check_policy_node)  # PolicyAgent decides best combo
    workflow.add_node("negotiation", negotiation_node)  # Only if no valid combo
    workflow.add_node("increment_backtrack", increment_backtrack_counter)
    workflow.add_node("check_time", check_time_node)
    workflow.add_node("time_policy_feedback", time_policy_feedback_node)
    workflow.add_node("select_options", select_options_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.set_entry_point("initialize")
    
    # NEW FLOW: Search → PolicyDecision (not negotiation first)
    workflow.add_edge("initialize", "parallel_search")
    workflow.add_edge("parallel_search", "check_policy")  # Go directly to PolicyAgent
    
    # PolicyAgent routing: if valid combo found → time check, else → negotiation
    workflow.add_conditional_edges(
        "check_policy", 
        should_route_after_policy,
        {
            "check_time": "check_time",       # Valid combo found
            "negotiation": "negotiation",     # Need to negotiate
            "finalize": "finalize"            # Error case
        }
    )
    
    # Negotiation loop - goes back to policy check
    workflow.add_conditional_edges(
        "negotiation",
        should_continue_negotiation,
        {
            "negotiation": "negotiation",     # More rounds needed
            "check_policy": "check_policy"    # Re-evaluate after refinement
        }
    )
    
    workflow.add_edge("increment_backtrack", "parallel_search")
    
    # Time check routing
    workflow.add_conditional_edges(
        "check_time", should_backtrack_after_time,
        {"select_options": "select_options", "time_policy_feedback": "time_policy_feedback"}
    )
    
    # Time feedback routing - CRITICAL: Connect backtracking loop
    # Previously: direct edge to select_options (backtracking never happened)
    # Now: conditional routing to increment_backtrack if alternatives available
    workflow.add_conditional_edges(
        "time_policy_feedback",
        should_backtrack_after_time_feedback,
        {
            "increment_backtrack": "increment_backtrack",  # Backtrack with alternatives
            "select_options": "select_options"             # No alternatives, proceed
        }
    )
    workflow.add_edge("select_options", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


def should_route_after_policy(state: TripPlanningState) -> Literal["check_time", "negotiation", "finalize"]:
    """
    CNP ROUTING: Check if negotiation is needed based on BUDGET, not just compliance flag.
    
    The key insight: PolicyAgent always returns success=True (best-effort for business travel),
    but we need to check budget_remaining to know if negotiation is actually needed.
    
    Flow:
    - budget_remaining >= 0 → Within budget, check if quality upgrade needed
    - negotiation_converged → Best effort accepted, proceed to time check
    - budget_remaining < 0 → Over budget, start/continue negotiation
    - max rounds reached → Accept best available and proceed
    """
    compliance = state.get("compliance_status", {})
    metrics = state.get("metrics", {})
    negotiation_rounds = metrics.get("negotiation_rounds", 0)
    negotiation_converged = metrics.get("negotiation_converged", False)
    quality_upgrade_attempted = metrics.get("quality_upgrade_attempted", False)
    
    # CRITICAL FIX: Check budget_remaining, not just is_compliant
    budget_remaining = compliance.get("budget_remaining", 0)
    is_compliant = compliance.get("is_compliant", False)
    total_cost = compliance.get("total_cost", 0)
    budget = state.get("budget", 2000)
    
    # Calculate budget utilization
    budget_utilization = (total_cost / budget * 100) if budget > 0 else 100
    
    # Case 1: Within budget - check if we should negotiate for QUALITY UPGRADE
    if budget_remaining >= 0:
        # If utilization is below 80% and we haven't tried upgrade yet, negotiate UP!
        if budget_utilization < 80 and not quality_upgrade_attempted and negotiation_rounds == 0:
            print(f"\n  💎 QUALITY UPGRADE OPPORTUNITY!")
            print(f"     Budget: ${budget:.0f}, Current: ${total_cost:.0f} ({budget_utilization:.0f}% utilization)")
            print(f"     ${budget_remaining:.0f} unused - seeking premium options...")
            return "negotiation"
        
        print(f"\n  ✅ Within budget: ${total_cost:.0f} / ${budget:.0f} (${budget_remaining:.0f} remaining)")
        return "check_time"
    
    # Case 2: Negotiation converged (best effort accepted) - proceed even if over budget
    if negotiation_converged:
        over_budget = abs(budget_remaining)
        print(f"\n  ✅ Negotiation converged - accepting best effort")
        print(f"     Total: ${total_cost:.0f} (${over_budget:.0f} over budget)")
        print(f"     Proceeding to time check...")
        return "check_time"
    
    # Case 3: Over budget but max rounds reached - accept best effort
    if negotiation_rounds >= MAX_NEGOTIATION_ROUNDS:
        over_budget = abs(budget_remaining)
        print(f"\n  ⚠️  Max negotiation rounds ({MAX_NEGOTIATION_ROUNDS}) reached")
        print(f"     Accepting best effort: ${total_cost:.0f} (${over_budget:.0f} over budget)")
        if state.get("selected_flight") and state.get("selected_hotel"):
            return "check_time"
        return "finalize"
    
    # Case 4: Over budget with rounds remaining - START NEGOTIATION
    over_budget = abs(budget_remaining)
    print(f"\n  💰 BUDGET EXCEEDED by ${over_budget:.0f}")
    print(f"     Current best: ${total_cost:.0f} vs Budget: ${budget:.0f}")
    print(f"     Starting negotiation round {negotiation_rounds + 1}/{MAX_NEGOTIATION_ROUNDS}")
    return "negotiation"


def create_trip_planning_app():
    """Create and compile the trip planning application."""
    return build_workflow().compile()


def plan_trip(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str = None,
    budget: float = None,
    hotel_checkin: str = None,
    hotel_checkout: str = None,
    meeting_time: str = None,
    meeting_date: str = None,
    meeting_coordinates: dict = None
) -> Dict[str, Any]:
    """
    Main entry point for multi-agent trip planning.
    
    Accepts structured input matching what the frontend sends.
    The agents will REASON about the best options based on budget,
    convenience, quality, and value - not just filter by criteria.
    
    Args:
        origin: Departure city code (e.g., 'NYC', 'SF')
        destination: Arrival city code
        departure_date: Date of departure (YYYY-MM-DD)
        return_date: Optional return date
        budget: Total budget (agents reason about how to use it wisely)
        hotel_checkin: Check-in date
        hotel_checkout: Check-out date
        meeting_time: Time of meeting (HH:MM)
        meeting_date: Date of meeting
        meeting_coordinates: Dict with 'lat' and 'lon' keys
    
    Returns:
        Final workflow state with recommendation
    
    Note:
        Agents decide on hotel quality, flight class, etc. based on reasoning.
        No filtering parameters - the LLM agents evaluate what's appropriate.
    """
    # Build preferences from structured input
    # Note: No filtering constraints - agents reason about quality/class based on budget
    preferences = {
        "hotel_checkin": hotel_checkin or departure_date,
        "hotel_checkout": hotel_checkout or return_date,
    }
    
    # Build meeting times list if provided
    if meeting_date and meeting_time:
        preferences["meeting_times"] = [f"{meeting_date} {meeting_time}"]
    
    if meeting_coordinates:
        preferences["meeting_location"] = meeting_coordinates
    
    # Budget allocation (agents will negotiate to maximize usage)
    if budget:
        preferences["adjusted_flight_budget"] = budget * 0.5  # 50/50 split as starting point
        preferences["adjusted_hotel_budget"] = budget * 0.5
    
    # Create initial state
    initial_state = create_initial_state(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        budget=budget,
        preferences=preferences
    )
    
    # Create and run the app
    app = create_trip_planning_app()
    
    # Run the workflow
    final_state = None
    for state in app.stream(initial_state):
        final_state = state
    
    return final_state


# Module can be imported but not run directly - use main.py or tests
