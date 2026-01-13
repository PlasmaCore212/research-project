"""
Negotiation Node - CNP negotiation between PolicyAgent and booking agents.

This node handles the Contract Net Protocol negotiation loop:
1. PolicyAgent analyzes current proposals and generates feedback
2. FlightAgent and HotelAgent refine their proposals based on feedback
3. Loop continues until budget is satisfied or max rounds reached
"""

import time
from typing import Dict, Any, Literal
from datetime import datetime
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, calculate_nights, MAX_NEGOTIATION_ROUNDS
from orchestrator.agents_config import flight_agent, hotel_agent, policy_agent


def negotiation_node(state: TripPlanningState) -> Dict[str, Any]:
    """CNP negotiation loop - PolicyAgent provides feedback, booking agents refine."""
    print("\n" + "-"*60)
    print("🤝 CNP NEGOTIATION - Policy ↔ Booking Agents")
    print("-"*60)
    
    flights = state.get("available_flights", [])
    hotels = state.get("available_hotels", [])
    budget = state.get("budget", 2000)
    metrics = state.get("metrics", {})
    messages = list(state.get("messages", []))
    messages_before = len(messages)
    
    negotiation_round = metrics.get("negotiation_rounds", 0)
    nights = calculate_nights(state)
    
    # Detect quality upgrade scenario
    compliance = state.get("compliance_status", {})
    total_cost = compliance.get("total_cost", 0)
    budget_remaining = compliance.get("budget_remaining", 0)
    is_quality_upgrade = budget_remaining > 0 and (total_cost / budget * 100) < 50 if budget > 0 else False
    
    if is_quality_upgrade:
        print(f"  💎 QUALITY UPGRADE NEGOTIATION")
        print(f"     Current: ${total_cost:.0f}, Budget: ${budget:.0f}")
        metrics["quality_upgrade_attempted"] = True
    
    feedback_history = metrics.get("feedback_history", [])
    
    if negotiation_round >= MAX_NEGOTIATION_ROUNDS:
        print(f"  ⚠️  Max negotiation rounds ({MAX_NEGOTIATION_ROUNDS}) reached")
        return {"current_phase": "policy_final", "metrics": metrics, "messages": messages}
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  Negotiation Round: {negotiation_round + 1}/{MAX_NEGOTIATION_ROUNDS} @ {timestamp}")
    print(f"  Current proposals: {len(flights)} flights, {len(hotels)} hotels")
    
    previous_min_cost = metrics.get("previous_min_cost", None)
    
    # Step 1: PolicyAgent generates feedback
    step_start = time.time()
    print(f"  [PolicyAgent] Reasoning about proposals...")
    feedback = policy_agent.generate_feedback(
        flights=flights, hotels=hotels, budget=budget, nights=nights,
        negotiation_round=negotiation_round,
        feedback_history=feedback_history,
        previous_min_cost=previous_min_cost
    )
    print(f"  [PolicyAgent] Done ({time.time() - step_start:.1f}s)")
    
    if feedback.get("current_min_cost"):
        metrics["previous_min_cost"] = feedback["current_min_cost"]
    
    if feedback.get("reasoning"):
        print(f"  💭 PolicyAgent reasoning: {feedback.get('reasoning', '')[:150]}...")
    
    # Check if PolicyAgent accepts proposals
    if not feedback.get("needs_refinement"):
        print(f"  ✅ PolicyAgent: Proposals accepted - proceeding to selection")
        metrics["message_exchanges"] = len(messages)
        metrics["negotiation_converged"] = True
        return {"current_phase": "policy_final", "metrics": metrics, "messages": messages}
    
    print(f"  📣 PolicyAgent requests refinement from booking agents")
    metrics["negotiation_rounds"] = negotiation_round + 1
    
    # Step 2: Refine proposals
    refined_flights, refined_hotels = flights, hotels
    feedback_given = False
    
    # Flight Agent refinement
    if feedback.get("flight_feedback"):
        feedback_given = True
        flight_feedback = feedback["flight_feedback"]
        issue = flight_feedback.get('issue', 'general') or 'general'
        reason = flight_feedback.get('reasoning', 'Refinement needed') or 'Refinement needed'
        
        print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
        print(f"  │ From: PolicyAgent → To: FlightAgent")
        print(f"  │ Performative: REJECT | Issue: {issue}")
        print(f"  └────────────────────────────────────────────────────────")
        
        messages.append(create_cnp_message(
            performative="reject", sender=AgentRole.POLICY_AGENT.value,
            receiver=AgentRole.FLIGHT_AGENT.value,
            content={"rejection_reason": issue, "feedback": flight_feedback, "round": negotiation_round + 1}
        ))
        
        refine_start = time.time()
        print(f"  [FlightAgent] Refining proposal...")
        flight_result = flight_agent.refine_proposal(feedback=flight_feedback, previous_flights=flights)
        print(f"  [FlightAgent] Done ({time.time() - refine_start:.1f}s)")
        
        if flight_result.flights:
            new_flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in flight_result.flights]
            seen_ids = {f.get('flight_id') for f in new_flights}
            for orig_f in flights:
                if orig_f.get('flight_id') not in seen_ids:
                    new_flights.append(orig_f)
                    seen_ids.add(orig_f.get('flight_id'))
            refined_flights = new_flights
            
            messages.append(create_cnp_message(
                performative="propose", sender=AgentRole.FLIGHT_AGENT.value,
                receiver=AgentRole.POLICY_AGENT.value,
                content={"proposal_type": "refined", "options_count": len(refined_flights), "round": negotiation_round + 1}
            ))
    
    # Hotel Agent refinement
    if feedback.get("hotel_feedback"):
        feedback_given = True
        hotel_feedback = feedback["hotel_feedback"]
        h_issue = hotel_feedback.get('issue', 'general') or 'general'
        
        print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
        print(f"  │ From: PolicyAgent → To: HotelAgent")
        print(f"  │ Performative: REJECT | Issue: {h_issue}")
        print(f"  └────────────────────────────────────────────────────────")
        
        messages.append(create_cnp_message(
            performative="reject", sender=AgentRole.POLICY_AGENT.value,
            receiver=AgentRole.HOTEL_AGENT.value,
            content={"rejection_reason": h_issue, "feedback": hotel_feedback, "round": negotiation_round + 1}
        ))
        
        refine_start = time.time()
        print(f"  [HotelAgent] Refining proposal...")
        hotel_result = hotel_agent.refine_proposal(feedback=hotel_feedback, previous_hotels=hotels)
        print(f"  [HotelAgent] Done ({time.time() - refine_start:.1f}s)")
        
        if hotel_result.hotels:
            new_hotels = [h.model_dump() if hasattr(h, 'model_dump') else h for h in hotel_result.hotels]
            seen_ids = {h.get('hotel_id') for h in new_hotels}
            for orig_h in hotels:
                if orig_h.get('hotel_id') not in seen_ids:
                    new_hotels.append(orig_h)
                    seen_ids.add(orig_h.get('hotel_id'))
            refined_hotels = new_hotels
            
            messages.append(create_cnp_message(
                performative="propose", sender=AgentRole.HOTEL_AGENT.value,
                receiver=AgentRole.POLICY_AGENT.value,
                content={"proposal_type": "refined", "options_count": len(refined_hotels), "round": negotiation_round + 1}
            ))
    
    if not feedback_given:
        messages.append(create_cnp_message(
            performative="inform", sender=AgentRole.POLICY_AGENT.value,
            receiver="all_agents",
            content={"status": "review_ongoing", "round": negotiation_round + 1}
        ))
    
    metrics["message_exchanges"] = len(messages)
    
    # Update feedback history
    for key in ["flight_feedback", "hotel_feedback"]:
        if feedback.get(key):
            issue = feedback[key].get("issue")
            if issue and issue not in feedback_history:
                feedback_history.append(issue)
    metrics["feedback_history"] = feedback_history
    
    print(f"\n  📊 Negotiation round {negotiation_round + 1} complete")
    print(f"     Refined flights: {len(refined_flights)}, Refined hotels: {len(refined_hotels)}")
    
    return {
        "available_flights": refined_flights,
        "available_hotels": refined_hotels,
        "current_phase": "negotiation",
        "messages": messages,
        "metrics": metrics
    }


def should_continue_negotiation(state: TripPlanningState) -> Literal["negotiation", "check_policy"]:
    """CNP routing: Always verify refined proposals with PolicyAgent."""
    current_phase = state.get("current_phase", "")
    metrics = state.get("metrics", {})
    negotiation_rounds = metrics.get("negotiation_rounds", 0)
    
    print(f"\n  🔀 Negotiation routing: phase='{current_phase}', rounds={negotiation_rounds}/{MAX_NEGOTIATION_ROUNDS}")
    
    if current_phase == "policy_final":
        print(f"  ✅ Proposals accepted - verifying with PolicyAgent")
        return "check_policy"
    
    if negotiation_rounds >= MAX_NEGOTIATION_ROUNDS:
        print(f"  ⚠️  Max rounds reached - final verification with PolicyAgent")
        return "check_policy"
    
    print(f"  🔄 Verifying refined proposals with PolicyAgent")
    return "check_policy"
