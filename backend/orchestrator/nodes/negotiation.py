"""
Negotiation Node - CNP negotiation coordinated by Orchestrator.

This node handles the Contract Net Protocol negotiation loop:
1. Orchestrator analyzes current proposals and generates feedback
2. FlightAgent and HotelAgent refine their proposals based on feedback
3. Loop continues until budget is satisfied or max rounds reached
"""

import time
from typing import Dict, Any, Literal, List
from datetime import datetime
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, calculate_nights, MAX_NEGOTIATION_ROUNDS
from orchestrator.agents_config import flight_agent, hotel_agent, orchestrator


def refine_agent_proposal(agent, agent_name, feedback, previous_items, negotiation_round, messages):
    """
    Refine agent proposal and log messages (reduces duplication).

    Args:
        agent: Agent instance (flight_agent or hotel_agent)
        agent_name: Name for display (FlightAgent or HotelAgent)
        feedback: Feedback dict from orchestrator
        previous_items: Previous flights/hotels
        negotiation_round: Current round number
        messages: Message list to append to

    Returns:
        Refined items list
    """
    issue = feedback.get('issue', 'general') or 'general'

    print(f"\n  ┌─ CNP MESSAGE ─────────────────────────────────────────")
    print(f"  │ From: PolicyAgent → To: {agent_name}")
    print(f"  │ Performative: REJECT | Issue: {issue}")
    print(f"  └────────────────────────────────────────────────────────")

    # Send rejection message
    agent_role = AgentRole.FLIGHT_AGENT if 'flight' in agent_name.lower() else AgentRole.HOTEL_AGENT
    messages.append(create_cnp_message(
        performative="reject", sender=AgentRole.POLICY_AGENT.value,
        receiver=agent_role.value,
        content={"rejection_reason": issue, "feedback": feedback, "round": negotiation_round + 1}
    ))

    # Refine proposal
    refine_start = time.time()
    print(f"  [{agent_name}] Refining proposal...")
    result = agent.refine_proposal(feedback=feedback, previous_flights=previous_items) if 'flight' in agent_name.lower() else agent.refine_proposal(feedback=feedback, previous_hotels=previous_items)
    print(f"  [{agent_name}] Done ({time.time() - refine_start:.1f}s)")

    # Extract refined items
    refined_items = []
    result_items = result.flights if hasattr(result, 'flights') else result.hotels
    if result_items:
        refined_items = [item.model_dump() if hasattr(item, 'model_dump') else item for item in result_items]

        messages.append(create_cnp_message(
            performative="propose", sender=agent_role.value,
            receiver=AgentRole.POLICY_AGENT.value,
            content={"proposal_type": "refined", "options_count": len(refined_items), "round": negotiation_round + 1}
        ))

    return refined_items


def negotiation_node(state: TripPlanningState) -> Dict[str, Any]:
    """CNP negotiation loop - Orchestrator coordinates, agents refine."""
    print("\n" + "-"*60)
    print("🤝 CNP NEGOTIATION - Orchestrator Coordination")
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

    # INCREMENT ROUND EARLY - before any termination checks
    # This ensures the counter always reflects that we attempted this round
    metrics["negotiation_rounds"] = negotiation_round + 1

    previous_min_cost = metrics.get("previous_min_cost", None)

    # Step 1: Orchestrator generates feedback (decides what agents should do)
    step_start = time.time()
    print(f"  [Orchestrator] Analyzing proposals and deciding refinements...")
    feedback = orchestrator.decide_negotiation_target(state)
    print(f"  [Orchestrator] Done ({time.time() - step_start:.1f}s)")

    if feedback.get("reasoning"):
        print(f"  💭 Orchestrator reasoning: {feedback.get('reasoning', '')[:150]}...")

    # Early termination: If orchestrator says to accept or at market max, stop negotiating
    if feedback.get("at_market_max") or feedback.get("should_accept"):
        print(f"  ✅ Orchestrator accepts current booking (at_market_max={feedback.get('at_market_max')}, should_accept={feedback.get('should_accept')})")
        print(f"  🔚 Terminating negotiation early - booking is acceptable")
        return {"current_phase": "policy_final", "metrics": metrics, "messages": messages, "negotiation_feedback": feedback}

    # If both strategies are "maintain", no feedback needed - accept current
    if feedback.get("flight_strategy") == "maintain" and feedback.get("hotel_strategy") == "maintain":
        print(f"  ✅ Both flight and hotel strategies are 'maintain' - accepting current booking")
        print(f"  🔚 Terminating negotiation - no changes needed")
        return {"current_phase": "policy_final", "metrics": metrics, "messages": messages, "negotiation_feedback": feedback}

    print(f"  📣 Orchestrator requesting refinements from booking agents")
    
    # Step 2: Refine proposals
    refined_flights, refined_hotels = flights, hotels
    feedback_given = False
    
    # Flight Agent refinement
    if feedback.get("flight_feedback"):
        feedback_given = True
        refined_flights = refine_agent_proposal(
            agent=flight_agent,
            agent_name="FlightAgent",
            feedback=feedback["flight_feedback"],
            previous_items=flights,
            negotiation_round=negotiation_round,
            messages=messages
        ) or flights  # Fallback to previous if refinement fails

    # Hotel Agent refinement
    if feedback.get("hotel_feedback"):
        feedback_given = True
        refined_hotels = refine_agent_proposal(
            agent=hotel_agent,
            agent_name="HotelAgent",
            feedback=feedback["hotel_feedback"],
            previous_items=hotels,
            negotiation_round=negotiation_round,
            messages=messages
        ) or hotels  # Fallback to previous if refinement fails
    
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
        "metrics": metrics,
        "negotiation_feedback": feedback  # Store feedback so routing can access at_market_max
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
