"""
Initialize Node - Workflow initialization and CFP broadcast.

This node initializes the trip planning workflow by:
1. Loading available flight and hotel data
2. Setting up orchestrator beliefs (BDI architecture)
3. Broadcasting Call for Proposals (CFP) to all agents
"""

from typing import Dict, Any
from datetime import datetime
from orchestrator.state import TripPlanningState, AgentRole
from orchestrator.helpers import create_cnp_message, MAX_BACKTRACKING_ITERATIONS
from orchestrator.agents_config import orchestrator, flight_loader, hotel_loader


def initialize_node(state: TripPlanningState) -> Dict[str, Any]:
    """Initialize workflow and broadcast CFP to agents."""
    print("\n" + "="*60)
    print("🚀 INITIALIZING AGENTIC TRIP PLANNING WORKFLOW")
    print("="*60)
    
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
    
    # Initialize tracking
    metrics = state.get("metrics", {})
    metrics["backtracking_count"] = 0
    metrics["negotiation_rounds"] = 0
    metrics["parallel_searches_executed"] = 0
    
    # Contract Net Protocol: Call for Proposals
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
            "data_available": {"flights": len(flights_data), "hotels": len(hotels_data)}
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
