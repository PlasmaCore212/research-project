"""
Test script for the Agentic Trip Planning Workflow

This tests the full multi-agent system with:
- ReAct reasoning agents
- Chain-of-Thought orchestration
- BDI state management
"""

from orchestrator.graph import plan_trip, create_trip_planning_app
from orchestrator.state import create_initial_state

def test_basic_trip():
    """Test a basic trip planning workflow."""
    print("\n" + "="*70)
    print("TEST: Basic Trip Planning with Agentic Architecture")
    print("="*70)
    
    result = plan_trip(
        user_request="I need to travel from New York to San Francisco for a business meeting",
        origin="NYC",
        destination="SF",
        departure_date="2026-01-20",
        return_date="2026-01-22",
        budget=1500,
        preferences={
            "min_rating": 4.0,
            "meeting_times": ["2026-01-20 14:00"]
        }
    )
    
    if result:
        # Get the final state from the last node
        for node_name, node_state in result.items():
            if "final_recommendation" in node_state:
                rec = node_state.get("final_recommendation", {})
                print("\n" + "="*70)
                print("TEST RESULTS")
                print("="*70)
                
                # Check flight
                flight = rec.get("flight")
                if flight:
                    print(f"✓ Flight: {flight.get('flight_id', 'N/A')} - ${flight.get('price', 'N/A')}")
                else:
                    print("✗ No flight selected")
                
                # Check hotel
                hotel = rec.get("hotel")
                if hotel:
                    print(f"✓ Hotel: {hotel.get('name', 'N/A')} - ${hotel.get('price', 'N/A')}/night")
                else:
                    print("✗ No hotel selected")
                
                # Check compliance
                print(f"✓ Compliance: {rec.get('compliance_status', 'unknown')}")
                
                # Check timeline
                print(f"✓ Timeline feasible: {rec.get('timeline_feasible', 'unknown')}")
                
                # Check total cost
                print(f"✓ Total estimated cost: ${rec.get('total_estimated_cost', 'N/A')}")
                
                # Check messages
                messages = node_state.get("messages", [])
                print(f"✓ Messages exchanged: {len(messages)}")
                
                # Check reasoning traces
                traces = node_state.get("reasoning_traces", {})
                print(f"✓ Agents with reasoning traces: {list(traces.keys())}")
                
                print("\n" + "="*70)
                print("TEST PASSED!" if flight and hotel else "TEST COMPLETED WITH ISSUES")
                print("="*70)
                return True
    
    print("TEST FAILED: No result returned")
    return False


def test_workflow_creation():
    """Test that the workflow can be created and compiled."""
    print("\n" + "="*70)
    print("TEST: Workflow Creation")
    print("="*70)
    
    try:
        app = create_trip_planning_app()
        print("✓ Workflow created successfully")
        print("✓ StateGraph compiled")
        return True
    except Exception as e:
        print(f"✗ Failed to create workflow: {e}")
        return False


def test_initial_state():
    """Test that initial state is created correctly."""
    print("\n" + "="*70)
    print("TEST: Initial State Creation")
    print("="*70)
    
    state = create_initial_state(
        user_request="Test trip",
        origin="NYC",
        destination="LA",
        departure_date="2026-02-01",
        budget=1000
    )
    
    checks = [
        ("user_request" in state, "user_request field"),
        ("origin" in state, "origin field"),
        ("destination" in state, "destination field"),
        ("messages" in state, "messages field"),
        ("reasoning_traces" in state, "reasoning_traces field"),
        ("metrics" in state, "metrics field"),
        (state.get("workflow_complete") == False, "workflow_complete is False"),
        (state.get("current_phase") == "initialization", "current_phase is initialization"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# AGENTIC TRIP PLANNING SYSTEM - TEST SUITE")
    print("#"*70)
    
    # Run tests
    test_initial_state()
    test_workflow_creation()
    test_basic_trip()