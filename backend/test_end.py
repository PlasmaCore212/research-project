#!/usr/bin/env python3
"""
End-to-End Integration Tests for Multi-Agent Trip Planning System

This test file demonstrates the CNP (Contract Net Protocol) negotiation
and backtracking mechanisms. Run with:
    python test_end.py

To run a single test:
    python test_end.py --test 6

To see only negotiation tests:
    python test_end.py --negotiation
"""

import sys
import os
import asyncio
from datetime import datetime
import json
import argparse

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import build_workflow
from orchestrator.state import create_initial_state

# =============================================================================
# TEST DATA
# =============================================================================

test_data = [
    # ==========================================================================
    # CATEGORY A: NO NEGOTIATION EXPECTED (generous budgets)
    # These prove the system is EFFICIENT - doesn't negotiate when not needed
    # ==========================================================================
    {
        "id": 1,
        "name": "Generous Budget CHI→SF",
        "origin": "CHI",
        "destination": "SF",
        "departure_date": "2026-02-10",
        "return_date": "2026-02-14",
        "hotel_checkin": "2026-02-10",
        "hotel_checkout": "2026-02-13",
        "meeting_date": "2026-02-11",
        "meeting_time": "14:00",
        "meeting_address": "345 Spear Street, San Francisco, CA 94105",  # Salesforce Tower
        "meeting_coordinates": {"lat": 37.7898, "lon": -122.3927},
        "budget": 5000,
        "user_amenities": ["WiFi", "Gym", "Business Center"],
        "expect_negotiation": True,
        "category": "A-NoNegotiation",
        "description": "High budget - may trigger QUALITY UPGRADE negotiation to use budget better"
    },
    {
        "id": 2,
        "name": "Comfortable SF→BOS",
        "origin": "SF",
        "destination": "BOS",
        "departure_date": "2026-03-15",
        "return_date": "2026-03-17",
        "hotel_checkin": "2026-03-15",
        "hotel_checkout": "2026-03-16",
        "meeting_date": "2026-03-16",
        "meeting_time": "11:00",
        "meeting_address": "1 Federal Street, Boston, MA 02110",  # Financial District
        "meeting_coordinates": {"lat": 42.3556, "lon": -71.0534},
        "budget": 2500,
        "user_amenities": ["WiFi", "Restaurant"],
        "expect_negotiation": False,
        "category": "A-NoNegotiation",
        "description": "Adequate budget for short trip - should succeed directly"
    },
    
    # ==========================================================================
    # CATEGORY B: NEGOTIATION EXPECTED (tight budgets)
    # These prove the CNP protocol WORKS - agents negotiate to find solutions
    # ==========================================================================
    {
        "id": 3,
        "name": "Tight Budget NYC→CHI",
        "origin": "NYC",
        "destination": "CHI",
        "departure_date": "2026-04-05",
        "return_date": "2026-04-08",
        "hotel_checkin": "2026-04-05",
        "hotel_checkout": "2026-04-07",
        "meeting_date": "2026-04-06",
        "meeting_time": "10:00",
        "meeting_address": "875 N Michigan Ave, Chicago, IL 60611",  # John Hancock Center
        "meeting_coordinates": {"lat": 41.8988, "lon": -87.6245},
        "budget": 850,
        "user_amenities": ["WiFi"],
        "expect_negotiation": True,
        "category": "B-NegotiationNeeded",
        "description": "⚡ Tight budget for 3-night stay - negotiation should trigger"
    },
    {
        "id": 4,
        "name": "Budget Constraint BOS→NYC",
        "origin": "BOS",
        "destination": "NYC",
        "departure_date": "2026-05-10",
        "return_date": "2026-05-12",
        "hotel_checkin": "2026-05-10",
        "hotel_checkout": "2026-05-11",
        "meeting_date": "2026-05-11",
        "meeting_time": "14:00",
        "meeting_address": "30 Rockefeller Plaza, New York, NY 10112",  # Rockefeller Center
        "meeting_coordinates": {"lat": 40.7587, "lon": -73.9787},
        "budget": 700,
        "user_amenities": ["WiFi", "Parking"],
        "expect_negotiation": True,
        "category": "B-NegotiationNeeded",
        "description": "⚡ Moderate constraint - expect 1-2 negotiation rounds"
    },
    
    # ==========================================================================
    # CATEGORY C: EXTREME CONSTRAINTS (tests max negotiation)
    # These prove the system handles IMPOSSIBLE constraints gracefully
    # ==========================================================================
    {
        "id": 5,
        "name": "Extreme Budget CHI→BOS",
        "origin": "CHI",
        "destination": "BOS",
        "departure_date": "2026-06-01",
        "return_date": "2026-06-03",
        "hotel_checkin": "2026-06-01",
        "hotel_checkout": "2026-06-02",
        "meeting_date": "2026-06-01",
        "meeting_time": "18:00",
        "meeting_address": "200 Clarendon Street, Boston, MA 02116",  # John Hancock Tower
        "meeting_coordinates": {"lat": 42.3495, "lon": -71.0765},
        "budget": 450,
        "user_amenities": ["WiFi"],
        "expect_negotiation": True,
        "category": "C-ExtremePressure",
        "description": "⚡⚡ Extreme constraint - tests negotiation convergence"
    },
    {
        "id": 6,
        "name": "Impossible Budget SF→NYC",
        "origin": "SF",
        "destination": "NYC",
        "departure_date": "2026-07-01",
        "return_date": "2026-07-04",
        "hotel_checkin": "2026-07-01",
        "hotel_checkout": "2026-07-03",
        "meeting_date": "2026-07-02",
        "meeting_time": "09:00",
        "meeting_address": "1 World Trade Center, New York, NY 10007",  # One WTC
        "meeting_coordinates": {"lat": 40.7127, "lon": -74.0134},
        "budget": 400,
        "user_amenities": ["WiFi", "Business Center"],
        "expect_negotiation": True,
        "category": "C-ExtremePressure",
        "description": "⚡⚡ Near-impossible budget - max rounds, best effort outcome"
    },
    
    # ==========================================================================
    # CATEGORY D: BUDGET UTILIZATION TEST (80% threshold)
    # Tests the new 80% threshold trigger for upgrade negotiation
    # ==========================================================================
    {
        "id": 7,
        "name": "Budget Utilization NYC→SF",
        "origin": "NYC",
        "destination": "SF",
        "departure_date": "2026-08-15",
        "return_date": "2026-08-17",
        "hotel_checkin": "2026-08-15",
        "hotel_checkout": "2026-08-16",
        "meeting_date": "2026-08-16",
        "meeting_time": "10:00",
        "meeting_address": "101 California Street, San Francisco, CA 94111",  # 101 California
        "meeting_coordinates": {"lat": 37.7930, "lon": -122.3983},
        "budget": 1200,
        "user_amenities": ["WiFi", "Gym", "Spa", "Room Service"],
        "expect_negotiation": True,
        "category": "D-BudgetUtilization",
        "description": "⚡ Tests 80% threshold - cheap options available should trigger UPGRADE negotiation"
    },
]


def print_header(data):
    """Print test header with clear formatting."""
    print("\n" + "=" * 80)
    print(f"🧪 TEST #{data['id']}: {data['name']}")
    print("=" * 80)
    print(f"📍 Route: {data['origin']} → {data['destination']}")
    print(f"📅 Dates: {data['departure_date']} to {data['return_date']}")
    print(f"🏢 Meeting: {data['meeting_address']}")
    print(f"    Time: {data['meeting_date']} @ {data['meeting_time']}")
    print(f"💵 Budget: ${data['budget']}")
    if data.get('user_amenities'):
        print(f"🛎️  Amenities: {', '.join(data['user_amenities'])}")
    print(f"📝 {data['description']}")
    if data.get('expect_negotiation'):
        print(f"⚡ EXPECTED: Negotiation should trigger")
    print("-" * 80)


def print_workflow_step(step_name: str, step_data: dict):
    """Print each workflow step as it executes."""
    # Extract the node name from the step
    if isinstance(step_data, dict):
        for node_name, node_output in step_data.items():
            print(f"\n{'─' * 40}")
            print(f"📌 WORKFLOW NODE: {node_name}")
            print(f"{'─' * 40}")
            
            # Show relevant data based on node
            if node_name == "initialize":
                print(f"   Phase: {node_output.get('current_phase', 'N/A')}")
                
            elif node_name == "parallel_search":
                flights = node_output.get('available_flights', [])
                hotels = node_output.get('available_hotels', [])
                print(f"   ✈️  Found {len(flights)} flights")
                print(f"   🏨 Found {len(hotels)} hotels")
                
            elif node_name == "check_policy":
                compliance = node_output.get('compliance_status', {})
                total = compliance.get('total_cost', 0)
                remaining = compliance.get('budget_remaining', 0)
                is_compliant = compliance.get('is_compliant', False)
                
                print(f"   💰 Total Cost: ${total}")
                print(f"   💵 Budget Remaining: ${remaining}")
                if remaining < 0:
                    print(f"   ⚠️  OVER BUDGET by ${abs(remaining)}")
                else:
                    print(f"   ✅ Within budget")
                    
                flight = node_output.get('selected_flight')
                hotel = node_output.get('selected_hotel')
                if flight:
                    print(f"   ✈️  Selected: {flight.get('airline', 'N/A')} - ${flight.get('price_usd', 0)}")
                if hotel:
                    print(f"   🏨 Selected: {hotel.get('name', 'N/A')} - ${hotel.get('price_per_night_usd', 0)}/night")
                    
            elif node_name == "negotiation":
                metrics = node_output.get('metrics', {})
                rounds = metrics.get('negotiation_rounds', 0)
                messages = metrics.get('message_exchanges', 0)
                
                print(f"   🤝 NEGOTIATION ACTIVE")
                print(f"   📊 Round: {rounds}/5")
                print(f"   📨 Messages exchanged: {messages}")
                
                # Show refined options
                flights = node_output.get('available_flights', [])
                hotels = node_output.get('available_hotels', [])
                if flights:
                    prices = [f.get('price_usd', 0) for f in flights[:5]]
                    print(f"   ✈️  Refined flights: {len(flights)} (prices: ${min(prices)}-${max(prices)})")
                if hotels:
                    prices = [h.get('price_per_night_usd', 0) for h in hotels[:5]]
                    print(f"   🏨 Refined hotels: {len(hotels)} (prices: ${min(prices)}-${max(prices)}/night)")
                    
            elif node_name == "check_time":
                time_constraints = node_output.get('time_constraints', {})
                feasible = time_constraints.get('feasible', True)
                conflicts = time_constraints.get('conflicts', [])
                
                print(f"   ⏰ Timeline Feasible: {'✅ Yes' if feasible else '❌ No'}")
                if conflicts:
                    print(f"   ⚠️  Conflicts: {len(conflicts)}")
                    for c in conflicts[:3]:
                        if isinstance(c, dict):
                            print(f"      - {c.get('type', 'Unknown')}: {c.get('message', '')[:50]}")
                            
            elif node_name == "time_policy_feedback":
                alternatives = node_output.get('flight_alternatives', [])
                print(f"   🔄 Time conflict feedback generated")
                print(f"   ✈️  Alternative flights: {len(alternatives)}")
                
            elif node_name == "increment_backtrack":
                metrics = node_output.get('metrics', {})
                count = metrics.get('backtracking_count', 0)
                print(f"   🔙 Backtracking iteration: {count}/5")
                
            elif node_name == "select_options":
                flight = node_output.get('selected_flight')
                hotel = node_output.get('selected_hotel')
                print(f"   ✅ Final selection confirmed")
                if flight:
                    print(f"   ✈️  {flight.get('airline', 'N/A')} - ${flight.get('price_usd', 0)}")
                if hotel:
                    print(f"   🏨 {hotel.get('name', 'N/A')} - ${hotel.get('price_per_night_usd', 0)}/night")
                    
            elif node_name == "finalize":
                recommendation = node_output.get('final_recommendation', {})
                print(f"   🎯 Trip planning complete!")
                if recommendation:
                    print(f"   📋 Recommendation generated")


def print_results(final_state: dict, data: dict):
    """Print final results with metrics."""
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    
    # Get metrics
    metrics = final_state.get('metrics', {})
    compliance = final_state.get('compliance_status', {})
    
    # Negotiation metrics
    negotiation_rounds = metrics.get('negotiation_rounds', 0)
    message_exchanges = metrics.get('message_exchanges', 0)
    backtracking_count = metrics.get('backtracking_count', 0)
    
    print(f"\n📈 WORKFLOW METRICS:")
    print(f"   Negotiation rounds: {negotiation_rounds}")
    print(f"   Message exchanges: {message_exchanges}")
    print(f"   Backtracking iterations: {backtracking_count}")
    
    # Check if negotiation happened as expected
    if data.get('expect_negotiation'):
        if negotiation_rounds > 0:
            print(f"   ✅ Negotiation triggered as expected!")
        else:
            print(f"   ⚠️  Expected negotiation but it didn't trigger")
    
    # Selected options
    flight = final_state.get('selected_flight')
    hotel = final_state.get('selected_hotel')
    
    print(f"\n✈️  SELECTED FLIGHT:")
    if flight:
        f_data = flight if isinstance(flight, dict) else flight.__dict__
        print(f"   Airline: {f_data.get('airline', 'N/A')}")
        print(f"   Flight: {f_data.get('flight_id', 'N/A')}")
        print(f"   Price: ${f_data.get('price_usd', 0)}")
        print(f"   Class: {f_data.get('class', 'Economy')}")
        print(f"   Time: {f_data.get('departure_time', 'N/A')} → {f_data.get('arrival_time', 'N/A')}")
    else:
        print("   ❌ No flight selected")
    
    print(f"\n🏨 SELECTED HOTEL:")
    if hotel:
        h_data = hotel if isinstance(hotel, dict) else hotel.__dict__
        print(f"   Name: {h_data.get('name', 'N/A')}")
        print(f"   Stars: {h_data.get('stars', 'N/A')}★")
        print(f"   Price: ${h_data.get('price_per_night_usd', 0)}/night")
        print(f"   Area: {h_data.get('business_area', 'N/A')}")
    else:
        print("   ❌ No hotel selected")
    
    # Budget analysis
    print(f"\n💰 BUDGET ANALYSIS:")
    total_cost = compliance.get('total_cost', 0)
    budget = data['budget']
    budget_remaining = compliance.get('budget_remaining', budget - total_cost)
    
    print(f"   Budget: ${budget}")
    print(f"   Total Cost: ${total_cost}")
    print(f"   Remaining: ${budget_remaining}")
    
    if budget_remaining >= 0:
        utilization = (total_cost / budget * 100) if budget > 0 else 0
        print(f"   Utilization: {utilization:.1f}%")
        print(f"   ✅ SUCCESS: Within budget!")
    else:
        print(f"   ⚠️  OVER BUDGET by ${abs(budget_remaining)}")
    
    print("\n" + "=" * 80)


async def run_test(data: dict, verbose: bool = True):
    """Run a single test case."""
    print_header(data)
    
    # Build workflow
    workflow = build_workflow()
    graph = workflow.compile()
    
    # Create initial state
    initial_state = create_initial_state(
        origin=data["origin"],
        destination=data["destination"],
        departure_date=data["departure_date"],
        return_date=data["return_date"],
        budget=data["budget"],
        preferences={
            "meeting_time": data["meeting_time"],
            "meeting_date": data["meeting_date"],
            "meeting_address": data["meeting_address"],
            "meeting_coordinates": data["meeting_coordinates"],
            "meeting_times": [f"{data['meeting_date']} {data['meeting_time']}"],
            "meeting_location": data["meeting_coordinates"],
            "hotel_checkin": data["hotel_checkin"],
            "hotel_checkout": data["hotel_checkout"],
        }
    )
    
    try:
        if verbose:
            # Stream workflow to show each step
            print("\n🔄 WORKFLOW EXECUTION:")
            final_state = None
            for step in graph.stream(initial_state):
                print_workflow_step("step", step)
                # Keep track of final state
                for node_name, node_output in step.items():
                    if final_state is None:
                        final_state = node_output
                    else:
                        # Merge outputs
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
        else:
            # Just invoke directly
            final_state = graph.invoke(initial_state)
        
        # Print results
        print_results(final_state, data)
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run trip planning integration tests")
    parser.add_argument("--test", type=int, help="Run a specific test by ID")
    parser.add_argument("--negotiation", action="store_true", help="Run only negotiation tests")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    print("\n" + "🚀" * 40)
    print("MULTI-AGENT TRIP PLANNING - INTEGRATION TESTS")
    print("🚀" * 40)
    
    tests_to_run = test_data
    
    if args.test:
        tests_to_run = [t for t in test_data if t['id'] == args.test]
        if not tests_to_run:
            print(f"❌ Test #{args.test} not found")
            return
    elif args.negotiation:
        tests_to_run = [t for t in test_data if t.get('expect_negotiation')]
        print(f"\n📋 Running {len(tests_to_run)} negotiation tests...")
    
    results = []
    for test in tests_to_run:
        result = await run_test(test, verbose=not args.quiet)
        results.append((test, result))
        print("\n" + "💤" * 20)
        print("Waiting before next test...")
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY - RESEARCH DATA")
    print("=" * 80)
    
    # Group by category for research analysis
    categories = {}
    for test, result in results:
        cat = test.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((test, result))
    
    # Detailed per-test results
    print("\n📋 INDIVIDUAL TEST RESULTS:")
    for test, result in results:
        if result:
            metrics = result.get('metrics', {})
            compliance = result.get('compliance_status', {})
            budget_remaining = compliance.get('budget_remaining', 0)
            negotiation_rounds = metrics.get('negotiation_rounds', 0)
            messages = metrics.get('message_exchanges', 0)
            
            status = "✅" if budget_remaining >= 0 else "⚠️"
            nego = f"[{negotiation_rounds} rounds, {messages} msgs]" if negotiation_rounds > 0 else "[no negotiation]"
            
            print(f"{status} #{test['id']} {test['name']}: ${compliance.get('total_cost', 0):.0f}/${test['budget']} {nego}")
        else:
            print(f"❌ #{test['id']} {test['name']}: FAILED")
    
    # Category analysis for research
    print("\n" + "-" * 80)
    print("📈 CATEGORY ANALYSIS (for research paper):")
    print("-" * 80)
    
    for cat, cat_results in categories.items():
        successful = [r for _, r in cat_results if r]
        avg_rounds = sum(r.get('metrics', {}).get('negotiation_rounds', 0) for r in successful) / len(successful) if successful else 0
        avg_messages = sum(r.get('metrics', {}).get('message_exchanges', 0) for r in successful) / len(successful) if successful else 0
        within_budget = sum(1 for r in successful if r.get('compliance_status', {}).get('budget_remaining', 0) >= 0)
        
        print(f"\n{cat}:")
        print(f"  Tests: {len(cat_results)}")
        print(f"  Within Budget: {within_budget}/{len(successful)}")
        print(f"  Avg Negotiation Rounds: {avg_rounds:.1f}")
        print(f"  Avg Message Exchanges: {avg_messages:.1f}")
    
    # Export research metrics
    print("\n" + "-" * 80)
    print("📊 KEY METRICS FOR CONTRACTPLAN:")
    print("-" * 80)
    
    all_successful = [r for _, r in results if r]
    if all_successful:
        total_negotiations = sum(1 for r in all_successful if r.get('metrics', {}).get('negotiation_rounds', 0) > 0)
        total_messages = sum(r.get('metrics', {}).get('message_exchanges', 0) for r in all_successful)
        max_rounds = max(r.get('metrics', {}).get('negotiation_rounds', 0) for r in all_successful)
        
        print(f"  Total tests run: {len(results)}")
        print(f"  Tests requiring negotiation: {total_negotiations}/{len(all_successful)} ({100*total_negotiations/len(all_successful):.0f}%)")
        print(f"  Total CNP messages: {total_messages}")
        print(f"  Max negotiation rounds in any test: {max_rounds}")
        print(f"  System efficiency: {len(all_successful)}/{len(results)} tests completed successfully")



if __name__ == "__main__":
    asyncio.run(main())
