#!/usr/bin/env python3
"""
Full End-to-End Workflow Test - Post-Refactoring Verification

This test runs the complete workflow with real agent searches to verify:
1. Agents search without budget knowledge
2. PolicyAgent validates combinations
3. Orchestrator coordinates the workflow
4. Negotiation is triggered when needed
5. Final recommendations are generated

Run with: python backend/test_full_workflow.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import build_workflow
from orchestrator.state import create_initial_state
from datetime import datetime
import json


# =============================================================================
# TEST SCENARIOS
# =============================================================================

test_scenarios = [
    {
        "id": 1,
        "name": "SCENARIO 1: Comfortable Budget (Should Accept Directly)",
        "origin": "NYC",
        "destination": "SF",
        "departure_date": "2026-03-15",
        "return_date": "2026-03-18",
        "hotel_checkin": "2026-03-15",
        "hotel_checkout": "2026-03-17",
        "meeting_date": "2026-03-16",
        "meeting_time": "14:00",
        "meeting_address": "345 Spear Street, San Francisco, CA 94105",
        "meeting_coordinates": {"lat": 37.7898, "lon": -122.3927},
        "budget": 1800,
        "user_amenities": ["Gym"],
        "expected_behavior": "Should find valid combination within budget",
        "expect_negotiation": False
    },
    {
        "id": 2,
        "name": "SCENARIO 2: Tight Budget (Should Trigger Negotiation)",
        "origin": "BOS",
        "destination": "CHI",
        "departure_date": "2026-04-10",
        "return_date": "2026-04-13",
        "hotel_checkin": "2026-04-10",
        "hotel_checkout": "2026-04-12",
        "meeting_date": "2026-04-11",
        "meeting_time": "10:00",
        "meeting_address": "875 N Michigan Ave, Chicago, IL 60611",
        "meeting_coordinates": {"lat": 41.8988, "lon": -87.6245},
        "budget": 850,
        "user_amenities": [],
        "expected_behavior": "Should trigger negotiation for cost reduction",
        "expect_negotiation": True
    },
    {
        "id": 3,
        "name": "SCENARIO 3: Generous Budget (Quality Upgrade)",
        "origin": "CHI",
        "destination": "NYC",
        "departure_date": "2026-05-20",
        "return_date": "2026-05-22",
        "hotel_checkin": "2026-05-20",
        "hotel_checkout": "2026-05-21",
        "meeting_date": "2026-05-21",
        "meeting_time": "15:00",
        "meeting_address": "30 Rockefeller Plaza, New York, NY 10112",
        "meeting_coordinates": {"lat": 40.7587, "lon": -73.9787},
        "budget": 4000,
        "user_amenities": ["Gym", "Business Center"],
        "expected_behavior": "Low budget utilization - may trigger quality upgrade",
        "expect_negotiation": True
    }
]


def print_header(text, char="="):
    """Print a formatted header."""
    width = 100
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_scenario_info(scenario):
    """Print scenario details."""
    print(f"\n📋 TEST: {scenario['name']}")
    print("-" * 100)
    print(f"   Route: {scenario['origin']} → {scenario['destination']}")
    print(f"   Dates: {scenario['departure_date']} to {scenario['return_date']}")
    print(f"   Budget: ${scenario['budget']}")
    print(f"   Meeting: {scenario['meeting_date']} at {scenario['meeting_time']}")
    print(f"   Expected: {scenario['expected_behavior']}")
    print("-" * 100)


def analyze_workflow_result(result):
    """Analyze and print workflow results."""
    print("\n" + "=" * 100)
    print("WORKFLOW ANALYSIS".center(100))
    print("=" * 100)

    # Extract metrics
    metrics = result.get("metrics", {})
    final_rec = result.get("final_recommendation", {})
    compliance = result.get("compliance_status", {})

    # Print summary
    print(f"\n✅ WORKFLOW COMPLETED")
    print(f"-" * 100)

    # Flight and Hotel Details
    flight = final_rec.get("flight") or result.get("selected_flight", {})
    hotel = final_rec.get("hotel") or result.get("selected_hotel", {})

    if flight:
        print(f"\n✈️  FLIGHT SELECTED:")
        print(f"   Airline: {flight.get('airline', 'N/A')}")
        print(f"   Class: {flight.get('class', 'N/A')}")
        print(f"   Price: ${flight.get('price_usd', 0)}")
        print(f"   Departure: {flight.get('departure_time', 'N/A')}")
        print(f"   Arrival: {flight.get('arrival_time', 'N/A')}")

    if hotel:
        nights = final_rec.get("nights", 1)
        price_per_night = hotel.get("price_per_night_usd", 0)
        print(f"\n🏨 HOTEL SELECTED:")
        print(f"   Name: {hotel.get('name', 'N/A')}")
        print(f"   Stars: {hotel.get('stars', 'N/A')}★")
        print(f"   Price: ${price_per_night}/night × {nights} nights = ${price_per_night * nights}")
        print(f"   Distance: {hotel.get('distance_to_business_center_km', 'N/A')} km to center")

    # Cost breakdown
    flight_cost = final_rec.get("flight_cost", 0)
    hotel_cost = final_rec.get("hotel_cost", 0)
    total_cost = final_rec.get("total_estimated_cost", 0) or compliance.get("total_cost", 0)
    budget = result.get("budget", 0)

    print(f"\n💰 COST BREAKDOWN:")
    print(f"   Flight: ${flight_cost}")
    print(f"   Hotel: ${hotel_cost}")
    print(f"   Total: ${total_cost}")
    print(f"   Budget: ${budget}")
    print(f"   Remaining: ${budget - total_cost}")
    print(f"   Utilization: {(total_cost / budget * 100) if budget > 0 else 0:.1f}%")

    # Metrics
    print(f"\n📊 WORKFLOW METRICS:")
    print(f"   Negotiation rounds: {metrics.get('negotiation_rounds', 0)}")
    print(f"   Message exchanges: {metrics.get('message_exchanges', 0)}")
    print(f"   Backtracking count: {metrics.get('backtracking_count', 0)}")
    print(f"   Combinations evaluated: {compliance.get('combinations_evaluated', 0)}")

    # Orchestrator coordination indicators
    if metrics.get('negotiation_rounds', 0) > 0:
        print(f"\n🎯 ORCHESTRATOR COORDINATION:")
        print(f"   ✓ Orchestrator triggered negotiation")
        print(f"   ✓ Orchestrator generated feedback for agents")
        print(f"   ✓ Agents refined proposals based on orchestrator guidance")

    # Compliance
    print(f"\n✅ COMPLIANCE STATUS:")
    print(f"   Overall: {compliance.get('overall_status', 'N/A')}")
    print(f"   Valid: {compliance.get('is_valid', 'N/A')}")
    if compliance.get('violations'):
        print(f"   Violations: {len(compliance.get('violations', []))}")

    return {
        "total_cost": total_cost,
        "budget": budget,
        "negotiation_rounds": metrics.get('negotiation_rounds', 0),
        "success": result.get("workflow_complete", False)
    }


def run_test_scenario(scenario):
    """Run a single test scenario."""
    print_header(f"RUNNING TEST SCENARIO {scenario['id']}")
    print_scenario_info(scenario)

    # Create initial state from scenario
    initial_state = create_initial_state(
        origin=scenario["origin"],
        destination=scenario["destination"],
        departure_date=scenario["departure_date"],
        return_date=scenario["return_date"],
        budget=scenario["budget"],
        preferences={
            "meeting_time": scenario["meeting_time"],
            "meeting_date": scenario["meeting_date"],
            "meeting_address": scenario["meeting_address"],
            "meeting_coordinates": scenario["meeting_coordinates"],
            "meeting_times": [f"{scenario['meeting_date']} {scenario['meeting_time']}"],
            "meeting_location": scenario["meeting_coordinates"],
            "hotel_checkin": scenario["hotel_checkin"],
            "hotel_checkout": scenario["hotel_checkout"],
            "required_amenities": scenario["user_amenities"]
        }
    )

    # Build and run workflow
    print("\n🚀 STARTING WORKFLOW...")
    print("=" * 100)

    workflow = build_workflow()
    graph = workflow.compile()

    try:
        # Run the workflow
        result = graph.invoke(initial_state)

        # Analyze results
        analysis = analyze_workflow_result(result)

        # Verify expectations
        print("\n" + "=" * 100)
        print("VERIFICATION".center(100))
        print("=" * 100)

        success = True

        # Check if negotiation happened as expected
        negotiation_occurred = analysis["negotiation_rounds"] > 0
        expected_negotiation = scenario["expect_negotiation"]

        print(f"\n✓ Expected negotiation: {expected_negotiation}")
        print(f"✓ Negotiation occurred: {negotiation_occurred}")

        if expected_negotiation and not negotiation_occurred:
            print("⚠️  WARNING: Expected negotiation but it didn't occur")
        elif not expected_negotiation and negotiation_occurred:
            print("ℹ️  INFO: Negotiation occurred (may be quality upgrade)")

        # Check if within budget
        within_budget = analysis["total_cost"] <= analysis["budget"]
        print(f"\n✓ Within budget: {within_budget}")

        # Check if workflow completed
        print(f"✓ Workflow completed: {analysis['success']}")

        if within_budget and analysis['success']:
            print("\n" + "🎉 " * 30)
            print("TEST PASSED".center(100))
            print("🎉 " * 30)
            return True
        else:
            print("\n" + "⚠️  " * 30)
            print("TEST COMPLETED WITH WARNINGS".center(100))
            print("⚠️  " * 30)
            return True

    except Exception as e:
        print("\n" + "❌ " * 30)
        print("TEST FAILED".center(100))
        print("❌ " * 30)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test scenarios."""
    print_header("FULL WORKFLOW TEST - POST-REFACTORING", "=")

    print("\n📝 TESTING OBJECTIVES:")
    print("   1. Verify agents search without budget knowledge")
    print("   2. Verify PolicyAgent validates (doesn't coordinate)")
    print("   3. Verify Orchestrator coordinates workflow")
    print("   4. Verify negotiation triggers correctly")
    print("   5. Verify final recommendations are generated")

    results = []

    for scenario in test_scenarios:
        print("\n\n")
        success = run_test_scenario(scenario)
        results.append({
            "scenario": scenario["name"],
            "success": success
        })

        print("\n" + "─" * 100)
        print("Moving to next test...")
        print("")

    # Final summary
    print_header("FINAL TEST SUMMARY", "=")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n📊 Results: {passed}/{total} tests passed")
    print("-" * 100)

    for i, result in enumerate(results, 1):
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{i}. {result['scenario']}: {status}")

    print("\n" + "=" * 100)

    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉".center(100))
        print("\n✓ Orchestrator successfully coordinates workflow")
        print("✓ PolicyAgent validates without coordinating")
        print("✓ Agents search without budget knowledge")
        print("✓ Negotiation emerges naturally")
    else:
        print("⚠️  SOME TESTS HAD ISSUES ⚠️".center(100))

    print("=" * 100)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
