#!/usr/bin/env python3
"""
Quick test scenarios to verify the trip planning system works.
Tests both successful and edge case scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import plan_trip

def print_result(result, scenario_name):
    """Print the result of a scenario."""
    print(f"\n{'='*60}")
    print(f"RESULT: {scenario_name}")
    print('='*60)
    
    if not result:
        print("❌ No result returned!")
        return False
    
    success = False
    for node_name, node_state in result.items():
        if "final_recommendation" in node_state:
            rec = node_state["final_recommendation"]
            flight = rec.get("flight", {})
            hotel = rec.get("hotel", {})
            
            if flight and hotel:
                print(f"✅ SUCCESS")
                print(f"   ✈️  Flight: {flight.get('airline', 'N/A')} - ${flight.get('price_usd', 'N/A')}")
                print(f"   🏨 Hotel: {hotel.get('name', 'N/A')} ({hotel.get('stars', '?')}★)")
                print(f"   💰 Total: ${rec.get('total_estimated_cost', 'N/A')}")
                print(f"   📊 Status: {rec.get('compliance_status', 'unknown')}")
                success = True
            else:
                print(f"⚠️  PARTIAL - Missing flight or hotel")
                success = False
                
        if "metrics" in node_state:
            m = node_state["metrics"]
            print(f"   📈 Negotiation rounds: {m.get('negotiation_rounds', 0)}")
            print(f"   📨 Messages: {m.get('message_exchanges', 0)}")
    
    return success


def test_scenario_1():
    """Test 1: Standard business trip CHI→BOS with good budget"""
    print("\n" + "#"*70)
    print("# TEST 1: Chicago → Boston, $2000 budget (should succeed easily)")
    print("#"*70)
    
    result = plan_trip(
        origin="CHI",
        destination="BOS",
        departure_date="2026-02-15",
        return_date="2026-02-17",
        hotel_checkin="2026-02-15",
        hotel_checkout="2026-02-17",
        meeting_date="2026-02-16",
        meeting_time="10:00",
        meeting_coordinates={"lat": 42.3601, "lon": -71.0589},  # Boston
        budget=2000
    )
    
    return print_result(result, "CHI→BOS Standard Trip")


def test_scenario_2():
    """Test 2: Tight budget scenario"""
    print("\n" + "#"*70)
    print("# TEST 2: SF → CHI, $800 budget (tight budget - may need negotiation)")
    print("#"*70)
    
    result = plan_trip(
        origin="SF",
        destination="CHI",
        departure_date="2026-03-10",
        return_date="2026-03-12",
        hotel_checkin="2026-03-10",
        hotel_checkout="2026-03-12",
        budget=800
    )
    
    return print_result(result, "SF→CHI Tight Budget")


def test_scenario_3():
    """Test 3: No meeting time - simple trip"""
    print("\n" + "#"*70)
    print("# TEST 3: BOS → SF, $1500 budget, no meeting (simple trip)")
    print("#"*70)
    
    result = plan_trip(
        origin="BOS",
        destination="SF",
        departure_date="2026-04-01",
        return_date="2026-04-03",
        budget=1500
    )
    
    return print_result(result, "BOS→SF Simple Trip")


def test_scenario_4():
    """Test 4: Very low budget - should still return something"""
    print("\n" + "#"*70)
    print("# TEST 4: CHI → SF, $400 budget (very low - expect budget exceeded)")
    print("#"*70)
    
    result = plan_trip(
        origin="CHI",
        destination="SF",
        departure_date="2026-05-20",
        return_date="2026-05-21",
        hotel_checkin="2026-05-20",
        hotel_checkout="2026-05-21",
        budget=400
    )
    
    return print_result(result, "CHI→SF Very Low Budget")


def test_scenario_5():
    """Test 5: Early meeting that might cause time conflicts"""
    print("\n" + "#"*70)
    print("# TEST 5: BOS → CHI, early meeting at 09:00 (potential time conflict)")
    print("#"*70)
    
    result = plan_trip(
        origin="BOS",
        destination="CHI",
        departure_date="2026-06-15",
        return_date="2026-06-16",
        hotel_checkin="2026-06-15",
        hotel_checkout="2026-06-16",
        meeting_date="2026-06-15",
        meeting_time="09:00",  # Very early - might conflict with arrival
        meeting_coordinates={"lat": 41.8781, "lon": -87.6298},  # Chicago
        budget=1800
    )
    
    return print_result(result, "BOS→CHI Early Meeting")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("   TRIP PLANNING SYSTEM - QUICK TEST SUITE")
    print("="*70)
    print("Running 5 test scenarios to verify system functionality...")
    
    results = []
    
    # Run tests
    results.append(("CHI→BOS Standard", test_scenario_1()))
    results.append(("SF→CHI Tight Budget", test_scenario_2()))
    results.append(("BOS→SF Simple", test_scenario_3()))
    results.append(("CHI→SF Very Low Budget", test_scenario_4()))
    results.append(("BOS→CHI Early Meeting", test_scenario_5()))
    
    # Summary
    print("\n" + "="*70)
    print("   TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} scenarios completed successfully")
    print("="*70)
