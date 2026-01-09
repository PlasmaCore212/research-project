#!/usr/bin/env python3
"""Minimal test to verify the trip planning system works."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import plan_trip

def run_test(name, **kwargs):
    """Run a single test and print result."""
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print('#'*60)
    
    try:
        result = plan_trip(**kwargs)
        
        if not result:
            print("❌ No result returned!")
            return False
        
        for node_name, node_state in result.items():
            if "final_recommendation" in node_state:
                rec = node_state["final_recommendation"]
                flight = rec.get("flight", {})
                hotel = rec.get("hotel", {})
                
                if flight and hotel:
                    print(f"\n✅ SUCCESS")
                    print(f"   ✈️  {flight.get('airline', 'N/A')} - ${flight.get('price_usd', 'N/A')}")
                    print(f"   🏨 {hotel.get('name', 'N/A')} ({hotel.get('stars', '?')}★) - ${hotel.get('price_per_night_usd', 'N/A')}/night")
                    print(f"   💰 Total: ${rec.get('total_estimated_cost', 'N/A')}")
                    return True
                else:
                    print(f"\n⚠️  Partial result - missing flight or hotel")
                    return False
        
        print("❌ No final recommendation found")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("   QUICK SYSTEM TEST")
    print("="*60)
    
    # Test 1: Simple trip with good budget
    test1 = run_test(
        "TEST 1: BOS → CHI ($1500 budget)",
        origin="BOS",
        destination="CHI",
        departure_date="2026-02-20",
        return_date="2026-02-22",
        budget=1500
    )
    
    # Test 2: Tight budget
    test2 = run_test(
        "TEST 2: SF → BOS ($700 tight budget)", 
        origin="SF",
        destination="BOS",
        departure_date="2026-03-15",
        return_date="2026-03-16",
        budget=700
    )
    
    print("\n" + "="*60)
    print("   SUMMARY")
    print("="*60)
    print(f"  Test 1: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Test 2: {'✅ PASS' if test2 else '❌ FAIL'}")
    print("="*60)
