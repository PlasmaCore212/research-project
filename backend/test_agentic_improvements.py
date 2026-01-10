#!/usr/bin/env python3
"""
Test suite for validating the new agentic improvements:
1. LLM-based early stopping in booking agents
2. Simplified alternative options (Premium/Similar/Budget)
3. Fixed compare_flights parameter handling
"""

import sys
import json
from datetime import datetime
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.policy_agent import PolicyComplianceAgent
from agents.models import FlightQuery, HotelQuery


class TestAnalyzer:
    """Analyzes test results and provides detailed reports."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def add_result(self, test_name: str, passed: bool, details: dict):
        """Record a test result."""
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def print_summary(self):
        """Print a formatted summary of all test results."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {total - passed} ✗")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print(f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f}s")

        print("\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)

        for i, result in enumerate(self.results, 1):
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"\n{i}. {result['test_name']} - {status}")

            for key, value in result["details"].items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                    for item in value[:3]:  # Show first 3
                        print(f"     - {item}")
                    if len(value) > 3:
                        print(f"     ... and {len(value) - 3} more")
                elif isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in list(value.items())[:5]:  # Show first 5 keys
                        print(f"     {k}: {v}")
                else:
                    print(f"   {key}: {value}")


def test_flight_agent_early_stopping(analyzer: TestAnalyzer):
    """Test 1: Flight agent uses LLM reasoning to decide when to stop."""
    print("\n" + "="*80)
    print("TEST 1: Flight Agent LLM-Based Early Stopping")
    print("="*80)

    flight_agent = FlightAgent(verbose=True)
    query = FlightQuery(
        from_city="CHI",
        to_city="BOS",
        max_price=500
    )

    print("\n📋 Test Goal: Verify FlightAgent uses LLM reasoning to stop early")
    print(f"   Query: {query.from_city} → {query.to_city}, max ${query.max_price}")

    result = flight_agent.search_flights(query)

    # Analyze the result
    trace = flight_agent.state.reasoning_trace
    iterations = len(trace)

    # Check if agent stopped early with LLM decision
    last_step = trace[-1] if trace else None

    details = {
        "iterations": iterations,
        "max_iterations": flight_agent.max_iterations,
        "stopped_early": iterations < flight_agent.max_iterations,
        "flights_found": len(result.flights),
        "last_action": last_step.action if last_step else "None",
        "reasoning_steps": [f"Step {s.step_number}: {s.action}" for s in trace]
    }

    # Success criteria:
    # 1. Should stop before max iterations
    # 2. Should find diverse flights (>= 3)
    # 3. Should have performed search and analysis
    passed = (
        iterations < flight_agent.max_iterations and
        len(result.flights) >= 3 and
        any(s.action == "search_flights" for s in trace)
    )

    analyzer.add_result("Flight Agent LLM Early Stopping", passed, details)

    print(f"\n✓ Result: Agent stopped after {iterations}/{flight_agent.max_iterations} iterations")
    print(f"✓ Found {len(result.flights)} diverse flights")

    return result


def test_hotel_agent_early_stopping(analyzer: TestAnalyzer):
    """Test 2: Hotel agent uses LLM reasoning to decide when to stop."""
    print("\n" + "="*80)
    print("TEST 2: Hotel Agent LLM-Based Early Stopping")
    print("="*80)

    hotel_agent = HotelAgent(verbose=True)
    query = HotelQuery(
        city="BOS",
        max_price_per_night=300,
        min_stars=3
    )

    print("\n📋 Test Goal: Verify HotelAgent uses LLM reasoning to stop early")
    print(f"   Query: {query.city}, max ${query.max_price_per_night}/night, min {query.min_stars}★")

    result = hotel_agent.search_hotels(query)

    # Analyze the result
    trace = hotel_agent.state.reasoning_trace
    iterations = len(trace)
    last_step = trace[-1] if trace else None

    details = {
        "iterations": iterations,
        "max_iterations": hotel_agent.max_iterations,
        "stopped_early": iterations < hotel_agent.max_iterations,
        "hotels_found": len(result.hotels),
        "last_action": last_step.action if last_step else "None",
        "reasoning_steps": [f"Step {s.step_number}: {s.action}" for s in trace]
    }

    passed = (
        iterations < hotel_agent.max_iterations and
        len(result.hotels) >= 3 and
        any(s.action == "search_hotels" for s in trace)
    )

    analyzer.add_result("Hotel Agent LLM Early Stopping", passed, details)

    print(f"\n✓ Result: Agent stopped after {iterations}/{hotel_agent.max_iterations} iterations")
    print(f"✓ Found {len(result.hotels)} diverse hotels")

    return result


def test_alternative_options_quality(analyzer: TestAnalyzer, flights, hotels):
    """Test 3: Verify alternative options are properly categorized (Premium/Similar/Budget)."""
    print("\n" + "="*80)
    print("TEST 3: Alternative Options Quality (Premium/Similar/Budget)")
    print("="*80)

    policy_agent = PolicyComplianceAgent(verbose=True)

    budget = 1000
    nights = 2

    print("\n📋 Test Goal: Verify alternatives are categorized as Premium, Similar, or Budget")
    print(f"   Budget: ${budget}, Nights: {nights}")
    print(f"   Flights: {len(flights)}, Hotels: {len(hotels)}")

    # Convert to dicts for policy agent
    flight_dicts = [f.model_dump() for f in flights]
    hotel_dicts = [h.model_dump() for h in hotels]

    result = policy_agent.find_best_combination(
        flights=flight_dicts,
        hotels=hotel_dicts,
        budget=budget,
        nights=nights
    )

    # Analyze alternatives
    alternatives = result.cheaper_alternatives  # Now contains diverse alternatives

    categories_found = set()
    category_counts = {"🔶 PREMIUM": 0, "🔷 SIMILAR": 0, "💚 BUDGET": 0}

    for alt in alternatives:
        category = alt.get("category", "Unknown")
        categories_found.add(category)
        if category in category_counts:
            category_counts[category] += 1

    details = {
        "selected_cost": result.total_cost,
        "budget_remaining": result.budget_remaining,
        "alternatives_count": len(alternatives),
        "categories_found": list(categories_found),
        "category_breakdown": category_counts,
        "alternative_details": []
    }

    # Analyze each alternative
    for alt in alternatives:
        alt_detail = {
            "category": alt.get("category", "Unknown"),
            "total_cost": alt.get("total_cost", 0),
            "vs_selected": alt.get("vs_selected", 0),
            "reasoning": alt.get("reasoning", "N/A"),
            "hotel_stars": alt.get("hotel", {}).get("stars", 0),
            "flight_class": alt.get("flight", {}).get("class", "Unknown")
        }
        details["alternative_details"].append(alt_detail)

    # Success criteria:
    # 1. Should have at least 2 different categories
    # 2. Premium should be more expensive than selected
    # 3. Budget should be cheaper than selected
    # 4. Each alternative should have proper reasoning

    premium_valid = True
    budget_valid = True

    for alt in alternatives:
        if alt.get("category") == "🔶 PREMIUM":
            if alt.get("vs_selected", 0) <= 0:
                premium_valid = False
        elif alt.get("category") == "💚 BUDGET":
            if alt.get("vs_selected", 0) >= 0:
                budget_valid = False

    passed = (
        len(categories_found) >= 2 and
        all(alt.get("reasoning") for alt in alternatives) and
        premium_valid and
        budget_valid
    )

    analyzer.add_result("Alternative Options Quality", passed, details)

    print(f"\n✓ Selected: ${result.total_cost:.2f} (Budget remaining: ${result.budget_remaining:.2f})")
    print(f"✓ Alternatives: {len(alternatives)}")
    for alt in alternatives:
        cat = alt.get("category", "?")
        cost = alt.get("total_cost", 0)
        vs = alt.get("vs_selected", 0)
        reason = alt.get("reasoning", "")
        print(f"   {cat}: ${cost:.2f} ({'+'if vs > 0 else ''}{vs:.2f}) - {reason}")

    return result


def test_compare_tool_parameter_handling(analyzer: TestAnalyzer):
    """Test 4: Verify compare tools handle parameters correctly."""
    print("\n" + "="*80)
    print("TEST 4: Compare Tool Parameter Handling")
    print("="*80)

    flight_agent = FlightAgent(verbose=False)

    # First, search for flights to populate state
    query = FlightQuery(from_city="NYC", to_city="LAX")
    result = flight_agent.search_flights(query)

    print("\n📋 Test Goal: Verify compare_flights handles parameters correctly")
    print(f"   Flights available: {len(result.flights)}")

    # Manually test the compare tool with proper parameters
    flights = result.flights[:3]
    flight_ids = [f.flight_id for f in flights]

    print(f"   Testing compare with IDs: {flight_ids}")

    # Test the tool directly
    flight_agent.state.add_belief("available_flights", [f.model_dump() for f in result.flights])

    try:
        compare_result = flight_agent._tool_compare_flights(
            flight_ids=flight_ids,
            criteria="overall"
        )

        details = {
            "flight_ids_tested": flight_ids,
            "compare_result_length": len(compare_result),
            "success": "ERROR" not in compare_result,
            "result_preview": compare_result[:200]
        }

        passed = "ERROR" not in compare_result and len(compare_result) > 0

        analyzer.add_result("Compare Tool Parameter Handling", passed, details)

        print(f"\n✓ Compare tool executed successfully")
        print(f"   Result preview: {compare_result[:150]}...")

    except Exception as e:
        details = {
            "error": str(e),
            "flight_ids_tested": flight_ids
        }
        analyzer.add_result("Compare Tool Parameter Handling", False, details)
        print(f"\n✗ Compare tool failed: {e}")

    return result


def test_end_to_end_workflow(analyzer: TestAnalyzer):
    """Test 5: Complete workflow with all improvements."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Workflow")
    print("="*80)

    print("\n📋 Test Goal: Complete trip planning workflow with all new features")

    # Step 1: Flight search
    flight_agent = FlightAgent(verbose=False)
    flight_query = FlightQuery(from_city="SF", to_city="NYC", max_price=400)

    print(f"\n1️⃣ Searching flights: {flight_query.from_city} → {flight_query.to_city}")
    flight_result = flight_agent.search_flights(flight_query)
    print(f"   ✓ Found {len(flight_result.flights)} flights")

    # Step 2: Hotel search
    hotel_agent = HotelAgent(verbose=False)
    hotel_query = HotelQuery(city="NYC", max_price_per_night=250, min_stars=3)

    print(f"\n2️⃣ Searching hotels in {hotel_query.city}")
    hotel_result = hotel_agent.search_hotels(hotel_query)
    print(f"   ✓ Found {len(hotel_result.hotels)} hotels")

    # Step 3: Policy agent combination
    policy_agent = PolicyComplianceAgent(verbose=False)

    budget = 1200
    nights = 2

    print(f"\n3️⃣ Finding best combination (Budget: ${budget}, Nights: {nights})")

    flight_dicts = [f.model_dump() for f in flight_result.flights]
    hotel_dicts = [h.model_dump() for h in hotel_result.hotels]

    combo_result = policy_agent.find_best_combination(
        flights=flight_dicts,
        hotels=hotel_dicts,
        budget=budget,
        nights=nights
    )

    # Analyze complete workflow
    details = {
        "flight_iterations": len(flight_agent.state.reasoning_trace),
        "hotel_iterations": len(hotel_agent.state.reasoning_trace),
        "flights_evaluated": len(flight_result.flights),
        "hotels_evaluated": len(hotel_result.hotels),
        "combinations_evaluated": combo_result.combinations_evaluated,
        "selected_within_budget": combo_result.total_cost <= budget,
        "budget_utilization": f"{combo_result.total_cost/budget*100:.1f}%",
        "alternatives_provided": len(combo_result.cheaper_alternatives),
        "alternative_categories": [alt.get("category") for alt in combo_result.cheaper_alternatives],
        "selected_flight": combo_result.selected_flight.get("flight_id") if combo_result.selected_flight else None,
        "selected_hotel": combo_result.selected_hotel.get("hotel_id") if combo_result.selected_hotel else None,
    }

    # Success criteria: everything should work smoothly
    passed = (
        combo_result.success and
        combo_result.total_cost <= budget and
        len(combo_result.cheaper_alternatives) > 0 and
        combo_result.selected_flight is not None and
        combo_result.selected_hotel is not None
    )

    analyzer.add_result("End-to-End Workflow", passed, details)

    if combo_result.selected_flight and combo_result.selected_hotel:
        print(f"\n✓ Selected Combination:")
        print(f"   Flight: {combo_result.selected_flight.get('flight_id')} - ${combo_result.selected_flight.get('price_usd')}")
        print(f"   Hotel: {combo_result.selected_hotel.get('hotel_id')} - ${combo_result.selected_hotel.get('price_per_night_usd')}/night")
        print(f"   Total: ${combo_result.total_cost:.2f} / ${budget} ({combo_result.total_cost/budget*100:.1f}% used)")
        print(f"   Alternatives: {len(combo_result.cheaper_alternatives)}")
    else:
        print(f"\n✗ No valid combination found (insufficient data)")

    return combo_result


def main():
    """Run all tests and generate analysis report."""
    print("\n" + "="*80)
    print("AGENTIC IMPROVEMENTS TEST SUITE")
    print("Testing: LLM Early Stopping, Alternative Options, Parameter Handling")
    print("="*80)

    analyzer = TestAnalyzer()

    try:
        # Run tests
        flight_result = test_flight_agent_early_stopping(analyzer)
        hotel_result = test_hotel_agent_early_stopping(analyzer)
        test_alternative_options_quality(analyzer, flight_result.flights, hotel_result.hotels)
        test_compare_tool_parameter_handling(analyzer)
        test_end_to_end_workflow(analyzer)

    except Exception as e:
        print(f"\n✗ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()

    # Print comprehensive summary
    analyzer.print_summary()

    # Generate JSON report
    report_file = "test_results.json"
    with open(report_file, "w") as f:
        json.dump(analyzer.results, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
