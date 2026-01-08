from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.policy_agent import PolicyAgent
from agents.models import FlightQuery, HotelQuery


def test_compliant_trip():
    """Test a trip that should pass policy"""
    print("=" * 60)
    print("TEST 1: Compliant Trip")
    print("=" * 60)
    
    # Search flights
    flight_agent = FlightAgent()
    flight_query = FlightQuery(
        from_city="NYC",
        to_city="SF",
        max_price=700,
        departure_after="09:00",
        departure_before="17:00"
    )
    flight_result = flight_agent.search_flights(flight_query)
    
    # Search hotels
    hotel_agent = HotelAgent()
    hotel_query = HotelQuery(
        city="SF",
        max_price_per_night=250,
        min_stars=3
    )
    hotel_result = hotel_agent.search_hotels(hotel_query)
    
    # Check policy
    policy_agent = PolicyAgent()
    policy_result = policy_agent.check_compliance(
        flight_result, hotel_result, policy_name="standard"
    )
    
    print("\nFlight:")
    if flight_result.flights:
        f = flight_result.flights[0]
        print(f"  {f.flight_id} - {f.airline}: ${f.price_usd}")
    
    print("\nHotel:")
    if hotel_result.hotels:
        h = hotel_result.hotels[0]
        print(f"  {h.hotel_id} - {h.name}: ${h.price_per_night_usd}/night, {h.stars}*")
    
    print(f"\nPolicy Compliance: {'✓ PASS' if policy_result.is_compliant else '✗ FAIL'}")
    print(f"\nReasoning:\n{policy_result.reasoning}")


def test_violation_trip():
    """Test a trip that should violate policy"""
    print("\n" + "=" * 60)
    print("TEST 2: Policy Violation Trip")
    print("=" * 60)
    
    # Search flights (expensive)
    flight_agent = FlightAgent()
    flight_query = FlightQuery(
        from_city="NYC",
        to_city="SF",
        max_price=2000,  # Allow expensive flights
        departure_after="06:00",
        departure_before="22:00"
    )
    flight_result = flight_agent.search_flights(flight_query)
    
    # Search hotels (luxury)
    hotel_agent = HotelAgent()
    hotel_query = HotelQuery(
        city="SF",
        max_price_per_night=600,  # Allow expensive hotels
        min_stars=5
    )
    hotel_result = hotel_agent.search_hotels(hotel_query)
    
    # Check against strict policy
    policy_agent = PolicyAgent()
    policy_result = policy_agent.check_compliance(
        flight_result, hotel_result, policy_name="strict"
    )
    
    print("\nFlight:")
    if flight_result.flights:
        f = flight_result.flights[0]
        print(f"  {f.flight_id} - {f.airline}: ${f.price_usd}")
    
    print("\nHotel:")
    if hotel_result.hotels:
        h = hotel_result.hotels[0]
        print(f"  {h.hotel_id} - {h.name}: ${h.price_per_night_usd}/night, {h.stars}*")
    
    print(f"\nPolicy Compliance: {'✓ PASS' if policy_result.is_compliant else '✗ FAIL'}")
    
    if policy_result.violations:
        print(f"\nViolations ({len(policy_result.violations)}):")
        for v in policy_result.violations:
            print(f"  [{v.severity.upper()}] {v.message}")
            print(f"    Actual: {v.actual_value}, Expected: {v.expected_value}")
    
    print(f"\nReasoning:\n{policy_result.reasoning}")


if __name__ == "__main__":
    test_compliant_trip()
    test_violation_trip()