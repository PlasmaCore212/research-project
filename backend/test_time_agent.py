# backend/test_time_agent.py
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.time_agent import TimeManagementAgent
from agents.models import FlightQuery, HotelQuery, Meeting
from data.loaders import CITIES


def test_feasible_timeline():
    """Test a timeline that should work"""
    print("=" * 60)
    print("TEST 1: Feasible Timeline")
    print("=" * 60)
    
    # Get flights arriving early
    flight_agent = FlightAgent()
    flight_result = flight_agent.search_flights(FlightQuery(
        from_city="NYC",
        to_city="SF",
        max_price=700,
        departure_after="06:00",
        departure_before="09:00"
    ))
    
    # Get hotel
    hotel_agent = HotelAgent()
    hotel_result = hotel_agent.search_hotels(HotelQuery(
        city="SF",
        max_price_per_night=300,
        min_stars=3
    ))
    
    # Meeting in afternoon (plenty of time)
    meetings = [
        Meeting(
            date="2025-01-15",
            time="16:00",
            location={"lat": 37.7946, "lon": -122.3999},  # Financial District
            duration_minutes=60,
            description="Client meeting"
        )
    ]
    
    # Check timeline
    time_agent = TimeManagementAgent()
    sf_coords = CITIES["SF"]["coordinates"]
    
    result = time_agent.check_feasibility(
        flight_result,
        hotel_result,
        meetings,
        sf_coords,
        sf_coords,  # Using city center as airport proxy
        departure_date="2025-01-15"
    )
    
    print(f"\nFlight: {flight_result.flights[0].flight_id} arrives {flight_result.flights[0].arrival_time}")
    print(f"Hotel: {hotel_result.hotels[0].hotel_id}")
    print(f"Meeting: 16:00 at Financial District")
    
    print(f"\n{'✓ FEASIBLE' if result.is_feasible else '✗ NOT FEASIBLE'}")
    print(f"\nTimeline:")
    for key, val in result.timeline.items():
        print(f"  {key}: {val}")
    
    if result.conflicts:
        print(f"\nConflicts ({len(result.conflicts)}):")
        for c in result.conflicts:
            print(f"  [{c.severity.upper()}] {c.message}")
    
    print(f"\n{result.reasoning}")


def test_tight_timeline():
    """Test a timeline with tight timing"""
    print("\n" + "=" * 60)
    print("TEST 2: Tight Timeline")
    print("=" * 60)
    
    # Get later flights
    flight_agent = FlightAgent()
    flight_result = flight_agent.search_flights(FlightQuery(
        from_city="NYC",
        to_city="SF",
        max_price=700,
        departure_after="10:00",
        departure_before="13:00"
    ))
    
    # Get hotel
    hotel_agent = HotelAgent()
    hotel_result = hotel_agent.search_hotels(HotelQuery(
        city="SF",
        max_price_per_night=300
    ))
    
    # Meeting shortly after likely arrival
    meetings = [
        Meeting(
            date="2025-01-15",
            time="15:00",  # Might be tight
            location={"lat": 37.7946, "lon": -122.3999},
            duration_minutes=90
        )
    ]
    
    time_agent = TimeManagementAgent()
    sf_coords = CITIES["SF"]["coordinates"]
    
    result = time_agent.check_feasibility(
        flight_result,
        hotel_result,
        meetings,
        sf_coords,
        sf_coords,
        departure_date="2025-01-15"
    )
    
    print(f"\nFlight: {flight_result.flights[0].flight_id} arrives {flight_result.flights[0].arrival_time}")
    print(f"Meeting: 15:00")
    
    print(f"\n{'✓ FEASIBLE' if result.is_feasible else '✗ NOT FEASIBLE'}")
    
    if result.conflicts:
        print(f"\nConflicts:")
        for c in result.conflicts:
            print(f"  [{c.severity.upper()}] {c.message}")


def test_impossible_timeline():
    """Test an impossible timeline"""
    print("\n" + "=" * 60)
    print("TEST 3: Impossible Timeline")
    print("=" * 60)
    
    # Get late flights
    flight_agent = FlightAgent()
    flight_result = flight_agent.search_flights(FlightQuery(
        from_city="NYC",
        to_city="SF",
        max_price=700,
        departure_after="15:00"
    ))
    
    hotel_agent = HotelAgent()
    hotel_result = hotel_agent.search_hotels(HotelQuery(
        city="SF",
        max_price_per_night=300
    ))
    
    # Meeting in early afternoon (impossible)
    meetings = [
        Meeting(
            date="2025-01-15",
            time="14:00",
            location={"lat": 37.7946, "lon": -122.3999}
        )
    ]
    
    time_agent = TimeManagementAgent()
    sf_coords = CITIES["SF"]["coordinates"]
    
    result = time_agent.check_feasibility(
        flight_result,
        hotel_result,
        meetings,
        sf_coords,
        sf_coords,
        departure_date="2025-01-15"
    )
    
    print(f"\nFlight arrives: {flight_result.flights[0].arrival_time}")
    print(f"Meeting: 14:00")
    print(f"\n{'✓ FEASIBLE' if result.is_feasible else '✗ NOT FEASIBLE'}")
    print(f"\nReason: {result.conflicts[0].message if result.conflicts else 'N/A'}")


if __name__ == "__main__":
    test_feasible_timeline()
    test_tight_timeline()
    test_impossible_timeline()