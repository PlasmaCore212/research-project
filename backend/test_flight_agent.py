from agents.flight_agent import FlightAgent
from agents.models import FlightQuery

agent = FlightAgent()

# Test 1: Morning departure window
print("=" * 60)
print("TEST 1: Early morning flights (06:00-09:00)")
print("=" * 60)

query1 = FlightQuery(
    from_city="NYC",
    to_city="SF",
    max_price=700,
    departure_after="06:00",
    departure_before="09:00"
)

result1 = agent.search_flights(query1)
print(result1.reasoning)
print(f"\nFound {len(result1.flights)} flights:")
for flight in result1.flights:
    print(f"  {flight.flight_id}: {flight.airline} - Departs {flight.departure_time} - ${flight.price_usd}")

# Test 2: Late afternoon departure
print("\n" + "=" * 60)
print("TEST 2: Late afternoon flights (after 16:00)")
print("=" * 60)

query2 = FlightQuery(
    from_city="BOS",
    to_city="CHI",
    max_price=500,
    departure_after="16:00"
)

result2 = agent.search_flights(query2)
print(result2.reasoning)
print(f"\nFound {len(result2.flights)} flights:")
for flight in result2.flights:
    print(f"  {flight.flight_id}: {flight.airline} - Departs {flight.departure_time} - ${flight.price_usd}")

# Test 3: Budget-conscious, any time
print("\n" + "=" * 60)
print("TEST 3: Budget flights, any time (max $400)")
print("=" * 60)

query3 = FlightQuery(
    from_city="CHI",
    to_city="NYC",
    max_price=400
)

result3 = agent.search_flights(query3)
print(result3.reasoning)
print(f"\nFound {len(result3.flights)} flights:")
for flight in result3.flights:
    print(f"  {flight.flight_id}: {flight.airline} - Departs {flight.departure_time} - ${flight.price_usd}")