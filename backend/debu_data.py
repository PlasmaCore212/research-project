from data.loaders import FlightDataLoader
from datetime import datetime

loader = FlightDataLoader()

# Check all NYC->SF flights
nyc_sf_flights = [f for f in loader.flights 
                   if f["from_city"] == "NYC" and f["to_city"] == "SF"]

print(f"Total NYC->SF flights: {len(nyc_sf_flights)}")
print("\nFirst 5 flights:")
for flight in nyc_sf_flights[:5]:
    dt = datetime.fromisoformat(flight["departure_time"])
    print(f"  {flight['flight_id']}: {dt.date()} at {dt.hour:02d}:{dt.minute:02d} - ${flight['price_usd']}")

# Check date range
dates = [datetime.fromisoformat(f["departure_time"]).date() for f in loader.flights]
print(f"\nDate range: {min(dates)} to {max(dates)}")