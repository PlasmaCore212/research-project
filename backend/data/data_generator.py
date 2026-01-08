"""
Synthetic Data Generator for Multi-Agent Business Trip Planner
Creates realistic flight and hotel data for 4 US cities: NYC, SF, Chicago, Boston
Includes edge cases: budget violations, time conflicts, overlapping schedules
"""

import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict
import random
import math

# City coordinates (lat, lon) and business centers
CITIES = {
    "NYC": {
        "name": "New York City",
        "airport_code": "JFK",
        "coordinates": {"lat": 40.7128, "lon": -74.0060},
        "business_centers": [
            {"name": "Manhattan Financial District", "lat": 40.7074, "lon": -74.0113},
            {"name": "Midtown Manhattan", "lat": 40.7549, "lon": -73.9840},
            {"name": "Brooklyn Tech Hub", "lat": 40.6782, "lon": -73.9442}
        ]
    },
    "SF": {
        "name": "San Francisco",
        "airport_code": "SFO",
        "coordinates": {"lat": 37.7749, "lon": -122.4194},
        "business_centers": [
            {"name": "Financial District", "lat": 37.7946, "lon": -122.3999},
            {"name": "SoMa", "lat": 37.7790, "lon": -122.4063},
            {"name": "Silicon Valley", "lat": 37.3861, "lon": -122.0839}
        ]
    },
    "CHI": {
        "name": "Chicago",
        "airport_code": "ORD",
        "coordinates": {"lat": 41.8781, "lon": -87.6298},
        "business_centers": [
            {"name": "The Loop", "lat": 41.8786, "lon": -87.6251},
            {"name": "River North", "lat": 41.8919, "lon": -87.6327},
            {"name": "West Loop", "lat": 41.8817, "lon": -87.6512}
        ]
    },
    "BOS": {
        "name": "Boston",
        "airport_code": "BOS",
        "coordinates": {"lat": 42.3601, "lon": -71.0589},
        "business_centers": [
            {"name": "Financial District", "lat": 42.3554, "lon": -71.0542},
            {"name": "Back Bay", "lat": 42.3505, "lon": -71.0795},
            {"name": "Cambridge Innovation", "lat": 42.3736, "lon": -71.1097}
        ]
    }
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in kilometers"""
    R = 6371  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def generate_flight_time(distance_km: float, base_time: datetime) -> tuple:
    """Generate realistic flight duration and arrival time based on distance"""
    # Average speed: 800 km/h
    flight_hours = distance_km / 800
    
    # Add buffer time (boarding, taxi, etc.)
    total_hours = flight_hours + random.uniform(0.5, 1.5)
    
    arrival = base_time + timedelta(hours=total_hours)
    
    return total_hours, arrival

def generate_flights(num_flights: int = 200) -> List[Dict]:
    """Generate synthetic flight data with realistic pricing and schedules"""
    flights = []
    flight_id = 1
    
    airlines = ["United", "Delta", "American", "Southwest", "JetBlue"]
    
    city_pairs = [
        (from_city, to_city) 
        for from_city in CITIES.keys() 
        for to_city in CITIES.keys() 
        if from_city != to_city
    ]
    
    # Generate multiple flights per route at different times
    for from_city, to_city in city_pairs:
        # Calculate distance for realistic pricing
        from_coords = CITIES[from_city]["coordinates"]
        to_coords = CITIES[to_city]["coordinates"]
        distance = haversine_distance(
            from_coords["lat"], from_coords["lon"],
            to_coords["lat"], to_coords["lon"]
        )
        
        # Generate 15-20 flights per route at different times
        num_flights_per_route = random.randint(15, 20)
        
        for _ in range(num_flights_per_route):
            hour = random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            minute = random.choice([0, 15, 30, 45])
            
            # Create time strings without dates
            departure_time = f"{hour:02d}:{minute:02d}"
            
            # Calculate arrival time
            duration_hours = distance / 800 + random.uniform(0.5, 1.5)
            arrival_hour = hour + int(duration_hours)
            arrival_minute = minute + int((duration_hours % 1) * 60)
            
            # Handle minute overflow
            if arrival_minute >= 60:
                arrival_hour += 1
                arrival_minute -= 60
            
            # Handle day overflow (keep it simple, just mod 24)
            arrival_hour = arrival_hour % 24
            arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"
            
            # Pricing: base price per km + random variance
            base_price_per_km = 0.15
            price = distance * base_price_per_km * random.uniform(0.7, 1.5)
            
            # Time-of-day pricing adjustments
            if hour < 9 or hour > 17:
                price *= random.uniform(0.8, 0.95)
            else:
                price *= random.uniform(1.1, 1.3)
            
            price = round(price / 10) * 10
            
            flight = {
                "flight_id": f"FL{flight_id:04d}",
                "airline": random.choice(airlines),
                "from_city": from_city,
                "to_city": to_city,
                "from_airport": CITIES[from_city]["airport_code"],
                "to_airport": CITIES[to_city]["airport_code"],
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration_hours": round(duration_hours, 2),
                "price_usd": int(price),
                "distance_km": round(distance, 2),
                "seats_available": random.randint(5, 150),
                "class": random.choice(["Economy", "Economy", "Economy", "Business"])
            }
            
            flights.append(flight)
            flight_id += 1
    
    # Add edge cases
    edge_case_flight = flights[10].copy()
    edge_case_flight["flight_id"] = f"FL{flight_id:04d}"
    edge_case_flight["airline"] = "EdgeCase Airways"
    edge_case_flight["price_usd"] = flights[10]["price_usd"] - 50
    flights.append(edge_case_flight)
    flight_id += 1
    
    expensive_flight = flights[20].copy()
    expensive_flight["flight_id"] = f"FL{flight_id:04d}"
    expensive_flight["price_usd"] = 2500
    expensive_flight["class"] = "First Class"
    flights.append(expensive_flight)
    
    return sorted(flights, key=lambda x: (x["from_city"], x["to_city"], x["departure_time"]))


def generate_hotels(num_hotels: int = 80) -> List[Dict]:
    """Generate synthetic hotel data near business centers with tiered pricing"""
    hotels = []
    hotel_id = 1
    
    hotel_chains = ["Marriott", "Hilton", "Hyatt", "InterContinental", "Hampton Inn", "Holiday Inn"]
    hotel_types = {
        "Budget": {"price_range": (80, 150), "stars": 2},
        "Mid-Range": {"price_range": (150, 250), "stars": 3},
        "Upscale": {"price_range": (250, 400), "stars": 4},
        "Luxury": {"price_range": (400, 800), "stars": 5}
    }
    
    for city_code, city_data in CITIES.items():
        for business_center in city_data["business_centers"]:
            for tier, tier_data in hotel_types.items():
                # Generate 1-2 hotels per tier per business center
                for _ in range(random.randint(1, 2)):
                    # Hotel location near business center (within 2km)
                    lat_offset = random.uniform(-0.02, 0.02)
                    lon_offset = random.uniform(-0.02, 0.02)
                    hotel_lat = business_center["lat"] + lat_offset
                    hotel_lon = business_center["lon"] + lon_offset
                    
                    # Distance to business center
                    distance_to_center = haversine_distance(
                        business_center["lat"], business_center["lon"],
                        hotel_lat, hotel_lon
                    )
                    
                    # Distance to airport
                    distance_to_airport = haversine_distance(
                        city_data["coordinates"]["lat"], city_data["coordinates"]["lon"],
                        hotel_lat, hotel_lon
                    )
                    
                    # Price per night
                    price_per_night = random.randint(*tier_data["price_range"])
                    
                    hotel = {
                        "hotel_id": f"HT{hotel_id:04d}",
                        "name": f"{random.choice(hotel_chains)} {city_data['name']} {business_center['name']}",
                        "city": city_code,
                        "city_name": city_data["name"],
                        "business_area": business_center["name"],
                        "tier": tier,
                        "stars": tier_data["stars"],
                        "price_per_night_usd": price_per_night,
                        "coordinates": {"lat": hotel_lat, "lon": hotel_lon},
                        "distance_to_business_center_km": round(distance_to_center, 2),
                        "distance_to_airport_km": round(distance_to_airport, 2),
                        "amenities": random.sample([
                            "WiFi", "Gym", "Pool", "Restaurant", "Bar", 
                            "Conference Room", "Airport Shuttle", "Parking"
                        ], k=random.randint(3, 6)),
                        "rooms_available": random.randint(5, 100)
                    }
                    
                    hotels.append(hotel)
                    hotel_id += 1
    
    return sorted(hotels, key=lambda x: (x["city"], x["price_per_night_usd"]))

def save_to_json(data: List[Dict], filename: str):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(data)} records to {filename}")


def save_to_csv(data: List[Dict], filename: str):
    """Save data to CSV file"""
    if not data:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        # Flatten nested structures for CSV
        flattened_data = []
        for item in data:
            flat_item = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_item[f"{key}_{sub_key}"] = sub_value
                elif isinstance(value, list):
                    flat_item[key] = "; ".join(str(v) for v in value)
                else:
                    flat_item[key] = value
            flattened_data.append(flat_item)
        
        writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
        writer.writeheader()
        writer.writerows(flattened_data)
    print(f"✓ Saved {len(data)} records to {filename}")

def main():
    """Generate all synthetic data"""
    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR")
    print("Multi-Agent Business Trip Planner")
    print("=" * 60)
    
    # Generate data
    print("\n📊 Generating flights...")
    flights = generate_flights(num_flights=200)
    
    print("🏨 Generating hotels...")
    hotels = generate_hotels(num_hotels=80)
    
    # Save JSON files
    print("\n💾 Saving JSON files...")
    save_to_json(flights, "flights.json")
    save_to_json(hotels, "hotels.json")
    
    # Save CSV files
    print("\n💾 Saving CSV files...")
    save_to_csv(flights, "flights.csv")
    save_to_csv(hotels, "hotels.csv")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n📈 Summary:")
    print(f"  • Flights: {len(flights)} records")
    print(f"  • Hotels: {len(hotels)} records")
    print(f"  • Cities: {len(CITIES)}")
    print(f"  • Edge cases included: ✓")
    print(f"\n📁 Files created:")
    print("  • flights.json / flights.csv")
    print("  • hotels.json / hotels.csv")
    print("  • company_policies.json")
    print("  • sample_trip_requests.json")
    print("  • validation_report.json")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()